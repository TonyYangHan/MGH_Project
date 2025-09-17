# =========================
# ======  CONFIG  =========
RAW_RNA_PATH   = "/mnt/d/Lab_project/project_0912/Resources/aci_raw_RNA_filtered_v2.h5ad"
RAW_ATAC_PATH  = "/mnt/d/Lab_project/project_0912/Resources/aci_raw_ATAC_filtered_v2_precomp.h5ad"

# donor column present in BOTH objects
DONOR_COL_RNA  = "orig.ident"
DONOR_COL_ATAC = "orig.ident"

# Keras models + AUROC for model selection
RNA_MODELS_DIR = "/mnt/d/Lab_project/project_0912/Resources/models_rna"
ATAC_MODELS_DIR= "/mnt/d/Lab_project/project_0912/Resources/models_atac"
RNA_AUROC_CSV  = "/mnt/d/Lab_project/project_0912/Resources/fold_auroc_rna.csv"
ATAC_AUROC_CSV = "/mnt/d/Lab_project/project_0912/Resources/fold_auroc_atac.csv"

# Outputs
OUT_DIR        = "/mnt/d/Lab_project/project_0912/Results/task2_donor_corr"
PLOTS_DIR      = OUT_DIR + "/plots"

# Preprocess params (must match training)
N_PCS          = 50
RNA_FRAC       = 1.0
RNA_FLAVOR     = "pearson_residuals"
ATAC_FRAC      = 1.0
USE_TFIDF      = False

# Averaging rule if best model predictions look “flat”
FLAT_STD_THRES = 0.06   # if std(best_probs) < this, average across all models

# Aggregate per-donor: "mean" or "median"
AGG_FUNC       = "mean"

# Candidate columns in RNA.obs that may encode AD/Control
STATUS_COL_CANDIDATES = ["AD", "ad", "Diagnosis", "diagnosis", "status", "Status", "group", "Group"]

# =========================
# ======  IMPORTS  ========
# =========================
import os, re, glob, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from scipy.stats import pearsonr

plt.rcParams["figure.dpi"] = 130

# =========================
# Safe-load ONLY preprocess_rna / preprocess_atac from preprocess.py
# =========================
def _safe_load_preprocess_funcs(preprocess_path="preprocess.py"):
    import ast, os
    wanted = {"preprocess_rna", "preprocess_atac"}
    with open(preprocess_path, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=preprocess_path)

    kept_nodes, seen = [], set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            kept_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted:
            kept_nodes.append(node); seen.add(node.name)

    missing = sorted(wanted - seen)
    if missing:
        present = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        raise KeyError(
            "Could not find expected functions in preprocess.py.\n"
            f"Missing: {missing}\nPresent: {sorted(set(present))}"
        )

    mod = ast.Module(body=kept_nodes, type_ignores=[])
    code = compile(mod, filename=os.path.abspath(preprocess_path), mode="exec")
    ns = {}
    exec(code, ns)
    return ns["preprocess_rna"], ns["preprocess_atac"]

_preprocess_rna, _preprocess_atac = _safe_load_preprocess_funcs("preprocess.py")
print("[INFO] Loaded preprocess functions from preprocess.py (top-level code skipped).")

# =========================
# ======  HELPERS  ========
# =========================
def ensure_outdir(p): os.makedirs(p, exist_ok=True)

def robust_read_h5ad(path):
    try:
        return sc.read_h5ad(path)
    except Exception as e:
        print(f"[WARN] read_h5ad failed for {path}: {e}")
        print("[INFO] Trying backed='r' rescue …")
        adata_b = sc.read_h5ad(path, backed="r")
        return adata_b.to_memory()

def features_from_adata(adata):
    if "X_pca" in adata.obsm: return np.asarray(adata.obsm["X_pca"])
    X = adata.X
    return X.A if hasattr(X, "A") else np.asarray(X)

def find_col(candidates, cols):
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low: return low[cand.lower()]
    return None

def _find_all_keras(d):
    return sorted(glob.glob(os.path.join(d, "**", "*.keras"), recursive=True))

def _fallback_model(d, strategy="mtime"):
    files = _find_all_keras(d)
    if not files: raise FileNotFoundError(f"No .keras files under {d}")
    files.sort(key=(lambda p: os.path.getmtime(p) if strategy=="mtime" else os.path.getsize(p)), reverse=True)
    return files[0]

def _parse_test_pair(val):
    if isinstance(val, (list, tuple)): return [str(x) for x in val]
    toks = re.findall(r"[A-Za-z0-9_-]+", str(val))
    nums = [t for t in toks if re.fullmatch(r"\d+", t)]
    return nums if nums else toks

def safe_read_auroc_table(path):
    df = pd.read_csv(path)
    if df.empty: raise ValueError(f"AUROC CSV empty: {path}")
    cand = [c for c in df.columns if c.lower() in ("auroc","roc_auc","val_auroc","mean_auroc","auroc_mean","auc","val_auc")]
    auroc_col = cand[0] if cand else next((c for c in df.columns if np.issubdtype(df[c].dtype, np.number)), None)
    if auroc_col is None: raise ValueError("No AUROC-like column found.")
    lower = {c.lower(): c for c in df.columns}
    pair_col = None
    for k in ("test_pair","pair","ids","model","model_id"):
        if k in lower: pair_col = lower[k]; break
    if pair_col is None:
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        if not obj_cols: raise ValueError("No pair/model column found.")
        pair_col = obj_cols[0]
    return df, auroc_col, pair_col

def _score_filename_for_tokens(basename, tokens):
    contained = [t for t in tokens if t in basename]
    score = (2.0 if (len(tokens) >= 2 and all(t in basename for t in tokens[:2])) else
             (1.0 if contained else 0.0))
    score += 0.1*len(contained) + 0.01*len(basename)
    return score

def pick_best_model_path(models_dir, auroc_csv):
    files = _find_all_keras(models_dir)
    if not files: raise FileNotFoundError(f"No .keras files under {models_dir}")
    basenames = [os.path.basename(p) for p in files]
    df, auroc_col, pair_col = safe_read_auroc_table(auroc_csv)
    df_sorted = df.sort_values(auroc_col, ascending=False).reset_index(drop=True)
    for _, row in df_sorted.iterrows():
        tokens = _parse_test_pair(row[pair_col])
        if not tokens: continue
        scored = []
        for p, b in zip(files, basenames):
            sc_ = _score_filename_for_tokens(b, tokens)
            if sc_ >= 1.0: scored.append((sc_, os.path.getmtime(p), p))
        if scored:
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return scored[0][2]
    print(f"[WARN] Could not map AUROC entries to files in {models_dir}. Using most recent.")
    return _fallback_model(models_dir)

def list_all_models(models_dir): return _find_all_keras(models_dir)

def extract_positive_class_proba(pred):
    arr = np.asarray(pred)
    if arr.ndim == 1: return arr
    if arr.ndim == 2 and arr.shape[1] == 1: return arr[:, 0]
    if arr.ndim == 2 and arr.shape[1] == 2: return arr[:, 1]
    raise ValueError(f"Unexpected prediction shape: {arr.shape}")

def model_predict_prob(model_path, X):
    model = load_model(model_path, compile=False)
    pred = model.predict(X, verbose=0)
    return extract_positive_class_proba(pred)

def avg_prediction_if_flat(best_probs, all_model_paths, X, flat_std_thres):
    if np.nanstd(best_probs) >= flat_std_thres:
        return best_probs, False
    probs = [model_predict_prob(p, X) for p in all_model_paths]
    return np.vstack(probs).mean(axis=0), True

# =========================
# ===== STATUS MAPPING ====
# =========================
def _normalize_status(value):
    """Map various encodings to 'AD' or 'Control'."""
    if pd.isna(value): return None
    s = str(value).strip().lower()
    ad_like = {"1", "ad", "case", "alz", "alzheimers", "alzheimer’s", "patient", "pos", "positive", "true", "yes"}
    ctrl_like = {"0", "control", "cn", "hc", "neg", "negative", "false", "no"}
    if s in ad_like: return "AD"
    if s in ctrl_like: return "Control"
    try:
        return "AD" if float(s) >= 0.5 else "Control"
    except Exception:
        return None

def build_status_map_from_rna(adata_rna, donor_col, candidates=STATUS_COL_CANDIDATES):
    status_col = next((c for c in adata_rna.obs.columns if c in candidates), None)
    if status_col is None:
        return {}
    tmp = adata_rna.obs[[donor_col, status_col]].copy()
    tmp[donor_col] = tmp[donor_col].astype(str).str.strip()
    tmp["__status__"] = tmp[status_col].map(_normalize_status)
    status_map = (
        tmp.dropna(subset=["__status__"])
           .drop_duplicates(subset=[donor_col])
           .set_index(donor_col)["__status__"]
           .to_dict()
    )
    return status_map

# =========================
# ======  PLOTTING  =======
# =========================
def _pearson_title(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 3 and np.std(x[mask]) > 0 and np.std(y[mask]) > 0:
        r, p = pearsonr(x[mask], y[mask])
        return f"Pearson r = {r:.3f}, p = {p:.2e}"
    return "Pearson r = NA"

def plot_donor_scatter_plain(df_join, out_dir):
    ensure_outdir(out_dir)
    x = df_join["mean_prob_rna"].values
    y = df_join["mean_prob_atac"].values

    plt.figure(figsize=(6.8, 5.8))
    sns.regplot(
        data=df_join,
        x="mean_prob_rna",
        y="mean_prob_atac",
        scatter_kws=dict(s=45, alpha=0.85),
        line_kws={"lw": 1.2, "ls": "--", "color": "black"}
    )
    for _, row in df_join.iterrows():
        plt.annotate(str(row["donor"]), (row["mean_prob_rna"], row["mean_prob_atac"]),
                     fontsize=8, alpha=0.75)
    plt.xlabel("RNA AD probability (per-donor mean)")
    plt.ylabel("ATAC AD probability (per-donor mean)")
    plt.title(f"RNA vs ATAC (donor-level)\n{_pearson_title(x, y)}")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "donor_scatter_plain.png")
    plt.savefig(out_path, dpi=160); plt.close()
    print(f"[SAVE] {out_path}")

def plot_donor_scatter_with_status(df_join, out_dir):
    ensure_outdir(out_dir)
    df_plot = df_join.copy()
    if "status" not in df_plot.columns:
        df_plot["status"] = "Unknown"

    # normalize labels again just in case
    df_plot["status"] = df_plot["status"].map(_normalize_status).fillna("Unknown")
    palette = {"AD": "#d62728", "Control": "#1f77b4", "Unknown": "#7f7f7f"}

    x = df_plot["mean_prob_rna"].values
    y = df_plot["mean_prob_atac"].values

    plt.figure(figsize=(7.2, 6.2))
    sns.scatterplot(
        data=df_plot,
        x="mean_prob_rna",
        y="mean_prob_atac",
        hue="status",
        palette=palette,
        s=55,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.5
    )
    sns.regplot(
        data=df_plot,
        x="mean_prob_rna",
        y="mean_prob_atac",
        scatter=False,
        line_kws={"lw": 1.2, "ls": "--", "color": "black"}
    )
    for _, row in df_plot.iterrows():
        plt.annotate(str(row["donor"]), (row["mean_prob_rna"], row["mean_prob_atac"]),
                     fontsize=8, alpha=0.75)
    plt.xlabel("RNA AD probability (per-donor mean)")
    plt.ylabel("ATAC AD probability (per-donor mean)")
    plt.title(f"RNA vs ATAC (donor-level) by status\n{_pearson_title(x, y)}")
    plt.legend(title="Group", frameon=False)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "donor_scatter_by_status.png")
    plt.savefig(out_path, dpi=160); plt.close()
    print(f"[SAVE] {out_path}")

# =========================
# ========= MAIN ==========
# =========================
def main():
    ensure_outdir(OUT_DIR); ensure_outdir(PLOTS_DIR)

    # ---- Load data
    print("[STEP] Reading .h5ad …")
    adata_rna  = robust_read_h5ad(RAW_RNA_PATH)
    adata_atac = robust_read_h5ad(RAW_ATAC_PATH)

    # ---- Preprocess (same as training)
    print("[STEP] Preprocessing RNA …")
    adata_rna  = _preprocess_rna(adata_rna,  frac=RNA_FRAC,  flavor=RNA_FLAVOR, n_pcs=N_PCS)
    print("[STEP] Preprocessing ATAC …")
    adata_atac = _preprocess_atac(adata_atac, frac=ATAC_FRAC, n_pcs=N_PCS, use_tfidf=USE_TFIDF)

    # ---- Features
    X_rna  = features_from_adata(adata_rna)
    X_atac = features_from_adata(adata_atac)
    print(f"[INFO] RNA features:  {X_rna.shape}")
    print(f"[INFO] ATAC features: {X_atac.shape}")

    # ---- Metadata (barcode + donor)
    bar_rna  = find_col(["barcode","cell","cell_barcode","obs_names"], adata_rna.obs.columns) or "__barcode__"
    bar_atac = find_col(["barcode","cell","cell_barcode","obs_names"], adata_atac.obs.columns) or "__barcode__"
    if bar_rna == "__barcode__": adata_rna.obs["__barcode__"]   = adata_rna.obs_names.astype(str)
    if bar_atac == "__barcode__": adata_atac.obs["__barcode__"] = adata_atac.obs_names.astype(str)

    if DONOR_COL_RNA not in adata_rna.obs.columns:
        raise ValueError(f"RNA donor column '{DONOR_COL_RNA}' not found. Columns: {list(adata_rna.obs.columns)}")
    if DONOR_COL_ATAC not in adata_atac.obs.columns:
        raise ValueError(f"ATAC donor column '{DONOR_COL_ATAC}' not found. Columns: {list(adata_atac.obs.columns)}")

    donors_rna  = adata_rna.obs[DONOR_COL_RNA].astype(str).str.strip().values
    donors_atac = adata_atac.obs[DONOR_COL_ATAC].astype(str).str.strip().values

    # ---- Pick best models (from AUROC CSVs)
    best_rna_model  = pick_best_model_path(RNA_MODELS_DIR,  RNA_AUROC_CSV)
    best_atac_model = pick_best_model_path(ATAC_MODELS_DIR, ATAC_AUROC_CSV)
    print(f"[INFO] Best RNA model:  {best_rna_model}")
    print(f"[INFO] Best ATAC model: {best_atac_model}")

    # ---- Predict per-cell probabilities
    print("[STEP] Predicting RNA …")
    rna_best = model_predict_prob(best_rna_model, X_rna)
    probs_rna, rna_avg = avg_prediction_if_flat(rna_best, list_all_models(RNA_MODELS_DIR), X_rna, FLAT_STD_THRES)
    print(f"[INFO] RNA std={np.std(rna_best):.4f} -> {'avg all models' if rna_avg else 'best only'}")

    print("[STEP] Predicting ATAC …")
    atac_best = model_predict_prob(best_atac_model, X_atac)
    probs_atac, atac_avg = avg_prediction_if_flat(atac_best, list_all_models(ATAC_MODELS_DIR), X_atac, FLAT_STD_THRES)
    print(f"[INFO] ATAC std={np.std(atac_best):.4f} -> {'avg all models' if atac_avg else 'best only'}")

    # ---- Build per-cell tables
    df_rna = pd.DataFrame({
        "barcode": adata_rna.obs[bar_rna].astype(str).values,
        "donor":   donors_rna,
        "prob":    probs_rna
    })
    df_atac = pd.DataFrame({
        "barcode": adata_atac.obs[bar_atac].astype(str).values,
        "donor":   donors_atac,
        "prob":    probs_atac
    })

    # ---- Common donors only
    common = sorted(set(df_rna["donor"]).intersection(set(df_atac["donor"])))
    print(f"[INFO] Common donors (RNA ∩ ATAC): {len(common)}")
    if len(common) < 3:
        print("[WARN] <3 donors in common; correlation will be unstable.")
    df_rna  = df_rna[df_rna["donor"].isin(common)].copy()
    df_atac = df_atac[df_atac["donor"].isin(common)].copy()

    # ---- Aggregate per donor
    agg = {"mean": np.mean, "median": np.median}[AGG_FUNC]
    rna_by_donor  = df_rna.groupby("donor")["prob"].agg(agg).rename("mean_prob_rna").reset_index()
    atac_by_donor = df_atac.groupby("donor")["prob"].agg(agg).rename("mean_prob_atac").reset_index()
    df_join = pd.merge(rna_by_donor, atac_by_donor, on="donor", how="inner")

    # ---- Attach AD/Control status (robust & type-safe)
    status_map = build_status_map_from_rna(adata_rna, DONOR_COL_RNA)
    if status_map:
        df_join["status"] = df_join["donor"].astype(str).str.strip().map(status_map).fillna("Unknown")
    else:
        print("[INFO] No AD/control status column found in RNA obs; setting all to 'Unknown'.")
        df_join["status"] = "Unknown"

    # ---- Save tables
    ensure_outdir(OUT_DIR)
    cell_csv_rna  = os.path.join(OUT_DIR, "per_cell_probs_used_rna.csv")
    cell_csv_atac = os.path.join(OUT_DIR, "per_cell_probs_used_atac.csv")
    donor_csv     = os.path.join(OUT_DIR, "per_donor_mean_probs.csv")
    df_rna.assign(modality="RNA").to_csv(cell_csv_rna,  index=False)
    df_atac.assign(modality="ATAC").to_csv(cell_csv_atac, index=False)
    df_join.to_csv(donor_csv, index=False)
    print(f"[SAVE] {cell_csv_rna}")
    print(f"[SAVE] {cell_csv_atac}")
    print(f"[SAVE] {donor_csv}")

    # ---- Plots
    ensure_outdir(PLOTS_DIR)
    plot_donor_scatter_plain(df_join, PLOTS_DIR)
    plot_donor_scatter_with_status(df_join, PLOTS_DIR)

    print("[DONE] Task 2 outputs in:", OUT_DIR)

if __name__ == "__main__":
    main()


