# =========================
# ======  CONFIG  =========
# =========================
RAW_RNA_PATH   = "/mnt/d/Lab_project/project_0912/Resources/aci_raw_RNA_filtered_v2.h5ad"
RAW_ATAC_PATH  = "/mnt/d/Lab_project/project_0912/Resources/aci_raw_ATAC_filtered_v2_precomp.h5ad"

# Both modalities: donor IDs live in 'orig.ident'
DONOR_COL_RNA  = "orig.ident"
DONOR_COL_ATAC = "orig.ident"

RNA_MODELS_DIR = "/mnt/d/Lab_project/project_0912/Resources/models_rna"
ATAC_MODELS_DIR= "/mnt/d/Lab_project/project_0912/Resources/models_atac"
RNA_AUROC_CSV  = "/mnt/d/Lab_project/project_0912/Resources/fold_auroc_rna.csv"
ATAC_AUROC_CSV = "/mnt/d/Lab_project/project_0912/Resources/fold_auroc_atac.csv"

OUT_DIR        = "/mnt/d/Lab_project/project_0912/Results/task1_outputs"
BOX_DIR        = OUT_DIR + "/boxplots"
FLAT_STD_THRES = 0.06
N_PCS          = 50

# Preprocess params (match training)
RNA_FRAC   = 1.0
RNA_FLAVOR = "pearson_residuals"
ATAC_FRAC  = 1.0
USE_TFIDF  = False  # True ONLY if ATAC models were trained w/ TF-IDF+LSI

# =========================
# ======  IMPORTS  ========
# =========================
import os, re, glob, warnings, gc
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model

# =========================
# Safe-load ONLY preprocess_rna / preprocess_atac from preprocess.py
# =========================
def _safe_load_preprocess_funcs(preprocess_path="preprocess.py"):
    """
    Load only 'preprocess_rna' and 'preprocess_atac' (plus import statements)
    from preprocess.py, without executing its top-level code.
    """
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
print("[INFO] Loaded preprocess_rna / preprocess_atac from preprocess.py (top-level code skipped).")

# =========================
# Utils
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

def _norm_donor_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

# Try to sort donor IDs numerically; fallback to RNA median order if not numeric
def _shared_donor_order(df_rna, df_atac):
    donors = sorted(set(df_rna["donor"]).intersection(set(df_atac["donor"])))
    def _is_intable(x):
        try:
            int(str(x)); return True
        except Exception:
            return False
    if donors and all(_is_intable(d) for d in donors):
        return sorted(donors, key=lambda d: int(str(d)))
    med = df_rna.groupby("donor")["prob_rna"].median().reindex(donors).sort_values(ascending=False)
    return med.index.tolist()

# =========================
# AUROC CSV → model selection
# =========================
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

# =========================
# Inference helpers
# =========================
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
# Plotting (box + dots, shared donor order)
# =========================
def save_box_dotplot(df, prob_col, title, fname, out_dir, ordered_donors):
    ensure_outdir(out_dir)
    df_plot = df.copy()
    df_plot["donor"] = pd.Categorical(df_plot["donor"], categories=ordered_donors, ordered=True)

    plt.figure(figsize=(max(6, 0.5*len(ordered_donors)+2), 6))
    ax = sns.boxplot(
        data=df_plot, x="donor", y=prob_col,
        color="#bdbdbd", fliersize=0
    )
    # Transparent black dots
    sns.stripplot(
        data=df_plot, x="donor", y=prob_col,
        color="black", alpha=0.20, size=2.5, jitter=0.30,
        linewidth=0, edgecolor=None
    )
    ax.axhline(0.5, linestyle="--", linewidth=1, color="black", alpha=0.6)
    ax.set_title(title); ax.set_xlabel("Donor"); ax.set_ylabel("Probability (AD)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"[SAVE] {out_path}")

# =========================
# Main
# =========================
def main():
    ensure_outdir(OUT_DIR); ensure_outdir(BOX_DIR)

    # ---- Load & preprocess ----
    print("[STEP] Reading .h5ad …")
    adata_rna  = robust_read_h5ad(RAW_RNA_PATH)
    adata_atac = robust_read_h5ad(RAW_ATAC_PATH)

    print("[STEP] Preprocessing RNA …")
    adata_rna  = _preprocess_rna(adata_rna,  frac=RNA_FRAC,  flavor=RNA_FLAVOR, n_pcs=N_PCS)
    print("[STEP] Preprocessing ATAC …")
    adata_atac = _preprocess_atac(adata_atac, frac=ATAC_FRAC, n_pcs=N_PCS, use_tfidf=USE_TFIDF)

    # ---- Features ----
    X_rna  = features_from_adata(adata_rna)
    X_atac = features_from_adata(adata_atac)
    print(f"[INFO] RNA features:  {X_rna.shape}")
    print(f"[INFO] ATAC features: {X_atac.shape}")

    # ---- Metadata (barcode + donor) ----
    bar_rna  = find_col(["barcode","cell","cell_barcode","obs_names"], adata_rna.obs.columns) or "__barcode__"
    bar_atac = find_col(["barcode","cell","cell_barcode","obs_names"], adata_atac.obs.columns) or "__barcode__"
    if bar_rna == "__barcode__": adata_rna.obs["__barcode__"]   = adata_rna.obs_names.astype(str)
    if bar_atac == "__barcode__": adata_atac.obs["__barcode__"] = adata_atac.obs_names.astype(str)

    if DONOR_COL_RNA not in adata_rna.obs.columns:
        raise ValueError(f"RNA donor column '{DONOR_COL_RNA}' not found. Columns: {list(adata_rna.obs.columns)}")
    if DONOR_COL_ATAC not in adata_atac.obs.columns:
        raise ValueError(f"ATAC donor column '{DONOR_COL_ATAC}' not found. Columns: {list(adata_atac.obs.columns)}")

    df_meta_rna  = adata_rna.obs[[bar_rna, DONOR_COL_RNA ]].copy().rename(columns={bar_rna:"barcode",  DONOR_COL_RNA :"donor"})
    df_meta_atac = adata_atac.obs[[bar_atac, DONOR_COL_ATAC]].copy().rename(columns={bar_atac:"barcode", DONOR_COL_ATAC:"donor"})
    df_meta_rna["donor"]  = df_meta_rna["donor"].astype(str).str.strip()
    df_meta_atac["donor"] = df_meta_atac["donor"].astype(str).str.strip()

    # ---- Pick best models ----
    best_rna_model  = pick_best_model_path(RNA_MODELS_DIR,  RNA_AUROC_CSV)
    best_atac_model = pick_best_model_path(ATAC_MODELS_DIR, ATAC_AUROC_CSV)
    print(f"[INFO] Best RNA model:  {best_rna_model}")
    print(f"[INFO] Best ATAC model: {best_atac_model}")

    # ---- Predict ----
    print("[STEP] Predicting RNA …")
    rna_best = model_predict_prob(best_rna_model, X_rna)
    probs_rna, rna_avg = avg_prediction_if_flat(rna_best, list_all_models(RNA_MODELS_DIR), X_rna, FLAT_STD_THRES)
    print(f"[INFO] RNA std={np.std(rna_best):.4f} -> {'avg all models' if rna_avg else 'best only'}")

    print("[STEP] Predicting ATAC …")
    atac_best = model_predict_prob(best_atac_model, X_atac)
    probs_atac, atac_avg = avg_prediction_if_flat(atac_best, list_all_models(ATAC_MODELS_DIR), X_atac, FLAT_STD_THRES)
    print(f"[INFO] ATAC std={np.std(atac_best):.4f} -> {'avg all models' if atac_avg else 'best only'}")

    # ---- Per-cell tables ----
    df_rna = pd.DataFrame({
        "barcode": df_meta_rna["barcode"].astype(str).values,
        "donor":   df_meta_rna["donor"].astype(str).values,
        "prob_rna": probs_rna
    })
    df_atac = pd.DataFrame({
        "barcode": df_meta_atac["barcode"].astype(str).values,
        "donor":   df_meta_atac["donor"].astype(str).values,
        "prob_atac": probs_atac
    })

    # ---- Keep ONLY donors present in BOTH modalities ----
    common_donors = sorted(set(df_rna["donor"]).intersection(set(df_atac["donor"])))
    print(f"[INFO] Common donors (RNA ∩ ATAC): {len(common_donors)}")
    if len(common_donors) == 0:
        print("[WARN] No common donors found—plots will be empty. Check donor labels.")
    df_rna  = df_rna[df_rna["donor"].isin(common_donors)].copy()
    df_atac = df_atac[df_atac["donor"].isin(common_donors)].copy()

    # ---- Save per-cell CSVs ----
    ensure_outdir(OUT_DIR)
    rna_csv  = os.path.join(OUT_DIR, "rna_cell_probs.csv")
    atac_csv = os.path.join(OUT_DIR, "atac_cell_probs.csv")
    df_rna.to_csv(rna_csv,  index=False)
    df_atac.to_csv(atac_csv, index=False)
    print(f"[SAVE] {rna_csv}")
    print(f"[SAVE] {atac_csv}")

    # ---- Shared donor order (numeric if possible; else RNA median order)
    ordered_donors = _shared_donor_order(df_rna, df_atac)
    print(f"[INFO] Donor order used for both plots: {ordered_donors}")

    # ---- Box + dot plots (same x-axis order in both)
    ensure_outdir(BOX_DIR)
    save_box_dotplot(df_rna,  "prob_rna",  "RNA probability by donor",  "rna_boxplot.png",  BOX_DIR, ordered_donors)
    save_box_dotplot(df_atac, "prob_atac", "ATAC probability by donor", "atac_boxplot.png", BOX_DIR, ordered_donors)
    print("[DONE] Box plots written to:", BOX_DIR)

if __name__ == "__main__":
    main()








