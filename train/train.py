import os, gc, itertools, numpy as np, pandas as pd, scanpy as sc, shap, torch, argparse
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn as nn, yaml
from torch.utils.data import TensorDataset, DataLoader

"""
This script trains a simple MLP on PCA features from an AnnData (.h5ad) object.
Now configurable via config.yaml and CLI args.
CLI args:
    --adata: path to .h5ad
    --result_dir: output directory root
    --analysis: 'rna' or 'atac' (controls feature column name and output subdir)
    --config: path to YAML config (default: config.yaml)
"""

# --- CLI args & config ---
parser = argparse.ArgumentParser(description="Train MLP on PCA features with SHAP explanations")
parser.add_argument("adata", help="Path to input .h5ad file")
parser.add_argument("analysis", choices=["rna", "atac"], help="Type of analysis")
parser.add_argument("result_dir", help="Directory to write outputs")
parser.add_argument("--config", default="config.yaml", help="Path to YAML config (default: config.yaml)")
args = parser.parse_args()

with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training hyperparams
tr_cfg = cfg.get("training", {})
seed = int(tr_cfg.get("seed", 42))
torch.manual_seed(seed); np.random.seed(seed)
batch_size = int(tr_cfg.get("batch_size", 256))
max_epochs = int(tr_cfg.get("max_epochs", 100))
lr = float(tr_cfg.get("lr", 1e-3))
early_stop_patience = int(tr_cfg.get("patience", 5))
val_split = float(tr_cfg.get("val_split", 0.2))
val_random_state = int(tr_cfg.get("val_random_state", 42))
use_stratify = bool(tr_cfg.get("stratify", True))

sch_cfg = cfg.get("scheduler", {})
use_scheduler = bool(sch_cfg.get("use_scheduler", True))
save_best_model = bool(sch_cfg.get("save_best_model", True))
early_stop_on_plateau = bool(sch_cfg.get("early_stop_on_plateau", True))
scheduler_name = sch_cfg.get("name", "reduce_lr_on_plateau")
scheduler_mode = sch_cfg.get("mode", "min")
scheduler_factor = float(sch_cfg.get("factor", 0.5))
scheduler_patience = int(sch_cfg.get("patience", 5))
scheduler_min_lr = float(sch_cfg.get("min_lr", 1e-6))

data_cfg = cfg.get("data", {})
pca_key = data_cfg.get("pca_key", "X_pca")
pcs_loadings_key = data_cfg.get("pcs_loadings_key", "PCs")
label_key = data_cfg.get("label_key", "AD")
donor_key = data_cfg.get("donor_key", "orig.ident")
ad_label = data_cfg.get("ad_label", "AD")
control_label = data_cfg.get("control_label", "Control")

out_cfg = cfg.get("output", {})
models_subdir = out_cfg.get("models_subdir", "models")
gene_importance_subdir = out_cfg.get("gene_importance_subdir", "gene_importance")
peak_importance_subdir = out_cfg.get("peak_importance_subdir", "peak_importance")
metrics_filename = out_cfg.get("metrics_filename", "fold_metrics.csv")

model_cfg = cfg.get("model", {})
hidden_sizes = list(model_cfg.get("hidden_sizes", [64, 64]))
activation_name = str(model_cfg.get("activation", "elu")).lower()

cv_cfg = cfg.get("cross_validation", {})
cv_seed = int(cv_cfg.get("seed", seed))
cv_max_folds = cv_cfg.get("max_folds", None)

shap_cfg = cfg.get("shap", {})
background_kmeans = int(shap_cfg.get("background_kmeans", 100))

metrics_cfg = cfg.get("metrics", {})
report_first_n_folds = int(metrics_cfg.get("report_first_n_folds", 5))
loss_cfg = cfg.get("loss", {})
loss_name = str(loss_cfg.get("name", "bce_with_logits")).lower()
use_logits_loss = loss_name in ["bce_with_logits", "bcewithlogits", "bce_with_logit"]

# --- data ---
adata = sc.read_h5ad(args.adata)
print(f"Data preprocessed: {adata.n_obs} cells, {adata.n_vars} genes.")

# result dirs
result_dir = args.result_dir
os.makedirs(result_dir, exist_ok=True)
models_dir = os.path.join(result_dir, models_subdir)
# Select importances subdir and feature label based on analysis
if args.analysis == "rna":
    importances_dir = os.path.join(result_dir, gene_importance_subdir)
    feature_label = "Gene"
    importance_prefix = "gene"
else:
    importances_dir = os.path.join(result_dir, peak_importance_subdir)
    feature_label = "Peak"
    importance_prefix = "peak"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(importances_dir, exist_ok=True)

# donor pairing
ad_donors   = adata.obs.loc[adata.obs[label_key] == ad_label, donor_key].unique()
ctrl_donors = adata.obs.loc[adata.obs[label_key] == control_label, donor_key].unique()
pairs = np.array(list(itertools.product(ad_donors, ctrl_donors)))
validation_folds = pairs.shape[0]
np.random.seed(42)
selected_pairs = np.random.choice(pairs.shape[0], size=validation_folds, replace=False)

aucs, accs, thrs = [], [], []

# --- model def ---
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        # activation
        act_layer = nn.ELU if activation_name == "elu" else nn.ReLU
        layers = []
        in_dim = d
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(act_layer())
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, 1))  # logits
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def make_loader(X, y=None, shuffle=False):
    X = torch.as_tensor(X, dtype=torch.float32)
    if y is None:
        return DataLoader(TensorDataset(X, torch.zeros(len(X))), batch_size=batch_size, shuffle=False, pin_memory=True)
    y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle, pin_memory=True)

for i in trange(validation_folds, desc="Folds"):
    test_donors = pairs[selected_pairs[i]]
    is_test = adata.obs[donor_key].isin(test_donors)
    adata_test = adata[is_test].copy()
    adata_rest = adata[~is_test].copy()

    X_test = adata_test.obsm[pca_key].astype(np.float32)
    y_test = (adata_test.obs[label_key] == ad_label).astype(np.int32).values

    X = adata_rest.obsm[pca_key].astype(np.float32)
    y = (adata_rest.obs[label_key] == ad_label).astype(np.int32).values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_split,
        random_state=val_random_state,
        stratify=y if use_stratify else None,
    )

    train_dl = make_loader(X_train, y_train, shuffle=True)
    val_dl   = make_loader(X_val,   y_val)

    model = MLP(X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if use_scheduler and scheduler_name == "reduce_lr_on_plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=scheduler_mode, factor=scheduler_factor,
            patience=scheduler_patience, min_lr=scheduler_min_lr
        )
    else:
        sched = None
    loss_fn = nn.BCEWithLogitsLoss() if use_logits_loss else nn.BCELoss()

    best_val, best_state, bad = float('inf'), None, 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(xb)
            preds = logits if use_logits_loss else torch.sigmoid(logits)
            loss = loss_fn(preds, yb)
            loss.backward(); opt.step()

        # val
        model.eval(); vloss, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                logits = model(xb)
                preds = logits if use_logits_loss else torch.sigmoid(logits)
                vloss += loss_fn(preds, yb).item() * xb.size(0); n += xb.size(0)
        vloss /= max(1, n)
        if sched is not None:
            sched.step(vloss)

        if vloss < best_val - 1e-6:
            best_val = vloss
            if save_best_model:
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            if early_stop_on_plateau:
                bad += 1
                if bad > early_stop_patience:
                    break

    if save_best_model and best_state is not None:
        model.load_state_dict(best_state)
    # save model
    torch.save(model.state_dict(), os.path.join(models_dir, f'fold{i}_model.pt'))

    # test preds
    model.eval()
    with torch.no_grad():
        p_val = torch.sigmoid(
            model(torch.as_tensor(X_val, dtype=torch.float32, device=device))
        ).squeeze(1).cpu().numpy()

    fpr_v, tpr_v, thr_v = roc_curve(y_val, p_val)
    t_opt = thr_v[(tpr_v - fpr_v).argmax()]   # <-- fixed threshold from validation

    with torch.no_grad():
        logits = torch.as_tensor(X_test, dtype=torch.float32, device=device)
        p_test = torch.sigmoid(model(logits)).squeeze(1).cpu().numpy()
    
    test_auc = roc_auc_score(y_test, p_test)
    test_acc = np.mean((p_test >= t_opt) == y_test)
    aucs.append(test_auc); accs.append(test_acc); thrs.append(t_opt)

    if i < report_first_n_folds:
        print(f"AUROC: {test_auc:.4f}, Accuracy: {test_acc:.4f} for fold {i}, threshold {t_opt:.4f}")

    # SHAP (same logic as original)
    background = shap.kmeans(X_train, 100).data
    explainer = shap.DeepExplainer(model, torch.as_tensor(background, dtype=torch.float32, device=device))
    sv = explainer.shap_values(torch.as_tensor(X_test, dtype=torch.float32, device=device))
    if isinstance(sv, list): sv = sv[0]
    all_shap = np.array(sv).squeeze()
    shap_values = np.mean(np.abs(all_shap), axis=0)  # (n_pcs,)

    pc_load = adata_rest.varm[pcs_loadings_key]  # (features, n_pcs)
    gene_scores = (np.abs(pc_load) @ np.abs(shap_values)).squeeze().tolist()
    feat_df = pd.DataFrame({feature_label: adata_rest.var_names, 'Importance': gene_scores}).sort_values('Importance', ascending=False)
    out_name = f'{importance_prefix}_shap_importance_fold{i}.csv'
    feat_df.to_csv(os.path.join(importances_dir, out_name), index=False)

    # cleanup
    del model, train_dl, val_dl, X_train, X_val, y_train, y_val, background, explainer, adata_test, adata_rest, X_test, y_test
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

metrics_df = pd.DataFrame({
    'Fold': list(range(validation_folds)),
    'test_pair': pairs[selected_pairs].tolist(),
    'AUROC': aucs, 
    'Accuracy': accs,
    'threshold': thrs
})
metrics_df.to_csv(os.path.join(result_dir, metrics_filename), index=False)
print(f"Mean AUROC across folds: {np.mean(aucs):.4f}")
