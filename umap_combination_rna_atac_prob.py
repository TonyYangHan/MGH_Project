#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_umap_abs_log2ratio_plasma_r.py
--------------------------------------------------
Plots |log2(RNA probability / ATAC probability)| on UMAP,
using the 'plasma_r' colormap (yellow=0, purple=high),
matching the visual style of the reference image.

Enhancements:
  - Uses 'plasma_r' (bright yellow for 0, dark purple for high values)
  - Yellow points (low abs logFC) drawn last (on top)
  - Increased transparency for readability in dense regions
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

# ========= CONFIG =========
RNA_H5AD  = "/mnt/d/Lab_project/accuracy_improvement/Resources/aci_rna_precomp_v3.h5ad"
ATAC_H5AD = "/mnt/d/Lab_project/accuracy_improvement/Resources/aci_atac_precomp_v3.h5ad"

RNA_PROBA_NPY  = "/mnt/d/Lab_project/accuracy_improvement/Resources/avg_proba_rna.npy"
ATAC_PROBA_NPY = "/mnt/d/Lab_project/accuracy_improvement/Resources/avg_proba_atac.npy"

OUT_DIR = "/mnt/d/Lab_project/accuracy_improvement/umap_avg_proba_plot"
os.makedirs(OUT_DIR, exist_ok=True)

UMAP_KEY = "X_umap"
FALLBACK_GRAPH_REP = "X_pca"

POINT_SIZE  = 8
POINT_ALPHA = 0.45  # semi-transparent for overlap clarity
ABS_LOG2_MAX = 2.0
EPS = 1e-6
CMAP = "plasma_r"   # same as your uploaded example (yellow→purple)

# ========= Load AnnData =========
print("🔹 Loading RNA/ATAC AnnData...")
rna  = sc.read_h5ad(RNA_H5AD)
atac = sc.read_h5ad(ATAC_H5AD)

# ========= Probabilities =========
def get_probs(adata, npy_path, label):
    if "p_cell" in adata.obs:
        print(f"✅ Using {label} probabilities from adata.obs['p_cell']")
        return pd.Series(adata.obs["p_cell"].to_numpy(), index=adata.obs_names, name=f"p_{label}")
    elif os.path.exists(npy_path):
        print(f"ℹ️ Using fallback {npy_path} for {label}")
        arr = np.load(npy_path)
        if arr.shape[0] != adata.n_obs:
            raise ValueError(f"{label} npy length mismatch")
        return pd.Series(arr, index=adata.obs_names, name=f"p_{label}")
    else:
        raise FileNotFoundError(f"No {label} probabilities found.")

p_rna  = get_probs(rna,  RNA_PROBA_NPY,  "RNA")
p_atac = get_probs(atac, ATAC_PROBA_NPY, "ATAC")

# ========= Align cells =========
common = rna.obs_names.intersection(atac.obs_names)
if len(common) == 0:
    raise ValueError("No overlapping cell barcodes between RNA and ATAC.")
print(f"✅ Common cells: {len(common):,}")

rna  = rna[common].copy()
atac = atac[common].copy()
p_rna  = p_rna.loc[common]
p_atac = p_atac.loc[common]

# ========= UMAP =========
if UMAP_KEY not in atac.obsm:
    print("ℹ️ ATAC UMAP missing; computing from PCA...")
    rep = FALLBACK_GRAPH_REP if FALLBACK_GRAPH_REP in atac.obsm else None
    sc.pp.neighbors(atac, use_rep=rep)
    sc.tl.umap(atac)

coords = atac.obsm[UMAP_KEY]

# ========= Compute abs(log2 ratio) =========
ratio = (p_rna.values + EPS) / (p_atac.values + EPS)
abs_log2ratio = np.abs(np.log2(ratio))
abs_log2_clip = np.clip(abs_log2ratio, 0.0, ABS_LOG2_MAX)

# Save numeric output
out_csv = os.path.join(OUT_DIR, "abs_log2ratio_rna_over_atac_plasma.csv")
pd.DataFrame({
    "cell": common,
    "abs_log2ratio": abs_log2ratio,
}).to_csv(out_csv, index=False)
print(f"✅ Saved numeric values → {out_csv}")

# ========= Plot =========
plt.figure(figsize=(8, 6))

# Sort so yellow (low values) are drawn on top
order = np.argsort(abs_log2_clip)[::-1]
coords_sorted = coords[order]
vals_sorted = abs_log2_clip[order]

sca = plt.scatter(
    coords_sorted[:, 0], coords_sorted[:, 1],
    c=vals_sorted, s=POINT_SIZE, alpha=POINT_ALPHA,
    cmap=CMAP, vmin=0.0, vmax=ABS_LOG2_MAX
)

cbar = plt.colorbar(sca)
cbar.set_label("|log2(RNA/ATAC)| (bright=0, dark=high)")
plt.title("UMAP — |log2(RNA probability / ATAC probability)| (plasma_r, bright=0.0)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.tight_layout()

out_png = os.path.join(OUT_DIR, "umap_abs_log2ratio_rna_over_atac_plasma_r.png")
plt.savefig(out_png, dpi=300)
plt.close()
print(f"✅ Saved: {out_png}")

print("🎉 Done — uses 'plasma_r' (yellow=0, purple=high) matching your reference image.")









