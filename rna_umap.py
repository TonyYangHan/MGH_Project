#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_umap_rna_over_atac_ratio_shared_axes.py
--------------------------------------------------
Builds a UMAP where color = RNA_prob / ATAC_prob for cells present in BOTH
datasets, and uses the SAME axis limits saved by your ATAC script
(OUT_DIR/shared_umap_axes.json).

Outputs (in OUT_DIR):
  - umap_ratio_rna_over_atac_linear.png  (linear ratio)
  - umap_ratio_rna_over_atac_log2.png    (log2 ratio, diverging cmap)

Assumptions:
  - Per-cell probabilities live in .obs["p_cell"] for both RNA & ATAC.
  - If missing, the script can optionally read a .npy fallback (same cell order).
  - Cell IDs overlap between RNA and ATAC (ratio is computed on the intersection).
  - ATAC UMAP coordinates are used for plotting; axis limits are loaded from JSON.

Requirements:
  pip install scanpy anndata pandas numpy matplotlib
"""

import os, json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ================================
# CONFIG — edit paths if needed
# ================================
RNA_H5AD  = "/mnt/d/Lab_project/accuracy_improvement/Resources/aci_rna_precomp_v3.h5ad"
ATAC_H5AD = "/mnt/d/Lab_project/accuracy_improvement/Resources/aci_atac_precomp_v3.h5ad"

# Optional fallbacks if obs['p_cell'] is missing
RNA_PROBA_NPY_FALLBACK  = "/mnt/d/Lab_project/accuracy_improvement/Resources/avg_proba_rna.npy"
ATAC_PROBA_NPY_FALLBACK = "/mnt/d/Lab_project/accuracy_improvement/Resources/avg_proba_atac.npy"

# Where plots + shared axes JSON live
OUT_DIR    = "/mnt/d/Lab_project/accuracy_improvement/umap_avg_proba_plot"
AXES_JSON  = os.path.join(OUT_DIR, "shared_umap_axes.json")
os.makedirs(OUT_DIR, exist_ok=True)

# Which embedding to use for the scatter coordinates
# ("ATAC" recommended since the axes JSON was produced from ATAC)
COORD_SOURCE = "ATAC"   # choices: "ATAC" or "RNA"
UMAP_KEY     = "X_umap"
NEIGHBOR_REP = "X_pca"  # if UMAP missing, compute from this rep

# Ratio handling
DENOM_EPS = 1e-6          # avoid division by zero
CLIP_LINEAR_TO_PCT = 99   # cap linear ratio color scale at this percentile (None to disable)
LINEAR_CMAP = "viridis"
LOG2_CMAP   = "coolwarm"  # centered at 0 via TwoSlopeNorm

POINT_SIZE  = 8
POINT_ALPHA = 0.55
BG_NAN_COLOR = (0.85, 0.85, 0.85, 0.35)  # light gray for any background scatter if needed

# ================================
# Helpers
# ================================
def ensure_umap(adata: sc.AnnData, label: str):
    """Ensure adata.obsm[UMAP_KEY] exists; if not, compute neighbors+umap."""
    if UMAP_KEY not in adata.obsm:
        print(f"ℹ️ '{UMAP_KEY}' not in {label}; computing neighbors/UMAP from '{NEIGHBOR_REP if NEIGHBOR_REP in adata.obsm else 'X'}'...")
        use_rep = NEIGHBOR_REP if NEIGHBOR_REP in adata.obsm else None
        sc.pp.neighbors(adata, use_rep=use_rep)
        sc.tl.umap(adata)

def get_probs(adata: sc.AnnData, npy_fallback: str, label: str) -> pd.Series:
    """Get per-cell probabilities as a pandas Series indexed by obs_names."""
    if "p_cell" in adata.obs:
        print(f"✅ Using {label} probabilities from adata.obs['p_cell']")
        vals = adata.obs["p_cell"].to_numpy()
    elif os.path.exists(npy_fallback):
        print(f"ℹ️ Using {label} probabilities from fallback: {npy_fallback}")
        vals = np.load(npy_fallback)
        if vals.shape[0] != adata.n_obs:
            raise ValueError(f"{label} fallback length ({vals.shape[0]}) != n_obs ({adata.n_obs})")
    else:
        raise FileNotFoundError(f"{label}: No probabilities found in obs['p_cell'] and fallback npy not found.")
    return pd.Series(vals, index=adata.obs_names, name=f"{label}_p")

def load_shared_axes(json_path: str, coords: np.ndarray):
    """Load X/Y limits from JSON; fallback to local computation if missing."""
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            shared = json.load(f)
        X_LIM = (float(shared["X_LIM"][0]), float(shared["X_LIM"][1]))
        Y_LIM = (float(shared["Y_LIM"][0]), float(shared["Y_LIM"][1]))
        print(f"✅ Loaded shared UMAP axis limits from {json_path}")
    else:
        print("⚠️ Shared axes JSON not found; computing limits from current coords.")
        xpad = 0.02 * (coords[:,0].max() - coords[:,0].min() + 1e-9)
        ypad = 0.02 * (coords[:,1].max() - coords[:,1].min() + 1e-9)
        X_LIM = (float(coords[:,0].min() - xpad), float(coords[:,0].max() + xpad))
        Y_LIM = (float(coords[:,1].min() - ypad), float(coords[:,1].max() + ypad))
    return X_LIM, Y_LIM

def fixed_axes(ax, xlim, ylim):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

# ================================
# Load data
# ================================
print("🔹 Loading RNA AnnData...")
rna = sc.read_h5ad(RNA_H5AD)
print("🔹 Loading ATAC AnnData...")
atac = sc.read_h5ad(ATAC_H5AD)

# Ensure UMAP coords exist (for both, in case you choose RNA coords)
ensure_umap(rna, "RNA")
ensure_umap(atac, "ATAC")

# Choose coordinate source
if COORD_SOURCE.upper() == "RNA":
    coords_source = rna
    print("ℹ️ Using RNA UMAP coordinates for plotting.")
else:
    coords_source = atac
    print("ℹ️ Using ATAC UMAP coordinates for plotting.")

coords_all = coords_source.obsm[UMAP_KEY]

# Load shared axes from JSON (written by your ATAC script)
X_LIM, Y_LIM = load_shared_axes(AXES_JSON, coords_all)

# ================================
# Pull probabilities and intersect cells
# ================================
rna_p   = get_probs(rna,  RNA_PROBA_NPY_FALLBACK,  "RNA")
atac_p  = get_probs(atac, ATAC_PROBA_NPY_FALLBACK, "ATAC")

common_cells = rna.obs_names.intersection(atac.obs_names)
if common_cells.size == 0:
    raise RuntimeError("No overlapping cell IDs between RNA and ATAC; cannot compute ratio.")
print(f"✅ Overlapping cells for ratio: {common_cells.size:,} of RNA={rna.n_obs:,}, ATAC={atac.n_obs:,}")

# Subset in the order of the coordinate source for consistent scatter
common_cells = coords_source.obs_names.intersection(common_cells)
if common_cells.size == 0:
    raise RuntimeError("No overlapping cells after aligning to coordinate source's obs_names order.")

coords = coords_source.obsm[UMAP_KEY][coords_source.obs_names.get_indexer(common_cells), :]

# Align probabilities to common cell order
rna_vals  = rna_p.loc[common_cells].to_numpy()
atac_vals = atac_p.loc[common_cells].to_numpy()

# ================================
# Compute ratio safely
# ================================
denom = np.clip(atac_vals, DENOM_EPS, None)
ratio = rna_vals / denom

# Replace inf/neg with NaN; keep finite positives
ratio[~np.isfinite(ratio)] = np.nan

# ================================
# Plot 1: Linear ratio
# ================================
finite = np.isfinite(ratio)
if CLIP_LINEAR_TO_PCT is not None and np.any(finite):
    vmax = float(np.nanpercentile(ratio, CLIP_LINEAR_TO_PCT))
    # Avoid tiny vmax
    if vmax < 1.0:
        vmax = 1.0
else:
    vmax = float(np.nanmax(ratio)) if np.any(finite) else 2.0
vmin = 0.0

fig, ax = plt.subplots(figsize=(8, 6))
# Background: (optional) plot nothing behind; or add very faint all-points if you want full silhouette
# ax.scatter(coords_all[:,0], coords_all[:,1], s=1, color=BG_NAN_COLOR, alpha=BG_NAN_COLOR[3])

scat = ax.scatter(coords[finite, 0], coords[finite, 1],
                  c=ratio[finite], s=POINT_SIZE, alpha=POINT_ALPHA,
                  cmap=LINEAR_CMAP, vmin=vmin, vmax=vmax)
# If there are NaNs (e.g., denom <= eps), optionally show them as faint gray
if np.any(~finite):
    ax.scatter(coords[~finite, 0], coords[~finite, 1],
               s=POINT_SIZE, alpha=0.25, color=BG_NAN_COLOR)

cbar = plt.colorbar(scat, ax=ax)
cbar.set_label("RNA probability / ATAC probability (linear)")
ax.set_title("UMAP — Ratio (RNA / ATAC)")
fixed_axes(ax, X_LIM, Y_LIM)
fig.tight_layout()
out_linear = os.path.join(OUT_DIR, "umap_ratio_rna_over_atac_linear.png")
fig.savefig(out_linear, dpi=300)
plt.close(fig)
print(f"✅ Saved: {out_linear} (vmin={vmin:.2f}, vmax≈{vmax:.2f})")

# ================================
# Plot 2: log2 ratio (centered at 0)
# ================================
log2_ratio = np.full_like(ratio, np.nan, dtype=float)
with np.errstate(divide='ignore', invalid='ignore'):
    log2_ratio[finite] = np.log2(ratio[finite])

# Robust symmetric limits around 0 using percentiles (avoid extreme outliers)
if np.any(np.isfinite(log2_ratio)):
    hi = float(np.nanpercentile(np.abs(log2_ratio), 99))
    # ensure a reasonable lower bound
    if hi < 0.5:
        hi = 0.5
    vmin_log2, vmax_log2 = -hi, +hi
else:
    vmin_log2, vmax_log2 = -1.0, 1.0

fig, ax = plt.subplots(figsize=(8, 6))
norm = TwoSlopeNorm(vmin=vmin_log2, vcenter=0.0, vmax=vmax_log2)
scat = ax.scatter(coords[np.isfinite(log2_ratio), 0], coords[np.isfinite(log2_ratio), 1],
                  c=log2_ratio[np.isfinite(log2_ratio)],
                  s=POINT_SIZE, alpha=POINT_ALPHA,
                  cmap=LOG2_CMAP, norm=norm)

# Optional NaN background points
if np.any(~np.isfinite(log2_ratio)):
    ax.scatter(coords[~np.isfinite(log2_ratio), 0], coords[~np.isfinite(log2_ratio), 1],
               s=POINT_SIZE, alpha=0.25, color=BG_NAN_COLOR)

cbar = plt.colorbar(scat, ax=ax)
cbar.set_label("log2(RNA probability / ATAC probability)")
ax.set_title("UMAP — log2 Ratio (RNA / ATAC)")
fixed_axes(ax, X_LIM, Y_LIM)
fig.tight_layout()
out_log2 = os.path.join(OUT_DIR, "umap_ratio_rna_over_atac_log2.png")
fig.savefig(out_log2, dpi=300)
plt.close(fig)
print(f"✅ Saved: {out_log2} (centered at 0, limits [{vmin_log2:.2f}, {vmax_log2:.2f}])")

print("🎉 Done.")



