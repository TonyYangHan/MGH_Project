#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_atac_maps_with_brain_region_shared_axes.py
--------------------------------------------------
Generates ATAC UMAPs (same axes for all plots) and saves shared axis limits to JSON
so RNA plots can reuse the exact same framing.

Outputs (in OUT_DIR):
  - atac_umap_probability.png
  - atac_umap_donor_identity_strongcolors.png
  - atac_umap_donor_identity_strongcolors_labeled.png
  - atac_umap_AD_vs_Control.png
  - atac_umap_region_strongcolors.png
  - atac_umap_region_strongcolors_labeled.png
  - shared_umap_axes.json        <-- RNA script reads this

Requirements:
  pip install scanpy pandas numpy matplotlib
"""

import os, json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from itertools import cycle

# ================================
# CONFIG — edit paths if needed
# ================================
ATAC_H5AD = "/mnt/d/Lab_project/accuracy_improvement/Resources/aci_atac_precomp_v3.h5ad"
P_CELL_NPY_FALLBACK = "/mnt/d/Lab_project/accuracy_improvement/Resources/avg_proba_atac.npy"

OUT_DIR = "/mnt/d/Lab_project/accuracy_improvement/umap_avg_proba_plot"
os.makedirs(OUT_DIR, exist_ok=True)

UMAP_KEY     = "X_umap"   # will compute if missing
NEIGHBOR_REP = "X_pca"    # for neighbors/UMAP if needed

DONOR_COL  = "orig.ident"
REGION_COL = "brain.region"   # exact obs column name

# Diagnosis (for AD vs Control)
CANDIDATE_DIAG_COLS = ["AD", "diagnosis", "diag", "group", "status", "condition"]

# Plot tuning
POINT_SIZE     = 8
POINT_ALPHA    = 0.55
LABEL_DONORS   = True
TOP_N_LABELS   = 25
LABEL_FONTSIZE = 8
SAVE_FOCUS_MAPS = False

# Probability color scale
PROBA_VMIN, PROBA_VMAX = 0.0, 1.0
PROBA_CMAP = "viridis"

# AD/Control colors
AD_COLOR   = "tab:red"
CTRL_COLOR = "tab:blue"
OTHER_GRAY = "lightgray"

# ================================
# Helpers
# ================================
def get_p_cell(adata, fallback_npy):
    if "p_cell" in adata.obs:
        arr = adata.obs["p_cell"].to_numpy()
        print("✅ Using p_cell from adata.obs['p_cell']")
        return arr
    if os.path.exists(fallback_npy):
        arr = np.load(fallback_npy)
        if arr.shape[0] != adata.n_obs:
            raise ValueError(f"Fallback npy length ({arr.shape[0]}) != n_obs ({adata.n_obs})")
        print(f"ℹ️ Using fallback probabilities from: {fallback_npy}")
        return arr
    raise FileNotFoundError("No per-cell probabilities found (obs['p_cell'] missing and npy fallback not found).")

def find_diag_col(adata):
    for c in CANDIDATE_DIAG_COLS:
        if c in adata.obs:
            return c
    return None

def make_ad_control_labels(ser):
    s = ser.astype(str).str.strip().str.lower()
    ad_mask   = s.str.contains(r"\bad\b", regex=True) | s.str.contains("alzheimer")
    ctrl_mask = s.str.contains("control") | s.str.contains(r"\bctrl\b") | s.str.contains(r"\bcn\b") | s.str.contains("healthy")
    return pd.Series(np.where(ad_mask, "AD", np.where(ctrl_mask, "Control", "Other")), index=ser.index)

def build_strong_palette(categories):
    """Build a visually distinct palette by concatenating multiple qualitative maps."""
    colors = []
    for name in ["tab10", "Set3", "Paired", "Dark2", "Accent"]:
        cmap = plt.get_cmap(name)
        try:
            colors.extend(list(cmap.colors))
        except AttributeError:
            colors.extend([cmap(i/10) for i in range(10)])
    # De-duplicate while preserving order
    seen = set(); uniq_colors = []
    for c in colors:
        t = tuple(np.asarray(c).round(6))
        if t not in seen:
            seen.add(t); uniq_colors.append(c)
    # Cycle if needed
    pal = list(uniq_colors)
    if len(categories) > len(pal):
        pal = list(pal) + list(cycle(uniq_colors))
    return {cat: pal[i] for i, cat in enumerate(categories)}

def fixed_axes(ax, xlim, ylim):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

# ================================
# Load data & ensure UMAP
# ================================
print("🔹 Loading ATAC AnnData...")
adata = sc.read_h5ad(ATAC_H5AD)

if UMAP_KEY not in adata.obsm:
    print(f"ℹ️ '{UMAP_KEY}' not found; computing neighbors/UMAP...")
    use_rep = NEIGHBOR_REP if NEIGHBOR_REP in adata.obsm else None
    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.umap(adata)

coords = adata.obsm[UMAP_KEY]
# Fixed axis limits (+ tiny margins)
xpad = 0.02 * (coords[:,0].max() - coords[:,0].min() + 1e-9)
ypad = 0.02 * (coords[:,1].max() - coords[:,1].min() + 1e-9)
X_LIM = (coords[:,0].min() - xpad, coords[:,0].max() + xpad)
Y_LIM = (coords[:,1].min() - ypad, coords[:,1].max() + ypad)

# Save shared limits for RNA script
axes_json = os.path.join(OUT_DIR, "shared_umap_axes.json")
with open(axes_json, "w") as f:
    json.dump({"X_LIM": X_LIM, "Y_LIM": Y_LIM}, f)
print(f"✅ Saved shared UMAP axis limits → {axes_json}")

# ================================
# 1) Probability map
# ================================
p_cell = get_p_cell(adata, P_CELL_NPY_FALLBACK)
fig, ax = plt.subplots(figsize=(8,6))
scat = ax.scatter(coords[:,0], coords[:,1],
                  c=p_cell, s=POINT_SIZE, alpha=POINT_ALPHA,
                  cmap=PROBA_CMAP, vmin=PROBA_VMIN, vmax=PROBA_VMAX)
cbar = plt.colorbar(scat, ax=ax); cbar.set_label("AD probability (per cell)")
ax.set_title("ATAC UMAP — Per-cell AD probability")
fixed_axes(ax, X_LIM, Y_LIM)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "atac_umap_probability.png"), dpi=300)
plt.close(fig)

# ================================
# 2) Donor identity (distinct colors)
# ================================
if DONOR_COL not in adata.obs:
    raise KeyError(f"'{DONOR_COL}' not found in adata.obs")
donors = np.asarray(adata.obs[DONOR_COL])
uniq_donors = np.unique(donors)
donor_colors = build_strong_palette(uniq_donors)

fig, ax = plt.subplots(figsize=(8,6))
for d in uniq_donors:
    idx = (donors == d)
    ax.scatter(coords[idx,0], coords[idx,1], s=POINT_SIZE,
               color=donor_colors[d], alpha=POINT_ALPHA, label=str(d))
ax.set_title("ATAC UMAP — Donor identity (distinct colors)")
fixed_axes(ax, X_LIM, Y_LIM)
ax.legend(loc="center left", bbox_to_anchor=(1,0.5), fontsize=7, frameon=False, ncol=1)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "atac_umap_donor_identity_strongcolors.png"), dpi=300)
plt.close(fig)

if LABEL_DONORS:
    counts = pd.Series(donors).value_counts()
    donors_to_label = counts.head(TOP_N_LABELS).index.to_numpy() if TOP_N_LABELS is not None else uniq_donors
    centroids = {}
    for d in donors_to_label:
        idx = np.where(donors == d)[0]
        if idx.size > 0:
            centroids[d] = (float(np.mean(coords[idx,0])), float(np.mean(coords[idx,1])))

    fig, ax = plt.subplots(figsize=(8,6))
    for d in uniq_donors:
        idx = (donors == d)
        ax.scatter(coords[idx,0], coords[idx,1], s=POINT_SIZE,
                   color=donor_colors[d], alpha=POINT_ALPHA)
    for d,(x,y) in centroids.items():
        ax.text(x, y, str(d), fontsize=LABEL_FONTSIZE, ha="center", va="center",
                color="black", weight="bold",
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", boxstyle="round,pad=0.2"))
    ax.set_title("ATAC UMAP — Donor identity (labels, distinct colors)")
    fixed_axes(ax, X_LIM, Y_LIM)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "atac_umap_donor_identity_strongcolors_labeled.png"), dpi=300)
    plt.close(fig)

# ================================
# 3) AD vs Control (discrete)
# ================================
def plot_ad_ctrl(labels, title, outfile, focus=None):
    fig, ax = plt.subplots(figsize=(8,6))
    order = ["Other", "Control", "AD"]
    color_map = {"AD": AD_COLOR, "Control": CTRL_COLOR, "Other": OTHER_GRAY}
    alpha_map = {"AD": POINT_ALPHA, "Control": POINT_ALPHA, "Other": 0.18}
    if focus in ("AD","Control"):
        order = ["Other", ("Control" if focus=="AD" else "AD"), focus]
        alpha_map = {"AD": 0.18, "Control": 0.18, "Other": 0.08}
        alpha_map[focus] = POINT_ALPHA
    for cls in order:
        idx = (labels.values == cls)
        if np.any(idx):
            ax.scatter(coords[idx,0], coords[idx,1],
                       s=POINT_SIZE, color=color_map[cls], alpha=alpha_map[cls],
                       label=f"{cls} (n={int(idx.sum()):,})")
    ax.set_title(title)
    fixed_axes(ax, X_LIM, Y_LIM)
    ax.legend(loc="center left", bbox_to_anchor=(1,0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"✅ Saved: {outfile}")

diag_col = find_diag_col(adata)
if diag_col is None:
    print("⚠️ No diagnosis-like column found; skipping AD vs Control map.")
else:
    labels = make_ad_control_labels(adata.obs[diag_col])
    plot_ad_ctrl(labels, "ATAC UMAP — AD vs Control",
                 os.path.join(OUT_DIR, "atac_umap_AD_vs_Control.png"), focus=None)
    if SAVE_FOCUS_MAPS:
        plot_ad_ctrl(labels, "ATAC UMAP — AD highlighted",
                     os.path.join(OUT_DIR, "atac_umap_AD_only.png"), focus="AD")
        plot_ad_ctrl(labels, "ATAC UMAP — Control highlighted",
                     os.path.join(OUT_DIR, "atac_umap_Control_only.png"), focus="Control")

# ================================
# 4) Brain Region (from obs['brain.region'])
# ================================
if REGION_COL not in adata.obs:
    raise KeyError(f"'{REGION_COL}' not found in adata.obs.")
regions = adata.obs[REGION_COL].astype(str).fillna("Unknown").to_numpy()
uniq_regions = np.unique(regions)
region_colors = build_strong_palette(uniq_regions)

# Plain region map
fig, ax = plt.subplots(figsize=(8,6))
for r in uniq_regions:
    idx = (regions == r)
    ax.scatter(coords[idx,0], coords[idx,1], s=POINT_SIZE,
               color=region_colors[r], alpha=POINT_ALPHA, label=str(r))
ax.set_title(f"ATAC UMAP — Brain Region: {REGION_COL}")
fixed_axes(ax, X_LIM, Y_LIM)
ax.legend(loc="center left", bbox_to_anchor=(1,0.5), fontsize=7, frameon=False, ncol=1)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "atac_umap_region_strongcolors.png"), dpi=300)
plt.close(fig)

# Labeled region map (centroids)
region_counts = pd.Series(regions).value_counts()
regions_to_label = region_counts.head(TOP_N_LABELS).index.to_numpy() if TOP_N_LABELS is not None else uniq_regions
centroids_r = {}
for r in regions_to_label:
    idx = np.where(regions == r)[0]
    if idx.size > 0:
        centroids_r[r] = (float(np.mean(coords[idx,0])), float(np.mean(coords[idx,1])))

fig, ax = plt.subplots(figsize=(8,6))
for r in uniq_regions:
    idx = (regions == r)
    ax.scatter(coords[idx,0], coords[idx,1], s=POINT_SIZE,
               color=region_colors[r], alpha=POINT_ALPHA)
for r,(x,y) in centroids_r.items():
    ax.text(x, y, str(r), fontsize=LABEL_FONTSIZE, ha="center", va="center",
            color="black", weight="bold",
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", boxstyle="round,pad=0.2"))
ax.set_title(f"ATAC UMAP — Brain Region (labels): {REGION_COL}")
fixed_axes(ax, X_LIM, Y_LIM)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "atac_umap_region_strongcolors_labeled.png"), dpi=300)
plt.close(fig)

print("🎉 Done.")




