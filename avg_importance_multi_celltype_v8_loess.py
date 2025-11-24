#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v8 PDF version with custom colors:
- Save all results as PDF instead of PNG
- Use exact hex colors provided for each cell type
- No 250 kb lines
- Taller figure and more spacing
- Uniform nbins per chromosome
- Output folder: avgimp_by_celltype_loess_pdf_v1/<celltype>/
"""

from __future__ import annotations
import argparse, csv, re, sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


RAW_COLOR = "#555555"; RAW_ALPHA = 0.35; RAW_LW = 1.0
FIT_ALPHA = 1.0; FIT_LW = 3.5; FIT_Z = 10


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise ValueError(f"None of {candidates} found in {list(df.columns)}")


def parse_peak(s: str) -> Tuple[str, int, int]:
    m = re.match(r"^(chr[0-9XYM]+)[-_](\d+)[-_](\d+)$", str(s).strip())
    if not m:
        raise ValueError(f"Bad peak format: {s}")
    chrom, a, b = m.group(1), int(m.group(2)), int(m.group(3))
    if a > b:
        a, b = b, a
    return chrom, a, b


def load_table(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        header = fh.readline()
    sep = "," if header.count(",") >= header.count("\t") else "\t"

    df = pd.read_csv(path, sep=sep)

    peak_col = _pick_col(df, ["Peak", "peak", "region", "locus"])
    imp_col = _pick_col(df, ["frac_imp", "importance", "imp", "imp_score", "score", "value"])

    parsed = df[peak_col].apply(parse_peak)
    bed = pd.DataFrame(parsed.tolist(), columns=["chrom", "start", "end"])
    bed["mid"] = ((bed["start"] + bed["end"]) // 2).astype(np.int64)

    bed["frac_imp"] = pd.to_numeric(df[imp_col], errors="coerce").astype(float)
    bed = bed.dropna(subset=["frac_imp"])

    bed = bed[bed["chrom"].str.match(r"^chr([1-9]|1\d|2[0-2])$")]
    return bed


def per_chrom_equal_bins(chrom_df: pd.DataFrame, nbins: int) -> np.ndarray:
    x = chrom_df["mid"].to_numpy(dtype=float)
    y = chrom_df["frac_imp"].to_numpy(dtype=float)
    mn, mx = x.min(), x.max()
    if mx <= mn:
        return np.full(nbins, np.nan)

    bins = np.linspace(mn, mx, nbins + 1)
    idx = np.clip(np.digitize(x, bins) - 1, 0, nbins - 1)

    means = np.array([
        np.mean(y[idx == i]) if np.any(idx == i) else np.nan
        for i in range(nbins)
    ])
    return means


def compute_global_bin_means(bed: pd.DataFrame, nbins: int = 50) -> np.ndarray:
    rows = []
    for i in range(1, 23):
        ch = bed[bed["chrom"] == f"chr{i}"]
        if ch.empty:
            continue
        rows.append(per_chrom_equal_bins(ch, nbins))

    if not rows:
        return np.full(nbins, np.nan)

    mat = np.vstack(rows)
    with np.errstate(invalid="ignore"):
        return np.nanmean(mat, axis=0)


# ---- LOESS ----
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    _HAS_SM = True
except Exception:
    _HAS_SM = False


def fit_loess(x: np.ndarray, y: np.ndarray, frac: float = 0.3) -> np.ndarray:
    mask = np.isfinite(x) & np.isfinite(y)
    x0, y0 = x[mask], y[mask]
    if x0.size < 3:
        return y

    order = np.argsort(x0)
    x0, y0 = x0[order], y0[order]

    if _HAS_SM:
        res = sm_lowess(y0, x0, frac=frac, return_sorted=False)
        return np.interp(x, x0, res, left=res[0], right=res[-1])

    deg = 3 if x0.size >= 4 else max(1, x0.size - 1)
    coef = np.polyfit(x0, y0, deg=deg)
    return np.polyval(coef, x)


def plot_for_chrom(
    bed: pd.DataFrame,
    chrom: str,
    global_bins: np.ndarray,
    out_pdf: Path,
    nbins: int,
    loess_frac: float,
    fit_color: str
):
    ch = bed[bed["chrom"] == chrom]
    if ch.empty:
        print(f"[WARN] No data for {chrom}")
        return

    y_top = per_chrom_equal_bins(ch, nbins)
    x_bins = np.arange(1, nbins + 1)

    y_top_fit = fit_loess(x_bins, y_top, frac=loess_frac)
    y_bot_fit = fit_loess(x_bins, global_bins, frac=loess_frac)

    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.6)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x_bins, y_top, color=RAW_COLOR, alpha=RAW_ALPHA, linewidth=RAW_LW, label="raw")
    ax1.plot(x_bins, y_top_fit, color=fit_color, alpha=FIT_ALPHA, linewidth=FIT_LW, zorder=FIT_Z, label="fit")
    ax1.set_title(f"{chrom}: avg. importance (equal {nbins} bins)")
    ax1.set_xlabel(f"Bin index (1..{nbins})")
    ax1.set_ylabel("avg. importance")
    ax1.grid(alpha=0.3)
    ax1.set_xlim(1, nbins)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x_bins, global_bins, color=RAW_COLOR, alpha=RAW_ALPHA, linewidth=RAW_LW, label="raw global")
    ax2.plot(x_bins, y_bot_fit, color=fit_color, alpha=FIT_ALPHA, linewidth=FIT_LW, zorder=FIT_Z, label="fit global")
    ax2.set_title("Global mean over chromosomes (equal bins)")
    ax2.set_xlabel(f"Bin index (1..{nbins})")
    ax2.set_ylabel("avg. importance")
    ax2.grid(alpha=0.3)
    ax2.set_xlim(1, nbins)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
    print(f"[Saved] {out_pdf}")


def main():
    ap = argparse.ArgumentParser(description="v8 LOESS PDF version with provided color map.")
    ap.add_argument("--outdir-root", type=Path,
                    default=Path("/mnt/d/Lab_project/11_02/results_plots/avgimp_by_celltype_loess_pdf_v1"))
    ap.add_argument("--nbins", type=int, default=50)
    ap.add_argument("--loess-frac", type=float, default=0.3)
    args = ap.parse_args()


    # EXACT color map you provided
    ct_color: Dict[str, str] = {
        "astro":     "#a14a77",
        "L4_IT":     "#c053be",
        "L5_IT":     "#8dbb40",
        "L6_IT":     "#8263d7",
        "L23_IT":    "#50c468",
        "microglia": "#e283a3",
        "oligo":     "#818238",
        "Pvalb":     "#58c2ab",
        "Sst":       "#5c9e44",
        "Vip":       "#30866c",
    }

    # Data file locations
    file_map = {
        "astro": "/mnt/d/Lab_project/11_02/result/astro/atac/common_peak_importance.csv",
        "L4_IT": "/mnt/d/Lab_project/11_02/result/L4_IT/atac/common_peak_importance.csv",
        "L5_IT": "/mnt/d/Lab_project/11_02/result/L5_IT/atac/common_peak_importance.csv",
        "L6_IT": "/mnt/d/Lab_project/11_02/result/L6_IT/atac/common_peak_importance.csv",
        "L23_IT": "/mnt/d/Lab_project/11_02/result/L23_IT/atac/common_peak_importance.csv",
        "microglia": "/mnt/d/Lab_project/11_02/result/microglia/atac/common_peak_importance.csv",
        "oligo": "/mnt/d/Lab_project/11_02/result/oligo/atac/common_peak_importance.csv",
        "Pvalb": "/mnt/d/Lab_project/11_02/result/Pvalb/atac/common_peak_importance.csv",
        "Sst": "/mnt/d/Lab_project/11_02/result/Sst/atac/common_peak_importance.csv",
        "Vip": "/mnt/d/Lab_project/11_02/result/Vip/atac/common_peak_importance.csv",
    }


    chroms = [f"chr{i}" for i in range(1, 23)]

    for ct, path_str in file_map.items():
        p = Path(path_str)
        if not p.exists():
            print(f"[WARN] Missing file for {ct}: {p}")
            continue

        try:
            bed = load_table(p)
        except Exception as e:
            print(f"[ERROR] Failed loading {ct} ({p}): {e}")
            continue

        global_means = compute_global_bin_means(bed, nbins=args.nbins)

        outdir = args.outdir_root / ct
        outdir.mkdir(parents=True, exist_ok=True)

        color = ct_color[ct]

        for chrom in chroms:
            out_pdf = outdir / f"avgimp_twoline_{chrom}.pdf"
            plot_for_chrom(
                bed, chrom, global_means,
                out_pdf,
                nbins=args.nbins,
                loess_frac=args.loess_frac,
                fit_color=color
            )


if __name__ == "__main__":
    main()


