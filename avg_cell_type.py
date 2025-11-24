#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v9 LOESS (all cell types) — PDF OUTPUT VERSION
- For each chromosome:
    * Top panel: average (across ALL cell types) of per-bin importance.
    * Bottom panel: global average across ALL chromosomes and ALL cell types.
- Uniform nbins per chromosome (default 50).
- X-axis is bin index (1..nbins) for both panels.
- RAW line subtle (dark gray); FIT line emphasized (red).
- Taller figure (figsize=(10, 12)) and hspace=0.6.
- Outputs 22 PDFs total:
    <outdir-root>/avgimp_twoline_chrN.pdf  for chr1..chr22
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
FIT_COLOR = "red";     FIT_ALPHA = 1.0;  FIT_LW = 3.5; FIT_Z = 10


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
    imp_col  = _pick_col(df, ["frac_imp", "importance", "imp", "imp_score",
                              "score", "value"])

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

    bins = np.linspace(mn, mx, nbins + 1, endpoint=True)
    idx = np.clip(np.digitize(x, bins) - 1, 0, nbins - 1)

    means = np.array(
        [np.mean(y[idx == i]) if np.any(idx == i) else np.nan
         for i in range(nbins)]
    )
    return means


# ---- LOESS helpers ----
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


def plot_for_chrom(chrom: str,
                   chrom_bins: np.ndarray,
                   global_bins: np.ndarray,
                   out_pdf: Path,
                   nbins: int,
                   loess_frac: float) -> None:

    if not np.any(np.isfinite(chrom_bins)):
        print(f"[WARN] No finite data for {chrom}; skipping plot.")
        return

    x_bins = np.arange(1, nbins + 1, dtype=float)
    y_top = chrom_bins
    y_bot = global_bins

    y_top_fit = fit_loess(x_bins, y_top, frac=loess_frac)
    y_bot_fit = fit_loess(x_bins, y_bot, frac=loess_frac)

    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.0], hspace=0.6)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x_bins, y_top, color=RAW_COLOR, alpha=RAW_ALPHA,
             linewidth=RAW_LW, label="raw (all CT avg)")
    ax1.plot(x_bins, y_top_fit, color=FIT_COLOR, alpha=FIT_ALPHA,
             linewidth=FIT_LW, zorder=FIT_Z, label="LOESS fit")
    ax1.set_title(f"{chrom}: avg. importance over ALL cell types "
                  f"(equal {nbins} bins)")
    ax1.set_xlabel(f"Bin index (1..{nbins})")
    ax1.set_ylabel("avg. importance")
    ax1.grid(alpha=0.3)
    ax1.set_xlim(1, nbins)
    ax1.legend(loc="upper right", frameon=False)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x_bins, y_bot, color=RAW_COLOR, alpha=RAW_ALPHA,
             linewidth=RAW_LW, label="raw (global avg)")
    ax2.plot(x_bins, y_bot_fit, color=FIT_COLOR, alpha=FIT_ALPHA,
             linewidth=FIT_LW, zorder=FIT_Z, label="LOESS fit")
    ax2.set_title(f"Global: mean over ALL chromosomes & ALL cell types "
                  f"(equal {nbins} bins)")
    ax2.set_xlabel(f"Bin index (1..{nbins})")
    ax2.set_ylabel("avg. importance")
    ax2.grid(alpha=0.3)
    ax2.set_xlim(1, nbins)
    ax2.legend(loc="upper right", frameon=False)

    for ax in (ax1, ax2):
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
    print(f"[Saved] {out_pdf}")


def main():
    ap = argparse.ArgumentParser(
        description="v9 LOESS: per-chrom avg over ALL cell types; "
                    "global avg over ALL chrom & cell types."
    )
    ap.add_argument(
        "--outdir-root",
        type=Path,
        default=Path("/mnt/d/Lab_project/11_02/results_plots/avgimp_allct_loess_v1"),
        help="Output directory root for averaged LOESS plots"
    )
    ap.add_argument("--nbins", type=int, default=50)
    ap.add_argument("--loess-frac", type=float, default=0.3)
    ap.add_argument(
        "--mapping-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns: cell_type,path"
    )
    args = ap.parse_args()

    file_map: Dict[str, str] = {
        "astro":      "/mnt/d/Lab_project/11_02/result/astro/atac/common_peak_importance.csv",
        "L4_IT":      "/mnt/d/Lab_project/11_02/result/L4_IT/atac/common_peak_importance.csv",
        "L5_IT":      "/mnt/d/Lab_project/11_02/result/L5_IT/atac/common_peak_importance.csv",
        "L6_IT":      "/mnt/d/Lab_project/11_02/result/L6_IT/atac/common_peak_importance.csv",
        "L23_IT":     "/mnt/d/Lab_project/11_02/result/L23_IT/atac/common_peak_importance.csv",
        "microglia":  "/mnt/d/Lab_project/11_02/result/microglia/atac/common_peak_importance.csv",
        "oligo":      "/mnt/d/Lab_project/11_02/result/oligo/atac/common_peak_importance.csv",
        "Pvalb":      "/mnt/d/Lab_project/11_02/result/Pvalb/atac/common_peak_importance.csv",
        "Sst":        "/mnt/d/Lab_project/11_02/result/Sst/atac/common_peak_importance.csv",
        "Vip":        "/mnt/d/Lab_project/11_02/result/Vip/atac/common_peak_importance.csv",
    }

    if args.mapping_csv is not None:
        if not args.mapping_csv.exists():
            sys.exit(f"Mapping CSV not found: {args.mapping_csv}")
        tmp: Dict[str, str] = {}
        with open(args.mapping_csv, newline="", encoding="utf-8") as fh:
            rdr = csv.DictReader(fh)
            for row in rdr:
                ct = (row.get("cell_type")
                      or row.get("celltype")
                      or row.get("ct"))
                path = row.get("path") or row.get("file")
                if ct and path:
                    tmp[ct] = path
        if tmp:
            file_map = tmp

    chroms = [f"chr{i}" for i in range(1, 23)]
    nbins = args.nbins

    chrom_bins_by_ct: Dict[str, List[np.ndarray]] = {chrom: [] for chrom in chroms}
    global_rows: List[np.ndarray] = []

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

        for chrom in chroms:
            ch = bed[bed["chrom"] == chrom]
            if ch.empty:
                continue
            bins_vec = per_chrom_equal_bins(ch, nbins=nbins)
            chrom_bins_by_ct[chrom].append(bins_vec)
            global_rows.append(bins_vec)

    if not global_rows:
        sys.exit("[ERROR] No data found for any cell type / chromosome.")

    global_mat = np.vstack(global_rows)
    with np.errstate(invalid="ignore"):
        global_bins = np.nanmean(global_mat, axis=0)

    outdir = args.outdir_root
    outdir.mkdir(parents=True, exist_ok=True)

    for chrom in chroms:
        rows = chrom_bins_by_ct[chrom]
        if not rows:
            print(f"[WARN] No data for {chrom} across any cell type.")
            continue
        mat = np.vstack(rows)
        with np.errstate(invalid="ignore"):
            chrom_mean_bins = np.nanmean(mat, axis=0)

        out_pdf = outdir / f"avgimp_twoline_{chrom}.pdf"
        plot_for_chrom(
            chrom=chrom,
            chrom_bins=chrom_mean_bins,
            global_bins=global_bins,
            out_pdf=out_pdf,
            nbins=nbins,
            loess_frac=args.loess_frac,
        )


if __name__ == "__main__":
    main()



