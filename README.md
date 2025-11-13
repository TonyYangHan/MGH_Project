# MGH_Project

Analyze 10x single-nucleus multi-omics sequencing data from Massachusetts General Hospital (MGH) Alzheimer’s disease (AD) brain samples. This repo provides:

- Reproducible training code to build and train a simple neural network classifier on PCA features from single-nucleus sequencing AnnData objects(.h5ad) with cross validation
- SHAP-based interpretation to calculate the contribution of each gene or peak to the classification of whether cells are from AD or Healthy donor
- Notebooks for interpreting results from both ML performance metrics and biological meanings
- Determine the genes and peaks that are considered important between cells from AD and healthy donors


## What’s in this repository

- `train/`
	- `train.py`: Train a multilayer perceptron (MLP) on PCA features stored in an AnnData object and produce metrics, model weights, and SHAP-based feature importances.
	- `config.yaml`: Default configuration for data keys, training hyperparameters, scheduler, and output locations.
	- `run_all.sh`: Convenience script to run many training jobs in parallel across cell types and modalities.
- `interpretation/`
	- Jupyter notebooks demonstrating how to interpret outputs from both ML and biology perspectives (`Evaluation_interpretation.ipynb`, `ML_interpret_joint_torch.ipynb`) plus helper utilities (`utils.py`).
- `omics_deep_ml.yml`: Conda environment spec so others can reproduce the software stack directly.
- Other folders (e.g., `preprocessing/`, `traditional_analysis/`, `tabular_graphical_results/`, `model_checkpoints/`) hold domain-specific analyses and results.


## Environment setup (reproducible)

Create and activate the provided environment using the Conda YAML:

```bash
conda env create -f omics_deep_ml.yml
conda activate omics_deep_ml
```

Notes:
- The environment targets GPU acceleration where available. CPU will also work but be slower.
- If you already have an environment, you can inspect `omics_deep_ml_simple.yml` and install only the bits you need.


## Data expectations for training

`train.py` expects a preprocessed AnnData `.h5ad` file with:

- PCA features in `adata.obsm["X_pca"]` (configurable via `data.pca_key`)
- PCA loadings in `adata.varm["PCs"]` (configurable via `data.pcs_loadings_key`)
- Sample/label metadata in `adata.obs`, including:
	- Disease/control label column (default `AD` with values `AD` and `Control`)
	- Donor identifier column (default `orig.ident`)

All of the above keys and labels can be changed in `train/config.yaml` or via a custom config passed to `train.py`.


## Training: `train/train.py`

Train a simple MLP on PCA features and compute SHAP importances.

Usage:

```bash
python train/train.py <adata.h5ad> <rna|atac> <result_dir> [--config path/to/config.yaml]
```

Positional arguments:
- `adata.h5ad`: Path to the input AnnData file.
- `rna|atac`: Analysis type. Controls output subfolders and feature labels (genes vs peaks).
- `result_dir`: Directory to write outputs (created if missing).

Optional arguments:
- `--config`: Path to a YAML config file. Default: `config.yaml` (relative to the working directory). The repo provides `train/config.yaml` as a sensible starting point.

Key configuration knobs (see `train/config.yaml`):
- Data keys: `data.pca_key`, `data.pcs_loadings_key`, `data.label_key`, `data.donor_key`, `data.ad_label`, `data.control_label`
- Training: `training.batch_size`, `training.max_epochs`, `training.lr`, `training.patience`, `training.val_split`, `training.stratify`, `training.seed`
- Scheduler and early stopping: `scheduler.*`
- Model: `model.hidden_sizes`, `model.activation`
- SHAP: `shap.background_kmeans`
- Output subdirs and filenames: `output.*`

Outputs written to `<result_dir>`:
- `models/`: `fold{i}_model.pt` PyTorch weights for the best model (determined by validation loss) of each validation fold
- `gene_importance/` or `peak_importance/`: CSV files like `{gene|peak}_shap_importance_fold{i}.csv` with per-feature importances
- `fold_metrics.csv`: Per-fold AUROC, Accuracy, threshold, and donor-pair index

Evaluation protocol (built into `train.py`):
- Enumerates all unique donor pairs {AD, Ctrl} and performs cross-validation by holding out pairs.
- Selects a classification threshold on the validation set using Youden’s J statistic (argmax of TPR − FPR).
- Reports AUROC and accuracy for the held-out donor pair.
- Note: Accuracy metrics is just for a reference. Since cells are donor-level labeled (i.e. all cells from an AD donor are labeled as "AD"), the discussion of accuracy that originates from hard-labeling of cells would be meaningless. Also, it is very difficult to select a correct threshold to obtain the hard labels. 

Examples to run the script on one modality for one cell type:

```bash
# Single run on RNA
python train/train.py /path/to/oligo_mgh_rna.h5ad rna ./results/oligo_rna --config train/config.yaml

# Single run on ATAC
python train/train.py /path/to/oligo_mgh_atac.h5ad atac ./results/oligo_atac --config train/config.yaml
```


## Batch runs: `train/run_all.sh`

`run_all.sh` demonstrates how to launch multiple trainings in parallel across cell types and modalities. By default it:

- Looks under `train/adata_objects/` for files named like `<celltype>_mgh_<rna|atac>_*.h5ad`
- Writes outputs under `train/result/<celltype>/<rna|atac>/`
- Passes `train/config.yaml` if present
- Limits concurrency via `MAX_JOBS`, which indicates how many jobs can run simultaneously at max.

Using it efficiently:

```bash
# Default behavior (no edits): looks for inputs under train/adata_objects/ and writes to train/result/
bash train/run_all.sh
```

Tips and customization:

- Inputs and outputs:
	- Place your precomputed `.h5ad` files in `train/adata_objects/` with names like `<celltype>_mgh_<rna|atac>_*.h5ad`, or edit `AD_DIR` to point to your project-level `adata_objects/` folder.
	- By default results go to `train/result/<celltype>/<rna|atac>/`. Edit `RESULT_BASE` to write to `result/` at project root instead.
- Cell types and modalities:
	- Edit `CELL_TYPES=("oligo" "microglia" "astro" "L23_IT" "L4_IT" "L5_IT" "L6_IT" "Pvalb" "Sst" "Vip")` to match your available data and file-name prefixes.
	- The script scans both `rna` and `atac` for each cell type; remove a modality from the inner loop if not needed.
- Concurrency and hardware:
	- Each training uses GPU when available and runs SHAP; on a single GPU, set `MAX_JOBS=1` to avoid OOM. Increase gradually if you have multiple GPUs or ample memory.
	- The script throttles background jobs and waits at the end. You can monitor progress via the echoed “Running:” lines.
- Quick sanity check:
	- Start with one cell type and `MAX_JOBS=1` to validate paths and naming, then widen to all cell types.


## Preprocessing: purpose and usage

The `preprocessing/` folder contains R and Python workflows to build the AnnData `.h5ad` inputs expected by `train/train.py`, along with helper utilities for clustering, cell type assignment, and DEG visualization.

Files and roles:

- `preprocessing_R.ipynb` (R)
	- Loads a Seurat object (e.g., `MGH.ALL.rds`).
	- Uses helper functions from `utils.R` to export matrix layers and metadata to text files via `convert_seurat_2(...)` into a components folder like `mgh_all_obj_components/` with a prefix (e.g., `MGH_atac_all`).
		- Outputs: `<prefix>_counts.mtx`, `<prefix>_genes.csv`, `<prefix>_barcodes.csv`, and `<prefix>_meta_data.csv`.
	- Includes an example to aggregate DEG result tables (`.tsv`) and write a combined gene list (e.g., `down_genes.txt`).

- `preprocessing_python.ipynb` (Python)
	- Assembles an AnnData object from the exported components and writes an `.h5ad` (e.g., `mgh_atac_all.h5ad`).
	- Per-cell-type preprocessing (“precomputed”):
		- Performs QC (min genes, max counts) and donor-prevalence filtering.
		- Computes PCA on GPU with `rapids_singlecell` when available, with CPU `scanpy` fallback (`use_gpu=False`).
		- Saves per-cell-type `.h5ad` files named `<celltype_alias>_mgh_<rna|atac>_v3_precomp.h5ad` plus barcodes/features lists.
	- Automated downsampling across requested cell types to balance cells per donor; writes downsampled `.h5ad` files and a summary CSV of cell counts per donor.
	- Public dataset example (microglia): loads an external `.h5ad`, filters samples (e.g., remove `earlyAD`), filters donors by minimum cells, preprocesses with `donor_key="subject"`, and saves outputs.

- `utils.R`
	- `try_set_params(obj, theta, lambda, sigma, k, reso, dims_use)`: Run Harmony integration (by `orig.ident`), build neighbor graph, cluster (Leiden), and compute UMAP with the provided parameters. Helpful for parameter sweeps.
	- `assign_cell_type(obj, celltype_markers)`: Compute module scores for marker lists, aggregate per cluster, and assign each cluster a cell type label; returns the updated Seurat object and a score table.
	- `plot_deg_volcano(path, save_dir, fdr_thresh=0.05, top=10)`: For each DEG CSV under `path`, plot a volcano with `logFC` vs `-log10(FDR)` and annotate the top rows. Saves PNGs to `save_dir`. Assumes a gene-name column; adjust labeling as needed.
	- `plot_deg_heatmap(path, save_dir, top=20)`: Build a heatmap of `-log10(FDR)` for the top genes across cluster-specific DEG CSVs and save to `top20_fdr_heatmap.png`.
	- `convert_seurat(obj_path, save_dir)` and `convert_seurat_2(aci, save_dir, save_name)`: Export counts (MatrixMarket), genes, barcodes, and metadata for consumption by the Python notebook. Prefer `convert_seurat_2` (the all-in-memory variant) for reliability.

Typical workflow:

1) Export components from Seurat (R notebook or R console)

```r
# In R (inside preprocessing_R.ipynb or a console)
source("preprocessing/utils.R")
mgh_all <- readRDS("/path/to/MGH.ALL.rds")
convert_seurat_2(mgh_all, "mgh_all_obj_components/", "MGH_atac_all")
```

2) Assemble and preprocess in Python

```python
# In preprocessing_python.ipynb
in_dir = "mgh_all_obj_components"  # from step 1
prefix = "MGH_atac_all"
out_dir = "./"

# Build AnnData from exported components
# writes: mgh_atac_all.h5ad

# Optionally, run per-cell-type preprocessing and downsampling
# writes: <alias>_mgh_<rna|atac>_v3_precomp.h5ad and summary CSVs
```

Notes:
- The Python notebook uses `rapids_singlecell` for GPU-accelerated QC/PCA when available; set `use_gpu=False` to use CPU (`scanpy`).
- Preprocessing produces PCA embeddings (`X_pca`) and loadings (`varm["PCs"]`) that `train/train.py` expects by default; if you change key names, update `train/config.yaml` accordingly.


## End-to-end: from raw AnnData to results

If you start with a raw AnnData object containing raw counts, follow these steps.

1) Preprocess to generate precomputed AnnData

- Create minimal folders used by the notebooks/scripts:

```bash
mkdir -p preprocessing/mgh_all_obj_components
mkdir -p adata_objects
```

- Export Seurat (if your source is an RDS Seurat object) in R:

```r
# In R (e.g., inside preprocessing_R.ipynb)
source("preprocessing/utils.R")
mgh_all <- readRDS("/path/to/MGH.ALL.rds")
dir.create("preprocessing/mgh_all_obj_components", showWarnings = FALSE)
convert_seurat_2(mgh_all, "preprocessing/mgh_all_obj_components/", "MGH_atac_all")
```

- Assemble and precompute in Python (inside `preprocessing/preprocessing_python.ipynb`):
	- Set input/output paths to write precomputed files directly to the project-level `adata_objects/` folder (no need for `train/adata_objects/`):

```python
in_dir = "preprocessing/mgh_all_obj_components"  # from R step above
prefix = "MGH_atac_all"
adata = ... # assemble as shown in notebook

# Per-cell-type precompute; write to adata_objects/
result_dir = "adata_objects/"
# This cell will write files like: adata_objects/<cell_alias>_mgh_<rna|atac>_v3_precomp.h5ad
```

2) Train the model and save results

- Single run:

```bash
python train/train.py adata_objects/oligo_mgh_rna_v3_precomp.h5ad rna result/oligo/rna --config train/config.yaml
```

- Batch across multiple cell types and modalities with the helper script:

```bash
mkdir -p adata_objects result
# Option A: use run_all.sh as-is (expects train/adata_objects and writes train/result)
bash train/run_all.sh

# Option B: if you want to keep inputs under adata_objects/ and outputs under result/ at project root,
# edit train/run_all.sh and set:
#   AD_DIR="$(git rev-parse --show-toplevel)/adata_objects"  # or absolute path
#   RESULT_BASE="$(git rev-parse --show-toplevel)/result"
```

Notes on folders created/required:
- `train/train.py` will create the provided `<result_dir>` and the needed subfolders: `models/` and either `gene_importance/` or `peak_importance/`.
- `train/run_all.sh` by default expects inputs under `train/adata_objects/` and writes to `train/result/<cell_type>/<rna|atac>/`. If you prefer `adata_objects/` and `result/` at project root, update `AD_DIR` and `RESULT_BASE` in the script (see Option B above) or create a symlink.

3) Interpret, visualize performance, and determine feature cutoffs

Primary option (recommended): use `interpretation/Eval_general.ipynb` to run the latest analysis automatically across all cell types and both modalities in one pass.

- Set the paths and lists at the top of the notebook:

```python
# If your adata_objects/ and result/ live at the project root, set:
root_dir = ""  # empty string means paths like "result/<ct>/<mod>/" and "adata_objects/..."

# Otherwise, point root_dir to a prefix folder that contains both adata_objects/ and result/
# e.g., root_dir = "combined_analysis/"

cell_types_alias = ["oligo", "astro", "microglia", "L23_IT", "L4_IT", "L5_IT", "L6_IT", "Pvalb", "Sst", "Vip"]
mods = ["rna", "atac"]
```

What it expects and produces per cell type/modality:
- Reads: `{root_dir}adata_objects/<ct>_mgh_<mod>_v3_precomp.h5ad` and `{root_dir}result/<ct>/<mod>/fold_metrics.csv`, plus `{feature_name}_importance/` and `models/` subfolders under `{root_dir}result/<ct>/<mod>/`.
- Writes in `{root_dir}result/<ct>/<mod>/`:
	- `fold_metrics_plot.png`, `donor_avg_metrics_plot.png`
	- `cell_proba.csv` and per-donor violin plots (`donor_proba_*.png`)
	- `common_{gene|peak}_importance.csv` and importance score plots
	- p-value/FDR plots for enrichment-based cutoff selection
- Also writes a summary CSV across cell types: `{root_dir}cell_types_feature_cutoffs.csv` with elbow- and p-value-based cutoffs.

Alternative (single analysis): if you prefer to run one cell type and one modality at a time, use `interpretation/Evaluation_interpretation.ipynb`. Set `ct`, `mod`, `adata` path, and `result_dir` at the top as shown above, then run through the cells for that pair.

4) Generate and visualize cell probabilities for both modalities

If you use `Eval_general.ipynb`, probabilities for both RNA and ATAC are generated automatically for all configured cell types. If you use the single-analysis notebook, repeat for `mod = "rna"` and `mod = "atac"`:

```python
# RNA
ct, mod = "oligo", "rna"
adata = sc.read_h5ad("adata_objects/oligo_mgh_rna_v3_precomp.h5ad")
result_dir = "result/oligo/rna/"

# ATAC
ct, mod = "oligo", "atac"
adata = sc.read_h5ad("adata_objects/oligo_mgh_atac_v3_precomp.h5ad")
result_dir = "result/oligo/atac/"
```

Use the same workflow in the notebook to load models from `result_dir/models/`, read thresholds from `fold_metrics.csv`, compute cell-wise probabilities, and plot donor-level violins and region-stratified distributions.


## Interpretation notebooks

Open the notebooks in `interpretation/` for downstream analysis:

- `Eval_general.ipynb` (recommended): Runs the most up-to-date, automated analysis across all configured cell types and both modalities in one run. Set `root_dir`, `cell_types_alias`, and `mods` at the top to match your folder layout; it reads precomputed inputs and training outputs, generates metrics/plots, computes per-cell probabilities, aggregates SHAP importances, and derives feature cutoffs. Use this as your main entry point.
- `Evaluation_interpretation.ipynb`: For running one cell type and one modality at a time. Set `ct`, `mod`, `adata` path, and `result_dir` accordingly.
- `ML_interpret_joint_torch.ipynb`: Generate and visualize single-cell AD probabilities and downstream plots; useful for exploratory ML/biology interpretation.


## Notes and tips

- GPU is recommended for faster SHAP computation and training; CPU works but will be slower.
- If your `.h5ad` uses different key names, update `train/config.yaml` or supply a custom config with `--config`.
- Keep an eye on `fold_metrics.csv` and the logged AUROC/accuracy to validate training behavior across donor pairs.


## License

See `LICENSE` for terms.

