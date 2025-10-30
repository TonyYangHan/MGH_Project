#!/usr/bin/env bash
set -euo pipefail

# Minimal runner: iterate over the provided cell types and run train.py on matching files
# Runs up to 3 jobs concurrently to save time.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/train.py"
AD_DIR="$SCRIPT_DIR/adata_objects"
RESULT_BASE="$SCRIPT_DIR/result"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"

[[ -f "$PYTHON_SCRIPT" ]] || { echo "Error: train.py not found at $PYTHON_SCRIPT" >&2; exit 1; }
mkdir -p "$RESULT_BASE"

# If config exists, pass it; otherwise rely on train.py defaults
cfg_args=()
[[ -f "$CONFIG_FILE" ]] && cfg_args=(--config "$CONFIG_FILE")

# Only use the provided list of cell types
CELL_TYPES=("oligo" "microglia" "astro" "L23_IT" "L4_IT" "L5_IT" "L6_IT" "Pvalb" "Sst" "Vip")

# Concurrency limit
MAX_JOBS=5

throttle() {
  # Wait until number of running jobs is below MAX_JOBS
  while [ "$(jobs -r -p | wc -l)" -ge "$MAX_JOBS" ]; do
    wait -n || true
  done
}

shopt -s nullglob
for ct in "${CELL_TYPES[@]}"; do
  for mod in rna atac; do
    for f in "$AD_DIR/${ct}_mgh_${mod}_"*.h5ad; do
      out_dir="$RESULT_BASE/$ct/$mod"
      mkdir -p "$out_dir"
      echo "Running: $(basename "$f") -> $out_dir"
      throttle
      python "$PYTHON_SCRIPT" "$f" "$mod" "$out_dir" "${cfg_args[@]}" &
    done
  done
done

# Wait for all background jobs to finish
wait || true

echo "Done. Outputs under: $RESULT_BASE/<cell_type>/<rna|atac>"
