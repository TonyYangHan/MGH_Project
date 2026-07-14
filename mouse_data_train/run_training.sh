set -e

python train.py micro_mouse_rna_training_v3_precomp_balanced.h5ad rna result_rna/

echo "Training completed successfully."