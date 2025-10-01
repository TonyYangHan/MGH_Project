import scanpy as sc, numpy as np, pandas as pd, shap, gc, cupy as cp, scipy.sparse as sp, os
import rapids_singlecell as rsc, itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from anndata import AnnData
from tqdm import trange
import tensorflow as tf
from keras import models, layers, callbacks, optimizers, backend as K

# Limit GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(e)

# ============================================================================
# MIL Attention Layer
# ============================================================================
class AttentionMIL(layers.Layer):
    """
    Attention-based MIL aggregation.
    Learns which cells (instances) are important for bag-level prediction.
    """
    def __init__(self, L=128, D=64, **kwargs):
        super().__init__(**kwargs)
        self.L = L  # attention dimension
        self.D = D  # gated attention dimension
        
    def build(self, input_shape):
        feature_dim = input_shape[-1]
        
        # Attention weights
        self.attention_V = self.add_weight(
            name='attention_V',
            shape=(feature_dim, self.L),
            initializer='glorot_uniform',
            trainable=True
        )
        self.attention_U = self.add_weight(
            name='attention_U',
            shape=(feature_dim, self.L),
            initializer='glorot_uniform',
            trainable=True
        )
        self.attention_w = self.add_weight(
            name='attention_w',
            shape=(self.L, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, x):
        # x shape: (batch, n_instances, features)
        # Gated attention mechanism
        A_V = tf.nn.tanh(tf.tensordot(x, self.attention_V, axes=[[2], [0]]))  # (batch, n_instances, L)
        A_U = tf.nn.sigmoid(tf.tensordot(x, self.attention_U, axes=[[2], [0]]))  # (batch, n_instances, L)
        A = tf.tensordot(A_V * A_U, self.attention_w, axes=[[2], [0]])  # (batch, n_instances, 1)
        
        # Softmax attention weights
        A = tf.nn.softmax(A, axis=1)  # normalize across instances
        
        # Weighted sum of instances
        Z = tf.reduce_sum(A * x, axis=1)  # (batch, features)
        
        return Z, A
    
    def get_config(self):
        config = super().get_config()
        config.update({"L": self.L, "D": self.D})
        return config


def create_mil_model(input_dim, L=64, D=32):
    """
    MIL model with attention aggregation.
    Reduced hidden dimensions for faster training.
    
    Architecture:
    1. Instance-level embedding (shared across all cells)
    2. Attention aggregation (learns which cells matter)
    3. Bag-level classifier
    """
    # Instance-level encoder
    instance_input = layers.Input(shape=(None, input_dim))  # (batch, n_cells, features)
    
    # Shared instance embedding - reduced to 64 -> 32
    h = layers.TimeDistributed(layers.Dense(64, activation='elu'))(instance_input)
    h = layers.TimeDistributed(layers.Dropout(0.2))(h)
    h = layers.TimeDistributed(layers.Dense(32, activation='elu'))(h)
    
    # Attention aggregation
    attention_mil = AttentionMIL(L=L, D=D)
    bag_representation, attention_weights = attention_mil(h)
    
    # Bag-level classifier - simplified
    output = layers.Dense(16, activation='elu')(bag_representation)
    output = layers.Dropout(0.3)(output)
    output = layers.Dense(1, activation='sigmoid')(output)
    
    model = models.Model(inputs=instance_input, outputs=[output, attention_weights])
    return model


def prepare_mil_batches(X, y, donor_ids, batch_donors=8, max_cells_per_donor=500, fixed_bag_size=None):
    """
    Prepare batches for MIL training.
    Each batch contains cells from multiple donors.
    
    Args:
        max_cells_per_donor: Subsample donors with many cells to speed up training
        fixed_bag_size: If provided, pad/truncate all bags to this size for faster training
    
    Returns:
        Generator yielding (X_batch, y_batch) where:
        - X_batch: list of arrays, each shape (n_cells_i, n_features)
        - y_batch: array of donor labels, shape (n_donors_in_batch,)
    """
    unique_donors = np.unique(donor_ids)
    np.random.shuffle(unique_donors)
    
    for i in range(0, len(unique_donors), batch_donors):
        batch_donor_list = unique_donors[i:i+batch_donors]
        
        X_bags = []
        y_bags = []
        
        for donor in batch_donor_list:
            donor_mask = donor_ids == donor
            X_donor = X[donor_mask]
            
            # Subsample or pad to fixed size if specified
            if fixed_bag_size is not None:
                if X_donor.shape[0] > fixed_bag_size:
                    subsample_idx = np.random.choice(X_donor.shape[0], fixed_bag_size, replace=False)
                    X_donor = X_donor[subsample_idx]
                elif X_donor.shape[0] < fixed_bag_size:
                    # Pad with zeros
                    padding = np.zeros((fixed_bag_size - X_donor.shape[0], X_donor.shape[1]), dtype=np.float32)
                    X_donor = np.vstack([X_donor, padding])
            elif X_donor.shape[0] > max_cells_per_donor:
                # Just subsample if no fixed size
                subsample_idx = np.random.choice(X_donor.shape[0], max_cells_per_donor, replace=False)
                X_donor = X_donor[subsample_idx]
            
            X_bags.append(X_donor)
            y_bags.append(y[donor_mask][0])  # all cells from same donor have same label
        
        yield X_bags, np.array(y_bags)


def pad_bags_to_batch(X_bags):
    """
    Pad bags to same length for batch processing.
    Returns: padded array (n_bags, max_n_cells, n_features) and mask
    """
    max_len = max(bag.shape[0] for bag in X_bags)
    n_features = X_bags[0].shape[1]
    
    padded = np.zeros((len(X_bags), max_len, n_features), dtype=np.float32)
    mask = np.zeros((len(X_bags), max_len), dtype=np.float32)
    
    for i, bag in enumerate(X_bags):
        n_cells = bag.shape[0]
        padded[i, :n_cells, :] = bag
        mask[i, :n_cells] = 1.0
    
    return padded, mask


# ============================================================================
# Main Training Loop
# ============================================================================

# 1. Load data
adata = sc.read_h5ad('../result_rna/aci_raw_RNA_filtered_v2.h5ad')

result_dir = "../result_rna/DEG_DNN_v4/"
os.makedirs(f'{result_dir}models', exist_ok=True)
os.makedirs(f'{result_dir}gene_importance', exist_ok=True)
os.makedirs(f'{result_dir}attention_weights', exist_ok=True)

def preprocess(adata: AnnData, frac, flavor, n_pcs, *, donor_key="orig.ident", min_donor_frac=0.5, min_per_donor=0.01):
    if not np.issubdtype(adata.X.dtype, np.floating): adata.X = adata.X.astype(np.float32)

    rsc.get.anndata_to_GPU(adata); rsc.pp.filter_cells(adata, min_genes=1000); rsc.get.anndata_to_CPU(adata)

    donors = adata.obs[donor_key].to_numpy(); U, inv = np.unique(donors, return_inverse=True)
    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    X_bin = X.copy(); X_bin.data[:] = 1
    pdp = np.vstack([(X_bin[inv==i].sum(axis=0).A1) / max(1, (inv==i).sum()) for i in range(len(U))]) if len(U) else np.zeros((0, X.shape[1]))
    keep = (pdp >= min_per_donor).mean(0) > min_donor_frac
    adata._inplace_subset_var(keep)

    rsc.get.anndata_to_GPU(adata)
    if frac < 1.0:
        rsc.pp.highly_variable_genes(adata, n_top_genes=int(adata.n_vars*frac), flavor=flavor)
        adata = adata[:, adata.var["highly_variable"]].copy()
    else:
        rsc.pp.filter_genes(adata, min_cells=max(1, int(0.01*adata.n_obs)))
    rsc.pp.normalize_total(adata, target_sum=1e4); rsc.pp.log1p(adata); rsc.tl.pca(adata, n_comps=n_pcs)

    rsc.get.anndata_to_CPU(adata); cp.get_default_memory_pool().free_all_blocks(); cp.get_default_pinned_memory_pool().free_all_blocks(); gc.collect()
    return adata

frac = 1.0
flavor = "pearson_residuals"
n_pcs = 30
adata = preprocess(adata, frac, flavor, n_pcs)
print(f"Data preprocessed: {adata.n_obs} cells, {adata.n_vars} genes.")
epochs = 150

ad_donors   = adata.obs.loc[adata.obs['AD'] == 'AD','orig.ident'].unique()
ctrl_donors = adata.obs.loc[adata.obs['AD'] == 'Control','orig.ident'].unique()
pairs = np.array(list(itertools.product(ad_donors, ctrl_donors)))
validation_folds = pairs.shape[0]
np.random.seed(42)
selected_pairs = np.random.choice(pairs.shape[0], size=validation_folds, replace=False)
aucs = list()

for fold_idx in trange(validation_folds, desc="Folds"):
    test_donors = pairs[selected_pairs[fold_idx]]
    is_test = adata.obs['orig.ident'].isin(test_donors)
    adata_test = adata[is_test].copy()
    adata_rest = adata[~is_test].copy()

    X_test = adata_test.obsm['X_pca']
    y_test = (adata_test.obs['AD'] == 'AD').astype(int).values
    test_donor_ids = adata_test.obs['orig.ident'].values

    X = adata_rest.obsm['X_pca']
    y = (adata_rest.obs['AD'] == 'AD').astype(int).values
    donor_ids = adata_rest.obs['orig.ident'].values

    # Split donors (not cells) into train/val
    unique_donors = np.unique(donor_ids)
    donor_labels = np.array([y[donor_ids == d][0] for d in unique_donors])
    train_donors, val_donors = train_test_split(
        unique_donors, test_size=0.2, random_state=42, stratify=donor_labels
    )

    train_mask = np.isin(donor_ids, train_donors)
    val_mask = np.isin(donor_ids, val_donors)

    X_train, y_train, train_donor_ids = X[train_mask], y[train_mask], donor_ids[train_mask]
    X_val, y_val, val_donor_ids = X[val_mask], y[val_mask], donor_ids[val_mask]

    # Create MIL model - reduced dimensions
    model = create_mil_model(input_dim=X_train.shape[1], L=64, D=32)
    
    adam = optimizers.Adam(learning_rate=1e-3)  # increased from 1e-4 for faster convergence
    model.compile(
        optimizer=adam,
        loss=['binary_crossentropy', lambda y_true, y_pred: 0.0],  # only loss on first output (predictions)
        loss_weights=[1.0, 0.0],  # ignore attention output loss
        metrics=[['accuracy'], []],  # only accuracy on first output
        jit_compile=False  # Disabled - XLA compilation is slow with variable length sequences
    )
    
    print(f"Fold {fold_idx}: Model compiled, starting training...")

    # Training with custom loop for MIL batching
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    patience = 5  # reduced from 10 for faster early stopping

    for epoch in range(epochs):
        # Training
        train_losses = []
        train_accs = []
        for X_bags, y_bags in prepare_mil_batches(X_train, y_train, train_donor_ids, batch_donors=16, max_cells_per_donor=300):  # increased batch size
            X_padded, _ = pad_bags_to_batch(X_bags)
            y_bags = y_bags.reshape(-1, 1)
            dummy_attn = np.zeros((len(y_bags), X_padded.shape[1], 1))
            
            loss = model.train_on_batch(X_padded, [y_bags, dummy_attn])
            train_losses.append(loss[1])  # loss[0] is total, loss[1] is prediction loss
            train_accs.append(loss[2])     # loss[2] is prediction accuracy
        
        # Validation - only every 2 epochs to save time
        if epoch % 2 == 0:
            val_losses = []
            val_accs = []
            for X_bags, y_bags in prepare_mil_batches(X_val, y_val, val_donor_ids, batch_donors=16, max_cells_per_donor=300):
                X_padded, _ = pad_bags_to_batch(X_bags)
                y_bags = y_bags.reshape(-1, 1)
                dummy_attn = np.zeros((len(y_bags), X_padded.shape[1], 1))
                
                loss = model.test_on_batch(X_padded, [y_bags, dummy_attn])
                val_losses.append(loss[1])
                val_accs.append(loss[2])
            
            val_loss = np.mean(val_losses)
            val_acc = np.mean(val_accs)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
                model.save(f'{result_dir}models/fold{fold_idx}_model.keras')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={np.mean(train_losses):.4f}, train_acc={np.mean(train_accs):.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    # Load best model
    model = models.load_model(
        f'{result_dir}models/fold{fold_idx}_model.keras',
        custom_objects={'AttentionMIL': AttentionMIL},
        safe_mode=False
    )

    # Test evaluation - bag level (subsample for speed)
    test_bag_preds = []
    test_bag_labels = []
    test_attention_weights = {}
    
    for donor in np.unique(test_donor_ids):
        donor_mask = test_donor_ids == donor
        X_donor = X_test[donor_mask]
        y_donor = y_test[donor_mask][0]
        
        # Subsample test donors with many cells
        if X_donor.shape[0] > 500:
            subsample_idx = np.random.choice(X_donor.shape[0], 500, replace=False)
            X_donor = X_donor[subsample_idx]
        
        X_padded = X_donor.reshape(1, X_donor.shape[0], X_donor.shape[1])
        pred, attn = model.predict(X_padded, verbose=0)
        
        test_bag_preds.append(pred[0, 0])
        test_bag_labels.append(y_donor)
        test_attention_weights[donor] = attn[0, :X_donor.shape[0], 0]  # remove padding

    test_auc = roc_auc_score(test_bag_labels, test_bag_preds)
    # Compute accuracy: compare thresholded predictions to labels elementwise
    test_acc = np.mean((np.array(test_bag_preds) > 0.5) == np.array(test_bag_labels))
    
    # Donor-level prediction via majority voting of cell predictions
    print("Computing donor-level predictions via cell-level majority voting...")
    donor_majority_preds = []
    donor_majority_labels = []
    
    for donor in np.unique(test_donor_ids):
        donor_mask = test_donor_ids == donor
        X_donor_cells = X_test[donor_mask]
        y_donor_true = y_test[donor_mask][0]
        
        # Get cell-level predictions for this donor
        cell_preds = []
        for cell_idx in range(len(X_donor_cells)):
            X_cell_bag = X_donor_cells[cell_idx:cell_idx+1].reshape(1, 1, -1)  # (1, 1, n_features)
            pred, _ = model.predict(X_cell_bag, verbose=0)
            cell_preds.append(pred[0, 0])
        
        # Majority voting: if >50% of cells predict AD (>0.5), predict donor as AD
        n_ad_cells = np.sum(np.array(cell_preds) > 0.5)
        donor_pred = 1 if n_ad_cells > len(cell_preds) / 2 else 0
        
        donor_majority_preds.append(donor_pred)
        donor_majority_labels.append(y_donor_true)
    
    # Compute donor-level accuracy from cell majority voting
    donor_majority_acc = np.mean(np.array(donor_majority_preds) == np.array(donor_majority_labels))
    print(f"Donor-level accuracy (cell majority voting): {donor_majority_acc:.4f}")
    
    aucs.append({
        'fold': fold_idx, 
        'auroc': test_auc, 
        'bag_accuracy': test_acc,
        'cell_majority_accuracy': donor_majority_acc,
        'val_accuracy': best_val_acc
    })
    
    # Save attention weights for interpretation
    attn_df = pd.DataFrame({
        'donor': list(test_attention_weights.keys()),
        'mean_attention': [w.mean() for w in test_attention_weights.values()],
        'max_attention': [w.max() for w in test_attention_weights.values()],
        'attention_weights': [w.tolist() for w in test_attention_weights.values()]
    })
    attn_df.to_csv(f'{result_dir}attention_weights/fold{fold_idx}_attention.csv', index=False)

    # Gene importance via SHAP (using KMeans centroids for stable background)
    # Use KMeans to find 100 representative centroids from training data
    n_clusters = min(100, len(X_train))  # Use 100 clusters or less if insufficient data
    print(f"Computing {n_clusters} KMeans centroids for SHAP background...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_train)
    background = kmeans.cluster_centers_  # Shape: (n_clusters, n_features)
    
    # Create instance-level model: PC features -> embedding -> logit (before sigmoid)
    # This explains importance of each PC for the cell-level representation
    print("Building instance-level model for SHAP...")
    instance_input = layers.Input(shape=(X_train.shape[1],))  # (n_features,)
    h = layers.Dense(64, activation='elu')(instance_input)
    h = layers.Dropout(0.2)(h)
    h = layers.Dense(32, activation='elu')(h)
    # Add final prediction layers (without attention, treat as single-cell prediction)
    logit = layers.Dense(16, activation='elu')(h)
    logit = layers.Dropout(0.3)(logit)
    logit = layers.Dense(1, activation='sigmoid')(logit)  # Sigmoid-processed logit
    
    instance_model = models.Model(inputs=instance_input, outputs=logit)
    
    # Copy weights from trained MIL model
    instance_model.layers[1].set_weights(model.layers[1].layer.get_weights())  # Dense(64)
    instance_model.layers[3].set_weights(model.layers[3].layer.get_weights())  # Dense(32)
    instance_model.layers[4].set_weights(model.layers[5].get_weights())        # Dense(16)
    instance_model.layers[6].set_weights(model.layers[7].get_weights())        # Dense(1)
    
    # Use DeepExplainer for fast SHAP computation
    print("Computing SHAP values for gene importance...")
    explainer = shap.DeepExplainer(instance_model, background)
    
    # Use full test set for SHAP computation
    shap_values = explainer.shap_values(X_test).squeeze()
    print(f"SHAP values have shape: {shap_values.shape}")
    
    # Average absolute SHAP values across test cells
    # shap_values shape: (n_samples, n_features) - one value per PC per sample
    avg_shap = np.mean(np.abs(shap_values), axis=0)  # (n_features,) - average importance per PC
    
    # Map back to genes via PCA loadings
    pc_load = adata_rest.varm['PCs']
    gene_scores = np.abs(pc_load) @ np.abs(avg_shap)  # (n_genes,)
    
    gene_df = pd.DataFrame({'Gene': adata_rest.var_names, 'Importance': gene_scores})
    gene_df = gene_df.sort_values('Importance', ascending=False)
    gene_df.to_csv(f'{result_dir}gene_importance/gene_shap_importance_fold{fold_idx}.csv', index=False)

    K.clear_session()
    del model, instance_model
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

metrics_df = pd.DataFrame(aucs)
metrics_df['test_pair'] = pairs[selected_pairs].tolist()
metrics_df.to_csv(f'{result_dir}fold_auroc.csv', index=False)
print(f"\n{'='*60}")
print(f"FINAL RESULTS ACROSS {validation_folds} FOLDS")
print(f"{'='*60}")
print(f"Mean AUROC: {metrics_df['auroc'].mean():.4f} ± {metrics_df['auroc'].std():.4f}")
print(f"Mean Bag-level Accuracy: {metrics_df['bag_accuracy'].mean():.4f} ± {metrics_df['bag_accuracy'].std():.4f}")
print(f"Mean Cell Majority Voting Accuracy: {metrics_df['cell_majority_accuracy'].mean():.4f} ± {metrics_df['cell_majority_accuracy'].std():.4f}")
print(f"Mean Val Accuracy: {metrics_df['val_accuracy'].mean():.4f} ± {metrics_df['val_accuracy'].std():.4f}")
print(f"{'='*60}\n")