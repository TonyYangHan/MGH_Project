suppressPackageStartupMessages({
	library(Seurat)
	library(harmony)
	library(dplyr)
	library(plyr)
	library(stringr)
	library(ggplot2)
	library(ComplexHeatmap)
	library(reticulate)
    library(Matrix)
})


try_set_params <- function(obj, theta, lambda, sigma, k, reso, dims_use) {
        set.seed(42)
        obj <- obj |>
                RunHarmony(group.by.vars = "orig.ident", theta = theta, 
                        reduction.use = "pca", lambda = lambda, sigma = sigma, 
                        max_iter = 30, verbose = FALSE) |>
                FindNeighbors(reduction = "harmony", dims = dims_use, k.param = k, verbose = FALSE) |>
                FindClusters(resolution = reso, algorithm = 4, random.seed = 1, cluster.name = "optm_clusters", verbose = FALSE) |>
                RunUMAP(reduction = "harmony", dims = dims_use, reduction.name = "umap_harmony", verbose = FALSE)
    return (obj)
}

assign_cell_type <- function(obj, celltype_markers){
	obj <- AddModuleScore(obj, features = celltype_markers, name = names(celltype_markers))
	l <- length(names(celltype_markers))
	type_names <- names(celltype_markers)
	score_cols <- rep(NA,l)

	for (i in 1:l){
			score_cols[i] <- paste0(type_names[i], i)
	}

	meta_scores <- obj@meta.data %>%
		select(optm_clusters, all_of(score_cols)) %>%
		group_by(optm_clusters) %>%
		dplyr::summarize(across(everything(), mean))
	
	meta_scores$assigned_celltype <- apply(meta_scores[,-1], 1, function(x) names(x)[which.max(x)])

	cluster_to_celltype <- meta_scores %>%
		select(optm_clusters, assigned_celltype)

	obj$celltype <- plyr::mapvalues(
	obj$optm_clusters,
	from = cluster_to_celltype$optm_clusters,
	to = cluster_to_celltype$assigned_celltype
	)

	return (list(
		seurat = obj,
		celltype_score = meta_scores
	))

}

plot_deg_volcano <- function(path, save_dir, fdr_thresh = 0.05, top = 10){
	for (file in list.files(path, pattern = "\\.csv$")){
		df <- read.csv(file.path(path, file))
		df$log10FDR <- -log10(df$FDR)
		df$color <- ifelse(df$FDR < fdr_thresh, "red", "black")
		top10 <- df[1:top,]

		p <- ggplot(df, aes(x = logFC, y = log10FDR, color = color)) +
		geom_point() + 
		geom_text_repel(
		data = top10,
		aes(label = X),  # assumes gene names are rownames
		size = 3,
		color = "blue"
		) +
		scale_color_identity() +
		labs(
			title = basename(file),
			x = "logFC",
			y = "-log10(FDR)"
		) + 
		theme_minimal() +
		theme(
		panel.background = element_rect(fill = "white", color = NA),
		plot.background = element_rect(fill = "white", color = NA)
		)

		outname <- paste0(save_dir, basename(file), "_volcano.png")
		ggsave(outname, plot = p)
	}
	return(NULL)
}



plot_deg_heatmap <- function(path, save_dir, top = 20){
	file_paths <- list.files(path, pattern = "*.csv", full.names = TRUE)

	top_genes <- c()
	for (f in file_paths) {
		df <- read.csv(f, row.names = 1)
		top_genes <- c(top_genes, rownames(slice_head(df, n = 20)))
	}
	top_genes <- unique(top_genes)

	# Step 2: Load full FDR values for top genes only
	fdr_mat <- sapply(file_paths, function(f) {
		df <- read.csv(f, row.names = 1)
		df[top_genes, "FDR"]
	})

	# Ensure correct row and column structure
	rownames(fdr_mat) <- top_genes
	colnames(fdr_mat) <- paste0("Cluster_", str_extract(basename(file_paths), "\\d+"))

	# Step 3: Compute -log10(FDR)
	log_fdr_mat <- -log10(fdr_mat + 1e-300)

	# Step 4: Order genes by cluster with most significant FDR
	cluster_order <- colnames(log_fdr_mat)
	gene_order <- cluster_order[apply(log_fdr_mat, 1, which.max)]
	log_fdr_mat <- log_fdr_mat[order(factor(gene_order, levels = cluster_order)), ]

	# Step 5: Plot
	png(paste0(save_dir,"top20_fdr_heatmap.png"), width = 2000, height = 3600, res = 300)
	p <- Heatmap(
		log_fdr_mat,
		name = "DEG_Heatmap",
		col = viridis(100),
		heatmap_legend_param = list(
			title = "-log10FDR"
		),
		cluster_rows = FALSE,
		cluster_columns = FALSE,
		show_row_names = TRUE,
		column_names_rot = 45
	)
	draw(p)
	dev.off()
	return (NULL)
}

convert_seurat <- function(obj_path, save_dir){
	obj <- readRDS(obj_path)
	writeMM(aci@assays$RNA$counts, file = file.path(save_dir,"counts.mtx"))
	write.csv(data.frame(Gene = rownames(aci)), file.path(save_dir,"genes.csv"), row.names = FALSE)
	write.csv(data.frame(Barcode = colnames(aci)), file.path(save_dir,"barcodes.csv"), row.names = FALSE)
	write.csv(aci@meta.data, file.path(save_dir,"meta_data.csv"))
	return (NULL)
}

convert_seurat_2 <- function(aci, save_dir, save_name){
	counts <- LayerData(aci, layer = "counts")
	writeMM(counts, file = file.path(save_dir,sprintf("%s_counts.mtx", save_name)))
	write.csv(data.frame(Gene = rownames(aci)), file.path(save_dir,sprintf("%s_genes.csv", save_name)), row.names = FALSE)
	write.csv(data.frame(Barcode = colnames(aci)), file.path(save_dir,sprintf("%s_barcodes.csv", save_name)), row.names = FALSE)
	write.csv(aci@meta.data, file.path(save_dir,sprintf("%s_meta_data.csv", save_name)))
	return (NULL)
}