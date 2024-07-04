# Load required libraries
library(monocle)
library(dplyr)
library(ggplot2)

broadqrs_data <- read.csv("../combined_all_45k_chb_ppm.csv")

# Selecting the latent ECG features from VAE
selected_columns <- c(
  "X20", "X42", "X41", "X8", "X7", "X22", "X50", "X40", "X31", "X29", "X21", 
  "X2", "X4", "X17", "X38", "X37", "X35", "X16", "X43", "X6", "X30", "X33", 
  "X15", "X9", "X18", "X19", "X23", "X25", "X1", "X12", "X46", "X34", "X45", 
  "X3", "X49", "X39", "X24", "X14", "X36", "X10", "X44", "X26", "X28", "X13", 
  "X48", "X11", "X27", "X47", "X5", "X51", "X32"
)
data_45k <- broadqrs_data %>% select(all_of(selected_columns))
labels_45k <- broadqrs_data %>% select(-all_of(selected_columns))

# Pre-processing for CellDataSet (CDS) format
data_45k_norm <- data.frame(scale(data_45k))
expression_mat <- as.matrix(t(data_45k_norm))
obs_metadata <- labels_45k
rownames(obs_metadata) <- colnames(expression_mat)
obs_metadata_df <- as.data.frame(obs_metadata)
feature_metadata <- data.frame(feature_names = colnames(data_45k), row.names = colnames(data_45k))

# Create CDS object
cds <- monocle::newCellDataSet(expression_mat,
                               phenoData = AnnotatedDataFrame(obs_metadata_df),
                               featureData = AnnotatedDataFrame(feature_metadata),
                               expressionFamily = uninormal())

# Create DDRTree and assigns branches
start_time <- Sys.time()

cds <- reduceDimension(cds,
                       reduction_method = "DDRTree", norm_method = "none",
                       pseudo_expr = 0, scaling = FALSE, verbose = TRUE,
                       relative_expr = FALSE, ncenter = 2000, maxIter = 100,
                       tol = 1e-6
)

cds <- orderCells(cds)
end_time <- Sys.time()
elapsed_time <- end_time - start_time

print(elapsed_time)

# Save DDRTree CDS object and visualise derived tree
plot <- plot_cell_trajectory(cds, color_by = "State")
ggsave("../broadQRS_allbranches_new.png", plot = plot, width = 8, height = 6, dpi = 300)
saveRDS(cds, "../broadQRS_ddrtree_cds.rds")

# Once the tree is run and analysed as in DDRTree_outputs, 
# pseudotime was re-ordered to begin from the core tree branch by loading the CDS object 
# and only running the following code to save the tree object
## cds <- orderCells(cds, root_state = 21) 





