# Load required libraries
library(monocle)
library(dplyr)
library(ggplot2)
library(dendextend)
library(cluster)

cds <- readRDS("../broadQRS_ddrtree_cds_reordered21.rds")

Z <- cds@reducedDimS
Y <- cds@reducedDimK
branch <- as.numeric(cds$State)

results <- tibble(
  Z1 = Z[1, ],
  Z2 = Z[2, ],
  branch = branch,
  Pseudotime = cds$Pseudotime,
  filename = cds$filename)

# Read in remaining labels directly 

combined_all_45k <- read.csv("../combined_all_45k_chb_ppm.csv")
tree_proj_complete = merge(results, combined_all_45k, by = "filename") 

# Merging sub-branches to derive representative branches/phenogroups 
# using hierarchical clustering on tree coordinates

tree_proj_hierach <- tree_proj_complete %>% select(branch,Z1, Z2)
cols_to_scale <- c("Z1", "Z2")
tree_proj_hierach[, cols_to_scale] <- scale(tree_proj_hierach[, cols_to_scale])
tree_proj_hierach_grouped <- tree_proj_hierach %>%
  group_by(branch) %>%
  dplyr::summarise(across(everything(), mean))  %>% select(-branch)
rownames(tree_proj_hierach_grouped) = levels(tree_proj_hierach$branch)

dist_mat <- dist(bidmc_full_data_grouped, method = 'euclidean')
hclust_avg <- hclust(dist_mat, method = 'average')
avg_dend_obj <- as.dendrogram(hclust_avg)
avg_col_dend <- color_branches(avg_dend_obj, h = 0.75)
plot(avg_col_dend) # dendrogram colored by height to cut with 

# Plot Silhouette plot across resulting clusters
sil_cl <- silhouette(cutree(hclust_avg, h=0.75) ,dist_mat, title=title(main = 'Silhoutte Scores across Clusters at height _'))
rownames(sil_cl) = rownames(bidmc_full_data_grouped)
plot(sil_cl)

tree_proj_complete$merged_branchcoords = tree_proj_complete$branch %>%
  plyr::mapvalues(., c(1, 2), c(1, 1)) %>%
  plyr::mapvalues(., c(3, 4, 5, 6, 25, 26, 27), c(2, 2, 2, 2, 2, 2, 2)) %>%
  plyr::mapvalues(., c(7, 8, 9, 10, 23, 24), c(3, 3, 3, 3, 3, 3)) %>%
  plyr::mapvalues(., c(11, 12, 21), c(4,4,4)) %>%
  plyr::mapvalues(., c(13, 14, 15, 16, 17, 18, 19, 20), c(5, 5, 5, 5, 5, 5, 5, 5)) %>%
  plyr::mapvalues(., c(22), c(6))

tree_proj_complete$merged_branchcoords <-  factor(tree_proj_complete$merged_branchcoords, 
                                                       levels = c(1, 2, 3, 4, 5, 6))

# Visualise merged branches used as phenogroups for remaining analyses
tree_proj_complete$merged_branchcoords <- factor(tree_proj_complete$merged_branchcoords, 
                                                           labels = c("Branch 1", "Branch 2", "Branch 3",
                                                                      "Branch 4", "Branch 5", "Branch 6"))

gg_full <- ggplot(tree_proj_complete) +
  geom_point(aes(x=Z1, y=Z2, color=merged_branchcoords), alpha = 0.1, size = 2) + 
  scale_color_manual(values = c("firebrick3", "gold3", "forestgreen", "lightseagreen", "royalblue", "orchid4"), 
                     guide = guide_legend(override.aes = list(size = 5, alpha = 1))) + 
  labs(
    x = "Dimension 1",
    y = "Dimension 2") +
  theme_bw() +
  coord_cartesian(xlim = c(NA, 4), ylim = c(-4, NA)) +
  theme(
    legend.text = element_text(size = 16, face = "bold"),  # Bolden legend text
    legend.title = element_blank(),  # Bolden legend title
    axis.text = element_text(size = 14, face = "bold"),     # Bolden axis label text
    axis.title = element_text(size = 16, face = "bold"),    # Bolden x and y axis label text
    plot.title = element_text(size = 16, hjust = 0.5, face = "bold")  # Bolden plot title
  )

gg_full

# Save BIDMC DDRTree outputs for UKB data projection
write.csv(tree_proj_complete, "../tree_proj_complete_BIDMC.csv")
