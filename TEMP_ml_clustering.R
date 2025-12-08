library(tidymodels)
library(tidyclust)
library(factoextra)
library(dbscan)
library(ggdendro)



df_Substance <- read.csv("data_share/df_2010_bafg.csv", row.names = 1)




# Drop rows where "date" is NA
df_Substance <- df_Substance[!is.na(df_Substance$date), ]

# Print all column names
print(paste("All columns:", paste(names(df_Substance), collapse = ", ")))

# Sampling locations (title case)
unique(df_Substance$sampling_location)

# Rivers (title case)
unique(df_Substance$river)

# Rivers at the sampling location Koblenz
unique(df_Substance$river[df_Substance$sampling_location == "KOBLENZ"])



# Discharge units
unique(df_Substance$unit_discharge)

# Concentration units
unique(df_Substance$unit_conc)

# Substances
unique(df_Substance$substance)


df_Substance$substance[df_Substance$substance == "AS"] <- "As"

unique(df_Substance$substance)

# 
df_substances <- df_Substance[!is.na(df_Substance$conc), ]

ggplot(df_substances, aes(x = conc)) +
  geom_histogram(bins = 50, color = color_RUB_blue, fill = color_RUB_green) +
  labs(
    x = "Concentration (µg/L)",
    y = "Count",
    title = "Histogram of substance concentrations"
  )


df_substances_quant <- df_substances[df_substances$less_than_conc == "False", ]
ggplot(df_substances_quant, aes(x = conc)) +
  geom_histogram(bins = 50, color = color_RUB_blue, fill = color_RUB_green) +
  labs(
    x = "Concentration (µg/L)",
    y = "Count",
    title = "Histogram of substance concentrations"
  )



all_substances <- unique(df_substances_quant$substance)
df_iqr <- do.call(rbind, lapply(all_substances, function(substance) {
  df_sub <- df_substances_quant[df_substances_quant$substance == substance, ]
  q1 <- quantile(df_sub$conc, 0.25, na.rm = TRUE)
  q3 <- quantile(df_sub$conc, 0.75, na.rm = TRUE)
  iqr_substance <- q3 - q1
  lower <- q1 - 1.5 * iqr_substance
  upper <- q3 + 1.5 * iqr_substance
  df_sub[df_sub$conc >= lower & df_sub$conc <= upper, ]
}))


ggplot(df_iqr, aes(x = conc)) +
  geom_histogram(bins = 50, color = color_RUB_blue, fill = color_RUB_green) +
  labs(
    x = "Concentration (µg/L)",
    y = "Count",
    title = "Histogram of substance concentrations"
  )


library(RColorBrewer)

# Create a color palette
colors <- scales::hue_pal()(length(all_substances))
names(colors) <- all_substances

# Scatter plot: Discharge vs Concentration
gp_Concen1 <- ggplot(df_iqr, aes(x = discharge, y = conc, color = substance)) +
  geom_point(alpha = 0.7) +
  scale_color_manual(values = colors) +
  labs(x = "Discharge (m³/s)", y = "Concentration (µg/L)") +
  theme(legend.position = "right")

# Scatter plot: pH vs Concentration
gp_Concen2 <- ggplot(df_iqr, aes(x = ph, y = conc, color = substance)) +
  geom_point(alpha = 0.7) +
  scale_color_manual(values = colors) +
  labs(x = "pH (-)", y = NULL) +
  theme(legend.position = "none")

# Combine plots
gp_Concen1 + gp_Concen2 + plot_layout(guides = "collect")



# Filter data
df_filtered <- df_iqr[df_iqr$ph > 2, ]
df_k <- df_filtered[df_filtered$sampling_location == "KARLSRUHE", ]
df_k_l <- df_filtered[df_filtered$sampling_location == "LAUTERBOURG-KARLSRUHE", ]

# Create as many colors as needed
colors <- scales::hue_pal()(length(all_substances))
names(colors) <- all_substances

# Plot: Karlsruhe
gp_Karl <- ggplot(df_k, aes(x = discharge, y = conc, color = substance)) +
  geom_point(alpha = 0.7) +
  scale_color_manual(values = colors) +
  labs(title = "Karlsruhe", x = "Discharge (m³/s)", y = "Concentration (µg/L)") +
  theme(legend.position = "right") +
  ylim(-5000, 80000)

# Plot: Lauterbourg-Karlsruhe
gp_Laut <- ggplot(df_k_l, aes(x = discharge, y = conc, color = substance)) +
  geom_point(alpha = 0.7) +
  scale_color_manual(values = colors) +
  labs(title = "Lauterbourg-Karlsruhe", x = "Discharge (m³/s)") +
  theme(legend.position = "none",
        axis.text.y = element_blank(),
        axis.title.y = element_blank()) +
  ylim(-5000, 80000)

# Combine plots
gp_Karl + gp_Laut + plot_layout(guides = "collect")

# Print unique rivers
unique(df_k$river)
unique(df_k_l$river)

df_final <- df_filtered[df_filtered$sampling_location != "KARLSRUHE", ]


# PCA -------------
df_PCA <- df_final
df_PCA <- df_PCA[df_PCA$year >= 2015, ]

# Create "site" combining sampling_location and river
df_PCA$site <- paste(df_PCA$sampling_location, df_PCA$river, sep = "_")

# Pivot to wide format with median concentrations
df_PCA_Summary <- df_PCA |>
  group_by(site, substance) |>
  summarize(conc_median = median(conc, na.rm = TRUE), .groups = "drop") |>
  pivot_wider(names_from = substance, values_from = conc_median, values_fill = 0) |> 
  select(-`NA`) |>
  filter(site != "NA_NA")


mat_PCA_Summary <- df_PCA_Summary[,-1]
df_PCA_Final <- t(mat_PCA_Summary) |> as.data.frame()
colnames(df_PCA_Final) <- df_PCA_Summary$site


rcp_Clust <- recipe(~ ., data = df_PCA_Final |> t()) |>
  step_normalize(all_predictors())  # normalize all columns

# Prep and bake
df_Clust_Normal <- prep(rcp_Clust) |> bake(new_data = NULL) |> t()
colnames(df_Clust_Normal) <- colnames(df_PCA_Final)

rcp_Clust2 <- recipe(~ ., data = df_Clust_Normal) |>
  step_normalize(all_predictors())  # normalize all columns

# Prep and bake
df_Clust_Normal2 <- prep(rcp_Clust2) |> bake(new_data = NULL)

# Perform PCA
pca_Clust <- prcomp(df_Clust_Normal2, scale. = FALSE)  # already normalized

# Eigenvalues
eigen_Clust <- pca_Clust$sdev^2

# Scree plot
ggplot() +
  geom_point(aes(x = 1:length(eigen_Clust), y = eigen_Clust)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = color_TUD_pink) +
  labs(x = "PC", y = "Eigenvalue")



# Recipe for PCA on clustering data
rcp_Clust_PCA <- recipe(~ ., data = df_Clust_Normal) |>
  step_normalize(all_predictors()) |>  # normalize all numeric columns
  step_pca(all_predictors(), num_comp = 3)  # keep first 4 PCs

# Prep and bake
df_Clust_PCA <- prep(rcp_Clust_PCA) |> bake(new_data = NULL)
df_Clust_PCA$substance <- rownames(df_Clust_Normal)
ggplot(df_Clust_PCA, aes(x = .panel_x, y = .panel_y)) +
  geom_point(aes(shape = substance), color = color_RUB_blue) +
  scale_shape_manual(values = 1:length(unique(df_Clust_PCA$substance))) +
  geom_autodensity(alpha = 0.8, fill = color_RUB_green, color = color_RUB_blue) +
  facet_matrix(vars(-substance), layer.diag = 2)


# HIERARCHICAL CLUSTERING ------------

mdl_Hclust <- hier_clust(
  num_clusters = 6,
  linkage_method = "complete"  # or "single", "average", "ward.D2"
) |>
  set_engine("stats")

# Workflow
wflow_Hclust <- workflow() |>
  add_recipe(rcp_Clust_PCA) |>
  add_model(mdl_Hclust)

# Fit the model
fit_Hclust <- fit(wflow_Hclust, data = df_Clust_Normal)

# Extract cluster assignments
clst_Hclust <- fit_Hclust |>
  extract_cluster_assignment()



label_clst_Hclust <- clst_Hclust$.cluster
names(label_clst_Hclust) <- rownames(df_Clust_Normal)
mat_Dist <- dist(df_PCA)
df_Hc <- hclust(mat_Dist, method = "complete")
# Set labels (same order as df_Clust_Normal)
df_Hc$labels <- rownames(df_Clust_Normal)
# Convert to dendro format
dend_Data <- as.dendrogram(df_Hc) |> dendro_data()
df_clst_Match <- tibble(
  label = dend_Data$labels$label,
  cluster = label_clst_Hclust[dend_Data$labels$label]
)

# Join segment → label → cluster
df_Segment_Hclst <- dend_Data$segments %>%
  left_join(dend_Data$labels %>% select(label, x), by = "x") %>%
  left_join(df_clst_Match, by = "label")

# Fill NA cluster values upward in tree
df_Segment_Hclst$cluster <- zoo::na.locf(df_Segment_Hclst$cluster, fromLast = TRUE)

ggplot() +
  geom_segment(data = df_Segment_Hclst,
               aes(x = x, y = y, xend = xend, yend = yend,
                   color = factor(cluster)),
               linewidth = 0.7) +
  geom_text(data = dend_Data$labels,
            aes(x = x, y = y, label = label),
            hjust = 1, angle = 90, size = 3) +
  labs(title = "Dendrogram (Complete Linkage)",
       x = "", y = "Height", color = "Cluster") +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        panel.grid = element_blank())


df_Hclust_Plot <- cbind(df_PCA, clst_Hclust)
df_Hclust_Plot$substance <- rownames(df_Clust_Normal)
# Scatter plot: PC1 vs PC2
gp_Hclust1 <- ggplot(df_Hclust_Plot, aes(x = PC2, y = PC1, color = .cluster, shape = substance)) +
  geom_point(size = 2) +
  scale_shape_manual(values = 1:length(unique(df_Clust_PCA$substance))) +
  geom_mark_ellipse(aes(x = PC2, y = PC1, group = .cluster), alpha = 0.2, show.legend = FALSE) +
  labs(x = "PC2", y = "PC1")

# Scatter plot: PC1 vs PC3
gp_Hclust2 <- ggplot(df_Hclust_Plot, aes(x = PC3, y = PC1, color = .cluster, shape = substance)) +
  geom_point(size = 2) +
  scale_shape_manual(values = 1:length(unique(df_Clust_PCA$substance))) +
  geom_mark_ellipse(aes(x = PC3, y = PC1, group = .cluster), alpha = 0.2, show.legend = FALSE) +
  labs(x = "PC3", y = "PC1")
gp_Hclust1 + (gp_Hclust2 + theme(axis.text.y = element_blank(),
                                 axis.title.y = element_blank())) + plot_layout(guides = "collect")

# kmeans ----------------------

df_Kmeans_Elbow <- tibble(k = 2:8, wss_value = NA)
df_Kmeans_Silhouette <- tibble(
  k = 2:8, 
  avg_silhouette = NA
)

for (i in 2:8) {
  fit_Temp <- workflow() |>
    add_recipe(rcp_Clust_PCA) |>
    add_model(k_means(num_clusters = i) |> set_engine("stats")) |>
    fit(data = df_Clust_Normal)
  
  df_Kmeans_Elbow$wss_value[df_Kmeans_Elbow$k == i] <- fit_temp |> 
    extract_fit_engine() |> 
    pluck("tot.withinss")
  

  clusters <- fit_Temp |> extract_cluster_assignment()
  # Prepare PCA data
  df_PCA <- rcp_Clust_PCA |> prep() |> bake(new_data = NULL)
  
  # Calculate silhouette scores
  sil <- cluster::silhouette(as.numeric(clusters$.cluster), dist(df_PCA))
  
  # Store average silhouette width
  df_Kmeans_Silhouette$avg_silhouette[df_Kmeans_Silhouette$k == i] <- 
    mean(sil[, 3])
  
}

# Plot elbow curve
ggplot(wss, aes(x = k, y = wss_value)) +
  geom_line(color = color_RUB_blue) +
  geom_point(color = color_RUB_green) +
  labs(title = "Elbow Method", 
       x = "Number of Clusters (k)", 
       y = "Inertia (Within-Cluster Sum of Squares)")


ggplot(df_Kmeans_Silhouette, aes(x = k, y = avg_silhouette)) +
  geom_line(color = color_RUB_blue) +
  geom_point(color = color_RUB_green) +
  labs(
    title = "Average Silhouette Score",
    x = "Number of Clusters (k)",
    y = "Average Silhouette Width"
  )


# Model specification
mdl_Kmeans <- k_means(num_clusters = 6)

# Workflow
wflow_Kmeans <- workflow() |>
  add_recipe(rcp_Clust_PCA) |>
  add_model(mdl_Kmeans)

# Fit the model
fit_Kmeans <- fit(wflow_Kmeans, data = df_Clust_Normal)
fit_Kmeans$fit
# Extract cluster assignments
clst_Kmeans <- fit_Kmeans |>
  extract_cluster_assignment()
clst_Kmeans_Centroids <- fit_Kmeans |> extract_centroids()

df_Kmeans_Plot <- cbind(df_PCA, clst_Kmeans)
df_Kmeans_Plot$substance <- rownames(df_Clust_Normal)
# Scatter plot: PC1 vs PC2
gp_Kmeans1 <- ggplot(df_Kmeans_Plot, aes(x = PC2, y = PC1, color = .cluster, shape = substance)) +
  geom_point(size = 2) +
  scale_shape_manual(values = 1:length(unique(df_Clust_PCA$substance))) +
  geom_point(data = clst_Kmeans_Centroids, 
             aes(x = PC2, y = PC1, color = .cluster), 
             shape = "X", size = 4) +
  geom_mark_ellipse(aes(x = PC2, y = PC1, group = .cluster), alpha = 0.2, show.legend = FALSE) +
  labs(x = "PC2", y = "PC1")

# Scatter plot: PC1 vs PC3
gp_Kmeans2 <- ggplot(df_Kmeans_Plot, aes(x = PC3, y = PC1, color = .cluster, shape = substance)) +
  geom_point(size = 2) +
  scale_shape_manual(values = 1:length(unique(df_Clust_PCA$substance))) +
  geom_point(data = clst_Kmeans_Centroids, 
             aes(x = PC3, y = PC1, color = .cluster), 
             shape = "X", size = 4) +
  geom_mark_ellipse(aes(x = PC3, y = PC1, group = .cluster), alpha = 0.2, show.legend = FALSE) +
  labs(x = "PC3", y = "PC1")
gp_Kmeans1 + (gp_Kmeans2 + theme(axis.text.y = element_blank(),
                                 axis.title.y = element_blank())) + plot_layout(guides = "collect")

# dbscan -----------

fit_DBSCAN <- dbscan(df_PCA, eps = 2, minPts = 2)
clst_DBSCAN <- paste0("Cluster_", fit_DBSCAN$cluster)
clst_DBSCAN[clst_DBSCAN == "Cluster_0"] <- NA


df_DBSCAN_Plot <- cbind(df_PCA, .cluster = clst_DBSCAN)
df_DBSCAN_Plot$substance <- rownames(df_Clust_Normal)
# Scatter plot: PC1 vs PC2
gp_DBSCAN1 <- ggplot(df_DBSCAN_Plot, aes(x = PC2, y = PC1, color = .cluster, shape = substance)) +
  geom_point(size = 2) +
  scale_shape_manual(values = 1:length(unique(df_Clust_PCA$substance))) +
  geom_mark_ellipse(aes(x = PC2, y = PC1, group = .cluster), alpha = 0.2, show.legend = FALSE) +
  labs(x = "PC2", y = "PC1")

# Scatter plot: PC1 vs PC3
gp_DBSCAN2 <- ggplot(df_DBSCAN_Plot, aes(x = PC3, y = PC1, color = .cluster, shape = substance)) +
  geom_point(size = 2) +
  scale_shape_manual(values = 1:length(unique(df_Clust_PCA$substance))) +
  geom_mark_ellipse(aes(x = PC3, y = PC1, group = .cluster), alpha = 0.2, show.legend = FALSE) +
  labs(x = "PC3", y = "PC1")
gp_DBSCAN1 + (gp_DBSCAN2 + theme(axis.text.y = element_blank(),
                                 axis.title.y = element_blank())) + plot_layout(guides = "collect")








library(tidymodels)
library(biclust)
library(blockcluster)
library(ggplot2)
library(pheatmap)
library(reshape2)













