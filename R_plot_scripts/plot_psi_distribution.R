#!/usr/bin/Rscript

# Function
sd_group <- function(x, c1, c2, c3){
  if (x[c3] < x[c1] - 2 * x[c2] | x[c3] > x[c1] + 2 * x[c2]) {
    group <- factor("2x SD")
  }
  else if (x[c3] < x[c1] - x[c2] | x[c3] > x[c1] + x[c2]) {
    group <- factor("1x SD")
  }
  else {
    group <- factor("within 1x SD")
  }
  return(group)
}

# Parser
args <- commandArgs(trailingOnly = TRUE)
# args[1] <- '~/DeepTFAS-in-Human/input/targets/'
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
}

# Read all data and save it as a single big dataframe
input_files <- dir(args[1], pattern = '*.target')
files_path <- file.path(args[1], input_files)
dat <- data.frame()
dat <- do.call(rbind, lapply(files_path, read.table))
colnames(dat) <- c("gene", "psi")

tissue_name <- tools::file_path_sans_ext(input_files)
gene_list <- unique(dat$gene)

# Create another dataframe for calculate mean and sd
df <- read.table(files_path[1], col.names = c("gene", tissue_name[1]))
for (ind in 2: length(tissue_name)){
  new_df <- read.table(files_path[ind], col.names = c("gene", tissue_name[ind]))
  df <- merge(df, new_df, by="gene", all = TRUE)
}

row.names(df) <- df$gene
df$gene = NULL
mean <- rowMeans(df, na.rm = TRUE)
sd <- matrixStats::rowSds(data.matrix(df), na.rm = TRUE)
# med <- Biobase::rowMedians(data.matrix(df), na.rm = TRUE)
df['mean'] <- mean
df['sd'] <- sd
# df['median'] <- med
df <- df[order(df$mean),]
gene_order <- rownames(df)

# Assign the mean and sd to original dataframe
dat$gene <- factor(dat$gene, levels = gene_order)
dat <- merge(dat, df[, c("mean", "sd")], by.x = "gene", by.y = "row.names", all = TRUE)
dat <- dat[order(dat$gene), ]
dat <- na.omit(dat)


dat$group <- apply(dat[, c("psi","mean","sd")], 1, sd_group, c1='mean', c2='sd', c3='psi')
# setwd('DeepTFAS-in-Human/')

# Plot the figure
library(ggplot2)
dir.create("./results", showWarnings = FALSE)
png(file = "./results/psi_distribution.png", width = 4.5, height = 6, units = "in", res = 300)
ggplot(dat, aes(psi, gene, colour = group))+
  geom_point(size = 0.01)+
  scale_color_manual(values=c("#808080", "#FF0000", "#56B4E9"))+
  theme(axis.text.y=element_blank(),
        axis.ticks.y=element_blank())
dev.off()
