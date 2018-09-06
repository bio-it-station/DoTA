#!/usr/bin/Rscript

# Parser
args <- commandArgs(trailingOnly = TRUE)
# args[1] <- '~/DoTA/output/ks_test_preprocess'
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
}

tf <- list.files(args[1])
files <- file.path(args[1], tf)

df <- data.frame(row.names = tf)
df['feature_0_1'] <- NA
df['statistic'] <- NA

for (i in 1: length(tf)){
  f_0 <- scan(files[i], nlines=1, what = double(), quiet = TRUE)
  f_1 <- scan(files[i], skip = 1, nlines = 2, what = double(), quiet = TRUE)
  
  ks_0_and_1 <- ks.test(f_0, f_1)
  
  df$feature_0_1[i] <- ks_0_and_1$p.value
  df$statistic[i] <- ks_0_and_1$statistic
}

adj_0_1 <- p.adjust(df[, 1])
adj_df <- data.frame(matrix(ncol=0,nrow=length(tf)))
adj_df['tf'] <- tf
adj_df['statistic'] <- df$statistic
adj_df['p_value'] <- df$feature_0_1
adj_df['adjusted_p_value'] <- adj_0_1
pass_adj <- adj_df[adj_df$adjusted_p_value < 1e-7, ]

save_dir <- ('../results/ks_test_result')
dir.create(save_dir, showWarnings = FALSE)
dir.create(file.path(save_dir,'pass'), showWarnings = FALSE)
dir.create(file.path(save_dir,'not_pass'), showWarnings = FALSE)
write.table(pass_adj, file = file.path(save_dir, "ks_table.csv"),
            quote = FALSE, sep = '\t', row.names = FALSE)

aCDFcolor <- rgb(1,0,0)
bCDFcolor <- rgb(0,0,1)
files <- file.path(dir, row.names(df))
for (i in files){
  tf_name <- basename(i)
  f_0 <- scan(i, nlines=1, what = double(), quiet = TRUE)
  f_1 <- scan(i, skip = 1, nlines = 2, what = double(), quiet = TRUE)
  if (tf_name %in% pass_adj$tf){
    sub_folder <- ('pass')
  }
  else{
    sub_folder <- ('not_pass')
  }
  file_name <- paste0(tf_name, ".png")
  png(filename = file.path(save_dir, sub_folder, file_name), width = 640, height = 480)
  plot(ecdf(f_0), xlab = 'psi', ylab = 'percent', col = aCDFcolor, main = tf_name, cex = .2)
  plot(ecdf(f_1), col = bCDFcolor, cex = .2, add = T)
  legend('right', c('unchanged', 'changed'), fill=c(aCDFcolor, bCDFcolor), border=NA)
  dev.off()
}
