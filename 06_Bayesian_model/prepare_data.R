library(brms)
library(dplyr)
library(ggplot2)
library(tidyr)
library(sjPlot)
library(spdep)
library(sf)
library(sjPlot)
library(tidybayes)
library(emmeans)
library(stringr)

# TODO read original data frame and then add the oridinal surrfmean values back to the df by using the field_id column 

df <- read.csv('F:/Field_farm_size_relationship/data/test_run2/all_predictors_w_grassland_.csv')
df_orig <- read.csv('F:/Field_farm_size_relationship/data/test_run2/all_predictors_w_grassland_.csv')

df_full <- df
#df_full$surrf_mean <- df_orig$surrf_mean
#df <- df_full


# zweite Ziffer = functional diversity 
########################################
df$farm_size <- df$farm_size * 10000
df$field_size <- df$field_size * 10000




df$farm_size <- df$farm_size - df$field_size
(df$farm_size < 0) %>% sum()

df$log_farm_size <- log(df$farm_size)
df[which(df$farm_size==0),"log_farm_size"] = 0
df$FormerDDRmember <- as.factor(df$FormerDDRm)
df$log_field_size <- log(df$field_size)

which(df$farm_size==0) %>% length()

########################################
# farm size groups


#df$farm_size_groups <- cut(df$farm_size, breaks = c(0, 100, 200, 300, 400, 500, 600, 700, 800, 1000), include.lowest = TRUE) %>% as.integer()
#df[is.na(df$farm_size_groups),"farm_size_groups"] = 10

#unique(df$farm_size_groups)
#df$farm_size <- df$farm_size - df$field_size




df$federal_st <- as.factor(df$federal_st)
#recl_frame <- data.frame("old"=c("MA", "WW", "SU", "OR", "PO", "SC", "TR", "WB", "WR", "LE", 
#                   "AR", "GR", "FA", "PE", "UN", "GA", "MI"), 
#           "recl"=c("MA", "CE", "SU", "OR", "PO", "CE", "CE","CE","CE", "LE", "GR", "GR", "OT", "OT", "OT", "OT", "OT"))

#for(i in 1:nrow(recl_frame)) {
#  df[which(df$ID_KTYP == recl_frame[i, "old"]), "new_ID_KTYP"] <- recl_frame[i, "recl"]
#  
#}
df$new_IDKTYP <- as.factor(df$new_IDKTYP)

df$surrf_mean_log <- log(df$surrf_mean*10000)

which(is.na(df$surrf_std), arr.ind=TRUE)
df[which(df$surrf_mean_log < 0), "surrf_mean_log"] = 0
df[,c(3, 13:26, 30:31)] <- scale(df[,c(3, 13:26, 30:31)])


########################################
# write to disk for reproducibility 
#write.csv(df, 'full_merged_predict.csv')
df_full <- df

write.csv(df_full, '05_Bayesian_model/data/df_sample_new.csv')
########################################
