library(brms)
library(dplyr)
library(ggplot2)
library(tidyr)
#library(sjPlot)
#library(spdep)
#library(sf)
#library(sjPlot)
library(tidybayes)
#library(emmeans)
library(stringr)

# TODO read original data frame and then add the oridinal surrfmean values back to the df by using the field_id column 

df <- read.csv('F:/Field_farm_size_relationship/data/tables/predictors/all_predictors_w_grassland_log_and_scaled.csv')
df_sample <- read.csv('F:/Field_farm_size_relationship/data/tables/predictors/all_field_ids_w_sample.csv')
df_full <- merge(df, df_sample, by="field_id")
#df_orig <- read.csv('F:/Field_farm_size_relationship/data/test_run2/all_predictors_w_grassland_.csv')


df_sample <- df_full[which(df_full$matched_sample == TRUE),]
hist(df_sample$log_field_size)
df_sample_sample <- sample_frac(df_sample, 0.01)


df$new_IDKTYP <- as.factor(df$new_ID_KTYP)

unique(df$new_IDKTYP)
