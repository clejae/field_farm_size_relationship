df_full <- df
df_sub_1 <- df_full[-which(is.na(df_full$SQRAvrg)),]
df_sub_1$farm_size_ha <- (df_sub_1$farm_size * 0.0001) + 1
which(df_sub_1$farm_size == 0) %>% length()

by_par = 10000

for(i in seq(by_par, nrow(df_sub_1), by = by_par)) {
  print(i)
  preceding = (i - by_par)+1
  print(paste(preceding, i))
  slicer = preceding:i
  preds <- posterior_predict(model_surrf_lognorm_7_full, newdata = df_sub_1[slicer, ]) %>% t() %>% as.data.frame()
  preds_mean <- apply(preds, 1, median)
  lb_05 <- apply(preds, 1, quantile, probs = 0.05)
  ub_95 <- apply(preds, 1, quantile, probs = 0.95)
  lb_1 <- apply(preds, 1, quantile, probs = 0.1)
  ub_9 <- apply(preds, 1, quantile, probs = 0.9)
  df_sub_1[slicer, "lb_05"] <- lb_05
  df_sub_1[slicer, "ub_95"] <- ub_95
  df_sub_1[slicer, "lb_1"] <- lb_1
  df_sub_1[slicer, "ub_9"] <- ub_9
  df_sub_1[slicer, "preds"] <- preds_mean 
  df_sub_1[slicer, "resid"] <- df_sub_1[slicer, "farm_size_ha"] - preds_mean
}
write.csv(df_sub_1, 'preds_full.csv')
df_sub_1 <- read.csv('preds_full.csv')

df_sub_1$IQR <- df_sub_1$ub_9 - df_sub_1$lb_1
df_sub_1 <- df_sub_1 %>% na.omit()
df_full_shp <- read_sf('F:/Field_farm_size_relationship/data/test_run2/all_predictors_w_grassland.shp')

#merged <- merge(df_full_shp, df_sub_1, by="field_id")
merged <- merge(df_full_shp, df_sub_1, by.x = "field_id", by.y = "field_id")
#merged <- merged[-which(merged$resid < -10000),]
ggplot(data=merged %>% sample_frac(0.001),aes(x=farm_size_ha, y=preds)) + geom_point() + 
  #geom_pointrange(data=merged %>% sample_frac(0.001),aes(x=farm_size, y=preds, ymin=lb_05, ymax=ub_95)) + 
  xlim(0, 1000) + ylim(0, 1000) + geom_smooth(method='lm')

write_sf(obj=merged, dsn="05_Bayesian_model/data/preds_full_sp.gpkg")


df_sub_1 %>% filter(preds < 10000) %>% sample_frac(0.01) %>%
  ggplot(., aes(fieldSizeM*0.0001, log(preds), col=resid)) + 
  geom_point() + geom_smooth(method='lm') #+ 
  ylim(0, 200) + xlim(0, 200)

  df_sub_1 %>% filter(preds < 10000) %>% sample_frac(0.01) %>%
    ggplot(., aes(fieldSizeM*0.0001, log(farm_size_ha), col=resid)) + 
    geom_point() + geom_smooth(method='lm') #+ 
  ylim(0, 200) + xlim(0, 200)
  

mean(df_sub_1$farm_size_ha)
mean(df_sub_1$preds, na.rm=T)
median(df_sub_1$farm_size_ha)
median(df_sub_1$preds, na.rm=T)

RMSE <- sqrt(sum(df_sub_1$resid, na.rm=TRUE)/nrow(df_sub_1))

sum(abs(df_sub_1$resid))/nrow(df_sub_1)

for(crop in crop_list) {
  sub_ds = df_sub_1[which(df_sub_1$new_IDKTYP==crop),]
  MAPE <- (sum(abs((sub_ds$farm_size_ha - sub_ds$preds) / sub_ds$farm_size_ha), na.rm=TRUE))/nrow(sub_ds)*100
  R2 <- 1 -( sum(sub_ds$resid^2, na.rm=TRUE) /sum((sub_ds$farm_size_ha - mean(sub_ds$farm_size_ha)^2)))
  
  print(paste(crop, MAPE))
  print(paste(crop, R2))
}

for(state in unique(df$federal_st)) {
  sub_ds = df_sub_1[which(df_sub_1$federal_st==state),]
  MAPE <- (sum(abs((sub_ds$farm_size_ha - sub_ds$preds) / sub_ds$farm_size_ha), na.rm=TRUE))/nrow(sub_ds)*100
  R2 <- 1 -( sum(sub_ds$resid^2, na.rm=TRUE) /sum((sub_ds$farm_size_ha - mean(sub_ds$farm_size_ha)^2)))
  
  print(paste(state, MAPE))
  print(paste(state, R2))
}

df_sub_1 %>% filter(preds > 2000)

ggplot(df_sample_large %>% sample_frac(., 0.25) %>% filter(new_IDKTYP == 'GR', federal_st=='TH'), 
       aes((log_field_size), (farm_size_ha))) + geom_point() + xlim(-6, 5) +
  geom_line(data=newdata %>% filter(new_IDKTYP == 'GR', federal_st=='TH'), aes(x=log_field_size, y=(pred))) + 
  geom_smooth(method='gam')

ggplot(df, aes((log_field_size), (farm_size_ha*10000))) + geom_point() + geom_smooth(method = 'gam')

R2 <- 1 -( sum(df_sub_1$resid^2, na.rm=TRUE) /sum((df_sub_1$farm_size_ha - mean(df_sub_1$farm_size_ha)^2)))
MAPE <- (sum((abs(df_sub_1$preds - df_sub_1$farm_size_ha) / df_sub_1$farm_size_ha), na.rm=TRUE))/nrow(df_sub_1)*100
MAPE
df_sample_large %>% #filter( federal_st=='TH') %>%
  ggplot(data=., aes(log_farm_size,col = federal_st)) + geom_histogram(bins=50) + facet_wrap( ~new_IDKTYP)
  summarise(med = mean(farm_size_ha))
 
hist()

bayes_R2(model_surrf_lognorm_7, robust=TRUE)
WAIC(model_surrf_lognorm_7)


df_sub_hist <- df_sub_1[,c("preds", "farm_size_ha")]
df_sub_hist <- gather(df_sub_hist)
ggplot(df_sub_hist, aes(value, fill=key)) + 
  geom_density(alpha=0.5) + xlim(0, 500)


df_sub_hist_small <- posterior_predict(model_surrf_lognorm_7) %>% apply(. , 2, median)
df$preds <- df_sub_hist_small
df_sub_hist_small <- df[,c("preds", "farm_size_ha")]
df_sub_hist_small <- gather(df_sub_hist_small)
ggplot(df_sub_hist_small, aes(value, fill=key)) + 
  geom_density(alpha=0.5) + xlim(0, 500)


df_sub_1 %>% filter(fieldSizeM*0.0001 > 10, fieldSizeM*0.0001 > 11, new_IDKTYP=='CE', federal_st=='BV') %>%
  ggplot(., aes(preds)) + geom_histogram(bins=100)
df_sub_1 %>% filter(fieldSizeM*0.0001 > 10, fieldSizeM*0.0001 > 11, new_IDKTYP=='CE', federal_st=='BV') %>%
  ggplot(., aes(farm_size_ha)) + geom_histogram(bins=100)

df_sub_1$matched_sa


ggplot() + geom_point(data= df_sub_1 %>% filter(new_IDKTYP=='CE', federal_st=='BV'), 
                           aes(log_field_size,  farm_size_ha, col=as.factor(new_IDKTYP))) + 
  geom_line(data = newdata %>% filter(new_IDKTYP == 'CE', federal_st=='BV') , aes((log_field_size), pred), col='green') +
    geom_ribbon(data =newdata %>% filter(new_IDKTYP == 'CE', federal_st=='BV'),
                aes(x = log_field_size, ymin = lower_bound1, ymax = upper_bound1, 
                    col=federal_st),
                alpha = 0.1) 


ggplot() + geom_point(data= df_sub_1 %>% filter(new_IDKTYP=='CE', federal_st=='BV'), 
                      aes(log_field_size,  log(farm_size_ha), col=as.factor(new_IDKTYP))) + 
  geom_line(data = newdata %>% filter(new_IDKTYP == 'CE', federal_st=='BV') , aes((log_field_size), log(pred)), col='green') +
  geom_ribbon(data =newdata %>% filter(new_IDKTYP == 'CE', federal_st=='BV'),
              aes(x = log_field_size, ymin = log(lower_bound1), ymax = log(upper_bound1), 
                  col=federal_st),
              alpha = 0.1) 


ggplot(data= df_sub_1 %>% filter(new_IDKTYP=='CE', federal_st=='BV'), 
       aes(x=log_field_size,  y=log(farm_size_ha))) + geom_hex() +
  scale_fill_continuous(type = "viridis") +
  theme_bw() + 
  geom_line(data = newdata %>% filter(new_IDKTYP == 'CE', federal_st=='BV') , 
            aes((log_field_size), log(pred)), col='green') + 
  geom_ribbon(data =newdata %>% filter(new_IDKTYP == 'CE', federal_st=='BV'),
              aes(x = log_field_size, ymin = lower_bound1, ymax = upper_bound1, 
                  col=federal_st),
              alpha = 0.1) 



