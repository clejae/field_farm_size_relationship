# Create new data

plot_model <- model_surrf_lognorm_7_full
federal_st_var <- 'BB'
df$federal_st
rep_var = 100
rep_var_2 = length(unique(df$new_IDKTYP))
newdata <- data.frame(log_field_size = rep(seq(min(df$log_field_size),
                                             max(df$log_field_size), 
                                             length.out = rep_var), rep_var_2),
                        FormerDDRmember = rep(1, rep_var),
                        avgTRI1000 = rep(rep(max(df$avgTRI1000), rep_var), rep_var_2),
                        propAg1000 = rep(rep(max(df$propAg1000), rep_var), rep_var_2),
                        SQRAvrg = rep(rep(mean(df$SQRAvrg), rep_var), rep_var_2),
                        new_IDKTYP = rep(c(unique(df$new_IDKTYP)), each=rep_var),
                        surrf_mean_log = rep(rep(min(df$surrf_mean_log), rep_var), rep_var_2),
                        federal_st = as.factor(rep(c(unique(df$federal_st)), each=rep_var*rep_var_2)))

newdata[which(newdata$federal_st == 'BV' |
                newdata$federal_st == 'LS'), "FormerDDRmember"] = 0

newdata$FormerDDRmember <- as.factor(newdata$FormerDDRmember)
newdata$new_IDKTYP <- as.factor(newdata$new_IDKTYP)

# Get the predicted values for each posterior sample

range(df$field_size*0.0001)

fs_ha = 50
crop_t = 'CE'
state = 'BV'
grouped_test <- df %>% group_by(federal_st, new_IDKTYP) %>% summarise(med=median(farm_size_ha), 
                                                      q1 = quantile(farm_size_ha, 0.1), 
                                                      q9 = quantile(farm_size_ha, 0.9)) 
fs_tranlated = df[which(df$fieldSizeM >= (fs_ha-0.1)*10000 & df$fieldSizeM <= (fs_ha+0.1)*10000),][1,"log_field_size"]
fs_tranlated = df[which(df$fieldSizeM >= (fs_ha-0.1)*10000),][1,"log_field_size"]
fs_ha = df[which(df$fieldSizeM >= (fs_ha-0.1)*10000),][1,"fieldSizeM"] * 0.0001
posterior_predict(plot_model, newdata = data.frame(log_field_size=fs_tranlated,  #
                                                   FormerDDRmember=as.factor(1),
                                                   avgTRI1000 = max(df$avgTRI1000),
                                                   propAg1000 = max(df$propAg1000),
                                                   SQRAvrg = max(df$SQRAvrg),
                                                   new_IDKTYP = as.factor(crop_t),
                                                   surrf_mean_log = mean(df$surrf_mean_log),
                                                   federal_st=state), ndraws = 200) %>%
  quantile(., probs=c(0.1, 0.5, 0.9)) + fs_ha

mean(df_full$fieldSizeM) * 0.0001
################################################################################
# Field size label translator
# TODO find better breaks
fs_label_list <- c()
range(df_full$log_field_size)
x_ref = seq(0, 50, by=10) %>% log()
x_ref = seq(-6, 4, by=1)

for(i in x_ref) {
  print(i)
  subtractor = 1e-2
  fs_label <- round(df_full[which((df_full$log_field_size >= i - subtractor) & (df_full$log_field_size <= i + subtractor)), "fieldSizeM"]*0.0001, 3)[1]
  fs_label_list <- c(fs_label_list, fs_label)
}

################################################################################
# Surrf label_lis
#range(df_full$surrf_mean)
fs_surff_label_list <- c()

x_ref_surrf = seq(-6, 6, by=1)
#ange(df_full$surrf_mean_log)
for(i in x_ref_surrf) {
  print(i)
  subtractor = 1e-1
  fs_label <- round(df_full[which((df_full$surrf_mean_log >= i - subtractor) & (df_full$surrf_mean_log <= i + subtractor)), "surrf_mean"], 2)[1]
  fs_surff_label_list <- c(fs_surff_label_list, fs_label)
}
fs_surff_label_list

################################################################################

post_pred <- as.data.frame(posterior_predict(plot_model, newdata = newdata, ndraws = 2000))

newdata$pred <- apply(post_pred, 2, median)
newdata$lower_bound1 <- apply(post_pred, 2, quantile, probs = 0.1)
newdata$upper_bound1 <- apply(post_pred, 2, quantile, probs = 0.9)

window_size = 10

for(crop in unique(newdata$new_IDKTYP)){
  for(fedS in unique(newdata$federal_st)) {
    newdata_sub = newdata %>% filter(new_IDKTYP==crop, federal_st==fedS)
    
    newdata_sub$pred <- stats::filter(newdata_sub$pred, rep(1/window_size, window_size), sides = 2)
    
    newdata_sub$lower_bound1<- stats::filter(newdata_sub$lower_bound1, rep(1/window_size, window_size), sides = 2)
    
    newdata_sub$upper_bound1 <- stats::filter(newdata_sub$upper_bound1, rep(1/window_size, window_size), sides = 2)
    
    newdata[which(newdata$new_IDKTYP == crop & newdata$federal_st == fedS), "pred"] <- newdata_sub$pred
    newdata[which(newdata$new_IDKTYP == crop & newdata$federal_st == fedS), "lower_bound1"] <- newdata_sub$lower_bound1
    newdata[which(newdata$new_IDKTYP == crop & newdata$federal_st == fedS), "upper_bound1"] <- newdata_sub$upper_bound1
  }
}



crop_type = 'GR'
DDR = 1
ggplot(data = newdata %>% filter(new_IDKTYP == crop_type, FormerDDRmember==DDR) , aes((log_field_size), pred, col=federal_st)) + #, col=new_IDKTYP
  geom_line(col='red') + 
  geom_ribbon(data = newdata %>% filter(new_IDKTYP == crop_type, FormerDDRmember==DDR) , aes(ymin = (lower_bound1), ymax = (upper_bound1)), alpha=0.2) +
  geom_point(data = df_sample_large %>% filter(new_IDKTYP==crop_type, FormerDDRmember==DDR), aes(x = (log_field_size), y = (farm_size_ha)), alpha = 0.5, shape=21) +
  labs(x = "field size ha", y = "farm size ha") +
  scale_x_continuous(breaks = x_ref, labels = fs_label_list) +
  ggtitle(paste("full model", (crop_type))) + facet_wrap(~federal_st) + 
  ylim(0, 6000) + xlim(-4, 4)




federal_st_select = 'BV'

ggplot(data = newdata %>% filter(federal_st == federal_st_select) , aes((log_field_size), pred, col=new_IDKTYP)) + #, col=new_IDKTYP
  geom_line(col='red') + 
  geom_ribbon(data = newdata %>% filter(federal_st == federal_st_select) , aes(ymin = (lower_bound1), ymax = (upper_bound1)), alpha=0.2) +
  geom_point(data = df %>% filter(federal_st==federal_st_select), aes(x = (log_field_size), y = (farm_size_ha)), alpha = 0.5, shape=21) +
  labs(x = "field size (log, scaled)", y = "farm size ha") +
  ggtitle(paste("full model", (federal_st_select))) + facet_wrap(~new_IDKTYP) + 
  ylim(0, 700) #+ xlim(-4, 4)


jpeg("F:/Max/Field_s_Farm_s/figures/predictions_DDR.jpeg", 
     
     width = 20, height = 15, quality = 100, units = "cm",res= 300,
     
     type = "cairo")

DDR = 1

ggplot() +
  #geom_point(data = df %>% filter(FormerDDRmember == DDR),
  #           aes(x = log_field_size, y = farm_size_ha),
  #           alpha = 0.2, shape = 21) +
  geom_line(data = newdata %>% filter(FormerDDRmember == DDR),
            aes(x = log_field_size, y = pred, col = federal_st)) +
  facet_wrap(~new_IDKTYP) +
  geom_ribbon(data = newdata %>% filter(FormerDDRmember == DDR),
              aes(x = log_field_size, ymin = lower_bound1, ymax = upper_bound1, 
                  col=federal_st),
              alpha = 0.1) +
  labs(x = "field size ha", y = "farm size ha", col='state') +
  ggtitle(paste("full model BB, TH, SA")) +
  ylim(0, 3000) + xlim(-4, 3) +  
  scale_x_continuous(breaks = x_ref, labels = fs_label_list) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_color_manual(values = c("#fc8d59", "#ffffbf", "#91bfdb"))

dev.off()


jpeg("F:/Max/Field_s_Farm_s/figures/predictions_West.jpeg", 
     
     width = 20, height = 15, quality = 100, units = "cm",res= 300,
     
     type = "cairo")
DDR = 0

ggplot() +
  #geom_point(data = df %>% filter(FormerDDRmember == DDR),
  #           aes(x = log_field_size, y = farm_size_ha),
  #           alpha = 0.2, shape = 21) +
  geom_line(data = newdata %>% filter(FormerDDRmember == DDR),
            aes(x = log_field_size, y = pred, col = federal_st)) +
  facet_wrap(~new_IDKTYP) +
  geom_ribbon(data = newdata %>% filter(FormerDDRmember == DDR),
              aes(x = log_field_size, ymin = lower_bound1, ymax = upper_bound1, 
                  col=federal_st),
              alpha = 0.1) +
  labs(x = "field size ha", y = "farm size ha", col='state') +
  ggtitle(paste("full model BB, TH, SA")) +
  ylim(0, 800) + xlim(-4, 3) +  
  scale_x_continuous(breaks = x_ref, labels = fs_label_list) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_color_manual(values = c("#fc8d59", "#91bfdb")) # "#ffffbf", 
dev.off()
######################################################
# experimental plot


vars_ranef <- ranef(plot_model, summary = FALSE) %>% as.data.frame() %>% c() 
model_surrf_lognorm_7$fit@mode
df_ranef <-  as.data.frame(do.call(cbind, vars_ranef)) %>% gather()
#ggplot(df_ranef %>% filter(grepl("sigma",key)), aes(x=value, col=as.factor(key))) + 
#  geom_histogram(bins=50) + facet_wrap(~key)

df_ranef$key <- df_ranef$key %>% gsub(patter='federal_st.new_IDKTYP.', replacement = '')
crop_list <- df$new_IDKTYP %>% unique()
for(crop in crop_list) {
  
  df_ranef[str_detect(df_ranef$key, crop), "CropType"] <- crop
}
state_list <- df$federal_st %>% unique()
for(state in state_list) {
  
  df_ranef[str_detect(df_ranef$key, state), "State"] <- state
}
df_ranef$FormerDDRmember = 'yes'
df_ranef[which(df_ranef$State == 'BV' |
                 df_ranef$State == 'LS'), "FormerDDRmember"] = 'no'

df_ranef$FormerDDRmember <- as.factor(df_ranef$FormerDDRmember)
sorted_keys <- df_ranef %>% group_by(key) %>% summarise(meano=mean(value)) %>% arrange(meano)


df_ranef$key <- factor(df_ranef$key, levels = sorted_keys$key)

df_ranef[which(df_ranef$CropType == 'MA'), "CropType"] <- "Maize"
df_ranef[which(df_ranef$CropType == 'CE'), "CropType"] <- "Cereals"
df_ranef[which(df_ranef$CropType == 'GR'), "CropType"] <- "Grassland"
df_ranef[which(df_ranef$CropType == 'OT'), "CropType"] <- "Other"
df_ranef[which(df_ranef$CropType == 'PO'), "CropType"] <- "Potato"
df_ranef[which(df_ranef$CropType == 'SU'), "CropType"] <- "Sugar beet"
df_ranef[which(df_ranef$CropType == 'OR'), "CropType"] <- "Oil-seed rape"
df_ranef[which(df_ranef$CropType == 'LE'), "CropType"] <- "Legumes"






df_ranef$linetype <- "22"
df_ranef[which(df_ranef$FormerDDRmember==1),"linetype"] <- "solid"
linetype_mapping <- c(df_ranef$linetype)
unique(linetype_mapping)
jpeg("F:/Max/Field_s_Farm_s/figures/varying_slopes.jpeg", 
     
     width = 20, height = 15, quality = 100, units = "cm",res= 300,
     
     type = "cairo")
ggplot()+
  ggridges::geom_density_ridges(data  = df_ranef %>% filter(grepl(".log_field_size",key)), 
                                aes(x      = (value),
                                    y      = key,
                                    height = after_stat(density), 
                                    fill=CropType, 
                                    linetype=FormerDDRmember,
                                    col=FormerDDRmember), scale=3, alpha=0.6) + 
  scale_x_continuous(limits = c(-1,1)) + 
  geom_vline(xintercept = 0, 
             col        = "red") + 
  scale_fill_brewer(palette="Set2") + scale_color_manual(values=c("black", "red")) +
  labs(col='former GDR member', linetype='former GDR member')



dev.off()

jpeg("F:/Max/Field_s_Farm_s/figures/varying_intercepts.jpeg", 
     
     width = 20, height = 15, quality = 100, units = "cm",res= 300,
     
     type = "cairo")
  
ggplot()+
  ggridges::geom_density_ridges(data  = df_ranef %>% filter(grepl("Intercept",key)) %>% filter(!grepl("federal",key)) %>%
                                  filter(!grepl("sigma",key)), 
                                aes(x      = (value),
                                    y      = key,
                                    height = after_stat(density), 
                                    fill=CropType,
                                    linetype=FormerDDRmember,
                                    col=FormerDDRmember), scale=3, alpha=0.6) + 
  scale_x_continuous(limits = c(-2,2)) + geom_vline(xintercept = 0, col= "red")+ 
  scale_fill_brewer(palette="Set2") +
  scale_color_manual(values=c("black", "red")) + 
  labs(col='former GDR member', linetype='former GDR member')

dev.off()





jpeg("F:/Max/Field_s_Farm_s/figures/yarying_intercepts_sigma.jpeg", 
     
     width = 20, height = 15, quality = 100, units = "cm",res= 300,
     
     type = "cairo")

ggplot()+
  ggridges::geom_density_ridges(data  = df_ranef %>% filter(grepl("sigma",key)), 
                                aes(x      = value,
                                    y      = key,
                                    height = after_stat(density), 
                                    fill=CropType,
                                    linetype=FormerDDRmember,
                                    col=FormerDDRmember), scale=3, alpha=0.6) + 
  scale_x_continuous(limits = c(-0.6,0.6)) + geom_vline(xintercept = 0, col= "red")+ 
  scale_fill_brewer(palette="Set2") +
  scale_color_manual(values=c("black", "red")) + 
  labs(col='former GDR member', linetype='former GDR member')
dev.off()


################################################################################
# fixed effects plot / population level effects

summary(model_surrf_lognorm_7)

jpeg("F:/Max/Field_s_Farm_s/figures/fixed_eff_fs.jpeg", 
     
     width = 20, height = 15, quality = 100, units = "cm",res= 300,
     
     type = "cairo")
plotto <- plot(conditional_effects(model_surrf_lognorm_7_full, effects='log_field_size'), points = TRUE) 
plotto$log_field_size + scale_x_continuous(breaks = x_ref, labels = fs_label_list) + 
  xlab('field size in ha') + ylab('farm size in ha')
dev.off()




jpeg("F:/Max/Field_s_Farm_s/figures/fixed_eff_surfs.jpeg", 
     
     width = 20, height = 15, quality = 100, units = "cm",res= 300,
     
     type = "cairo")
plotto <- plot(conditional_effects(model_surrf_lognorm_7_full, effects='surrf_mean_log'), points = TRUE) 
plotto$surrf_mean_log + scale_x_continuous(breaks = x_ref_surrf, labels = fs_surff_label_list) +
  xlab('surrounding field size in ha') + ylab('farm size in ha')
dev.off()

################################################################################
# determine sample size per class

ct_group <- df_sample_large %>% group_by(federal_st, new_IDKTYP) %>% summarise(ct = n())



ggplot() +
  #geom_point(data = df %>% filter(FormerDDRmember == DDR),
  #           aes(x = log_field_size, y = farm_size_ha, col = federal_st),
  #           alpha = 0.2, shape = 21) +
  geom_line(data = newdata %>% filter(FormerDDRmember == DDR),
            aes(x = log_field_size, y = log(pred), col = federal_st)) +
  facet_wrap(~new_IDKTYP) +
  geom_ribbon(data = newdata %>% filter(FormerDDRmember == DDR),
              aes(x = log_field_size, ymin = log(lower_bound1), ymax = log(upper_bound1), 
                  col=federal_st),
              alpha = 0.1) +
  labs(x = "field size (log, scaled)", y = "farm size ha") +
  ggtitle(paste("full model BV, LS"))
 




federal_st_select = 'LS'

ggplot(data = newdata %>% filter(federal_st == federal_st_select) , aes((log_field_size), pred, col=new_IDKTYP)) + #, col=new_IDKTYP
  geom_line(col='red') + 
  geom_ribbon(data = newdata %>% filter(federal_st == federal_st_select) , aes(ymin = (lower_bound1), ymax = (upper_bound1)), alpha=0.2) +
  geom_point(data = df %>% filter(federal_st==federal_st_select), aes(x = (log_field_size), y = (farm_size_ha)), alpha = 0.5, shape=21) +
  labs(x = "field size (log, scaled)", y = "farm size ha") +
  ggtitle(paste("full model", (federal_st_select))) + facet_wrap(~new_IDKTYP) + 
  ylim(0, 700) #+ xlim(-4, 4)


################################################################################
# 
plot_model <- model_surrf_lognorm_7
post_pred <- as.data.frame(posterior_predict(plot_model, ndraws = 2))

post_pred_vector <- apply(post_pred, 2, median) %>% c()

ref_data <- plot_model$data$farm_size_ha
df_qq <- data.frame(ref=ref_data, pred=runif(min=0, max=1000, n=length(ref_data))) #post_pred_vector
df_qq$resid <- df_qq$ref - df_qq$pred
df_qq$fs <- plot_model$data$log_field_size

abs(sum(df_qq$resid))

ggplot(df_qq, aes(ref, pred)) + geom_point() + 
  xlim(0, 6000) + ylim(0, 6000)

gathered <- df_qq %>% gather(., key="key", value="value", -resid, -fs)
ggplot(gathered, aes(fs, value, col=as.factor(key))) + geom_point(alpha=0.6) +
  ylim(0, 6000) + facet_wrap(~as.factor(key))

