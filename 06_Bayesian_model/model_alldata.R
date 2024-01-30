################################################################################
# model_surrf_lognorm_7

prior_1 = c(prior(normal(4.5, 1.1), class = "Intercept"), 
            prior(normal(0, 0.5), class = "b", coef = "log_field_size"), # positive effect assumed
            prior(normal(0, 0.5), class = "b", coef = "log_surrf_mean"), # positive effect assumed
            prior(normal(0, 0.5), class = "b", coef = "propAg1000"), # effect  in both directiosn possible
            #prior(normal(0, 0.5), class ='b', coef='avgTRI1000'), # effect  in both directiosn possible
            prior(normal(0, 0.5), class ='b', coef='SQRAvrg'), # effect  in both directiosn possible
            prior(normal(0, 0.5), class ='b'), # this is the prior for the IDKTYP interaction with fieldsize; each croptype can have different effect
            prior(normal(0.5, 0.5), coef='Intercept', group="federal_st:new_IDKTYP", class = "sd"),  # mit avg farm size Bayern vs TH begründen
            prior(normal(0.5, 0.5),  coef='log_field_size', group="federal_st:new_IDKTYP", class = "sd"),  
            prior(normal(0.5, 0.5),  coef='Intercept', group="federal_st:new_IDKTYP", class = "sd", dpar='sigma'),  
            #prior(normal(0, 0.5), lb=0, class = "sd", group='new_IDKTYP'), # mean to to zero because not necessarily different
            prior(normal(1, 0.5), lb=0, class = "Intercept",  dpar="sigma"), 
            prior(exponential(5), lb=0, class = "sd", dpar='sigma'),
            prior(normal(-0.1, 0.2), class = "b",  dpar="sigma")) # larger fields, less uncertainty)
#log_field_size:new_IDKTYP:federal_st + 

model_formula <- bf(farm_size_ha ~ 1 + 
                      log_field_size + log_surrf_mean*propAg1000 + 
                      SQRAvrg + 
                      (0 + log_field_size | federal_st:new_IDKTYP) + 
                      (1 | federal_st:new_IDKTYP),
                    sigma ~ log_field_size + (1 | federal_st:new_IDKTYP))

df_sample_large <- df_sample
df_sample_large$farm_size_ha <- df_sample_large$farm_size
df_sample_large$ID_KTYP <- as.factor(df_sample_large$ID_KTYP)
df_sample_large$new_IDKTYP <- as.factor(df_sample_large$new_ID_KTYP)
df_sample_large$federal_st <- as.factor(df_sample_large$federal_st)

unique(df_sample_large$new_IDKTYP )
#df_sample_large[which(df_sample_large$fieldSizeM < 100),]
#write.csv(df_sample_large, '05_Bayesian_model/data/sample_large_025.csv')
#df_sample_large <- read.csv('06_Bayesian_model/data/sample_large_025.csv')


model_surrf_lognorm_7_full<- brm(model_formula, 
                            data = df_sample_large, 
                            family = lognormal(link='identity'), 
                            cores=4, chains=4, prior=prior_1)

model_surrf_lognorm_7_full %>% summary()

tab_model(model_surrf_lognorm_7_full)
model_surrf_lognorm_7_full <- add_criterion(model_surrf_lognorm_7_full, c("loo"))


save(model_surrf_lognorm_7_full, file='06_Bayesian_model/models/model_surrf_lognorm_7_full.rda')
load('06_Bayesian_model/models/model_surrf_lognorm_7_full.rda')

# Extract parameters for the lognormal distribution for the specific predictor values
lognormal_params <- conditional_effects(model_surrf_lognorm_7_full, newdata = new_data, terms = "mu_sigma")
pp_check(model_surrf_lognorm_7_full, ndraws = 100) + xlim(0, 1000)
plot(model_surrf_lognorm_7_full)


################################################################################
# model_surrf_lognorm_8

prior_1 = c(prior(normal(4.5, 1.1), class = "Intercept"), 
            prior(normal(0, 0.5), class = "b", coef = "log_field_size"), # positive effect assumed
            prior(normal(0, 0.5), class = "b", coef = "log_surrf_mean"), # positive effect assumed
            prior(normal(0, 0.5), class = "b", coef = "propAg1000"), # effect  in both directiosn possible
            #prior(normal(0, 0.5), class ='b', coef='avgTRI1000'), # effect  in both directiosn possible
            prior(normal(0, 0.5), class ='b', coef='SQRAvrg'), # effect  in both directiosn possible
            prior(normal(0, 0.5), class ='b'), # this is the prior for the IDKTYP interaction with fieldsize; each croptype can have different effect
            prior(normal(0.5, 0.5), coef='Intercept', group="federal_st:new_IDKTYP", class = "sd"),  # mit avg farm size Bayern vs TH begründen
            prior(normal(0.5, 0.5),  coef='log_field_size', group="federal_st:new_IDKTYP", class = "sd"),  
            prior(normal(0.5, 0.5),  coef='Intercept', group="federal_st:new_IDKTYP", class = "sd", dpar='sigma'),  
            #prior(normal(0, 0.5), lb=0, class = "sd", group='new_IDKTYP'), # mean to to zero because not necessarily different
            prior(normal(1, 0.5), lb=0, class = "Intercept",  dpar="sigma"), 
            prior(exponential(5), lb=0, class = "sd", dpar='sigma'),
            prior(normal(-0.1, 0.2), class = "b",  dpar="sigma")) # larger fields, less uncertainty)
#log_field_size:new_IDKTYP:federal_st + 

model_formula <- bf(farm_size_ha ~ 1 + 
                      log_field_size + log_surrf_mean*propAg1000 + 
                      SQRAvrg + 
                      (0 + log_field_size | federal_st:new_IDKTYP) + 
                      (1 | federal_st:new_IDKTYP),
                    sigma ~ log_field_size + (1 | federal_st:new_IDKTYP))



unique(df_sample_large$new_IDKTYP )
#df_sample_large[which(df_sample_large$fieldSizeM < 100),]
#write.csv(df_sample_large, '05_Bayesian_model/data/sample_large_025.csv')
#df_sample_large <- read.csv('06_Bayesian_model/data/sample_large_025.csv')


model_surrf_lognorm_8_full<- brm(model_formula, 
                                 data = df_sample_large, 
                                 family = lognormal(link='identity'), 
                                 cores=4, chains=4, prior=prior_1)

model_surrf_lognorm_8_full %>% summary()




save(model_surrf_lognorm_8_full, file='06_Bayesian_model/models/model_surrf_lognorm_8_full.rda')
load('06_Bayesian_model/models/model_surrf_lognorm_8_full.rda')

# Extract parameters for the lognormal distribution for the specific predictor values
lognormal_params <- conditional_effects(model_surrf_lognorm_7_full, newdata = new_data, terms = "mu_sigma")
pp_check(model_surrf_lognorm_7_full, ndraws = 100) + xlim(0, 1000)
plot(model_surrf_lognorm_7_full)


nrow(model_surrf_lognorm_7_full$data)
nrow(model_surrf_lognorm_8_full$data) == nrow(df_sample_large)

################################################################################
# 
library(gamlss)
library(dplyr)
library(ggplot2)
library(ltm)
df_sample_large_omitted <- na.omit(df_sample_large)

model_gamlss <- gamlss(formula = farm_size_ha ~ 1 + 
         log_field_size + surrf_mean_log*propAg1000 + new_IDKTYP:federal_st +
           log_field_size:new_IDKTYP:federal_st, 
         data=df, family = LOGNO2())

summary(model_gamlss)

# if we don't control for state and croptype positive effect surrf_mean_log*propAg1000 
glm(formula = log(farm_size_ha) ~ 1 + 
      log_field_size + surrf_mean_log*propAg1000 , 
    data=df, family = gaussian()) %>% summary()
# effect of surrf_mean_log*propAg1000 dissappears
glm(formula = log(farm_size_ha) ~ 1 + 
      log_field_size + surrf_mean_log*propAg1000 + new_IDKTYP+federal_st, 
    data=df, family = gaussian()) %>% summary()
#
glm(formula = log(farm_size_ha) ~ 1 + log_field_size +
      + surrf_mean_log:propAg1000:federal_st+ new_IDKTYP+federal_st, 
    data=df_sample_large, family = gaussian()) %>% summary()
df$FormerDDRmember
ggplot(df_sample_large, aes(propAg1000 * surrf_mean_log, log(farm_size_ha))) + geom_point() + 
  facet_wrap(~federal_st)+ xlim(-5, 10) + geom_smooth(method='lm') 


################################################################################
# why does the coefficient of propAg1000 flip when I include surrf_mean_log?
df <- df_sample_large
model_msk <- glm(formula = (farm_size_ha) ~ log_surrf_mean +
       propAg1000, 
    data=df, family = gaussian(link='log'))
summary(model_msk)
plot(model_msk)

ggplot(df, aes(propAg1000 , surrf_mean_log)) + geom_point()  + 
  facet_wrap(~federal_st) + geom_smooth(method='gam') 



# Create new data

plot_model <- model_msk
federal_st_var <- 'BB'
df$federal_st
rep_var = 100
rep_var_2 = length(unique(df$new_IDKTYP))
newdata <- data.frame(log_field_size = rep(mean(df$log_field_size)),
                      FormerDDRmember = rep(1, rep_var),
                      avgTRI1000 = rep(rep(max(df$avgTRI1000), rep_var), rep_var_2),
                      propAg1000 = rep(seq(min(df$propAg1000),
                                           max(df$propAg1000), 
                                           length.out = rep_var), rep_var_2),
                      SQRAvrg = rep(rep(mean(df$SQRAvrg), rep_var), rep_var_2),
                      new_IDKTYP = rep(c(unique(df$new_IDKTYP)), each=rep_var),
                      log_surrf_mean = rep(rep(mean(df$log_surrf_mean), rep_var), rep_var_2),
                      federal_st = as.factor(rep(c(unique(df$federal_st)), each=rep_var*rep_var_2)))

newdata$pred <- predict(model_msk, newdata = newdata)

ggplot(df, aes(propAg1000, log(farm_size_ha))) + geom_point() + geom_smooth(method='lm') + 
  geom_line(data=newdata, aes(propAg1000, pred), col='red') 



################################################################################
# SQR ? -> SQR and crop type are highly correlated -> we should not include both
# in regression
aov(SQRAvrg ~ new_IDKTYP, data=df) %>% summary()

model_msk <- glm(formula = log(farm_size_ha) ~ log_field_size + log_surrf_mean + SQRAvrg*federal_st*new_IDKTYP , 
                 data=df, family = gaussian())

summary(model_msk)
#plot(model_msk)


ggplot(df, aes(y=SQRAvrg , col=new_IDKTYP)) + geom_boxplot()  #+ 
  facet_wrap(~federal_st) 
# Create new data

plot_model <- model_msk
federal_st_var <- 'BB'
df$federal_st
rep_var = 100
rep_var_2 = length(unique(df$new_IDKTYP))
newdata <- data.frame(log_field_size = rep(mean(df$log_field_size)),
                      FormerDDRmember = rep(1, rep_var),
                      avgTRI1000 = rep(rep(max(df$avgTRI1000), rep_var), rep_var_2),
                      propAg1000 = rep(mean(df$propAg1000)),
                      SQRAvrg =  rep(seq(min(df$SQRAvrg),
                                         max(df$SQRAvrg), 
                                         length.out = rep_var), rep_var_2),
                      new_IDKTYP = rep(c(unique(df$new_IDKTYP)), each=rep_var),
                      surrf_mean_log = rep(rep(mean(df$surrf_mean_log), rep_var), rep_var_2),
                      federal_st = as.factor(rep(c(unique(df$federal_st)), each=rep_var*rep_var_2)))

newdata$pred <- predict(model_msk, newdata = newdata)

ggplot(df, aes(SQRAvrg, log(farm_size_ha))) + geom_point() + geom_smooth(method='gam') + 
  facet_wrap(~federal_st) + 
  geom_line(data=newdata, aes(SQRAvrg, pred, col=new_IDKTYP))

################################################################################
# elevation TRI etc.; too much correlation between TRI and surff mean 
ggplot(df, aes(avgTRI1000, log(farm_size_ha))) + geom_point() + 
  #facet_wrap(~federal_st) + 
  geom_smooth(method='lm') 

df$avgTRI1000
model_msk <- glm(formula = log(farm_size_ha) ~ log_field_size +  surrf_mean_log:
                   avgTRI1000, 
                 data=df, family = gaussian())
summary(model_msk)


ggplot(df, aes(avgTRI1000, (surrf_mean_log))) + geom_point() + 
  #facet_wrap(~federal_st) + 
  geom_smooth(method='lm') 

cor(df$surrf_mean_log, df$avgTRI1000)


################################################################################
# surff sd ? too much correlation with surrf_mean ?
ggplot(df, aes((log(surrf_std+5)), log(farm_size_ha))) + geom_point() + 
  #facet_wrap(~federal_st) + 
  geom_smooth(method='lm') 

df$surrf_std
model_msk <- glm(formula = (farm_size_ha) ~ log_field_size*federal_st:new_IDKTYP +  surrf_mean_log , 
                 data=df_sample_large, family = gaussian(link='log'))
summary(model_msk)

cor(df$surrf_std, df$surrf_mean_log)



plot_model <- model_msk
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
                      surrf_std = rep(rep(mean(df$surrf_mean_log), rep_var), rep_var_2),
                      
                      federal_st = as.factor(rep(c(unique(df$federal_st)), each=rep_var*rep_var_2)))

newdata$pred <- predict(model_msk, newdata = newdata)

ggplot(df, aes(log_field_size, log(farm_size_ha))) + geom_point() + geom_smooth(method='gam') + 
  facet_wrap(~federal_st) + 
  geom_line(data=newdata, aes(log_field_size, pred, col=new_IDKTYP))

################################################################################
# include number of fields as well? 
df$surrf_mean_log
ggplot(df, aes((log(surrf_no_f+2)),log(farm_size_ha))) + geom_point() + 
  #facet_wrap(~federal_st) + 
  geom_smooth(method='gam') 


model_msk <- glm(formula = (farm_size_ha) ~ log_field_size*surrf_mean_log*log(surrf_no_f+2) + federal_st+new_IDKTYP, 
                 data=df, family = gaussian(link='log'))
model_msk_2 <- glm(formula = (farm_size_ha) ~ log_field_size*surrf_mean_log + federal_st+new_IDKTYP, 
                 data=df, family = gaussian(link='log'))
summary(model_msk)
model_msk$aic - model_msk_2$aic


################################################################################
# VIF

model_msk <- glm(formula = log(farm_size_ha) ~ log_field_size  + surrf_mean_log*federal_st + propAg1000 +  new_IDKTYP, 
                 data=df, family = gaussian())
summary(model_msk)
vif(model_msk)

subsets <- regsubsets(x=log(df$farm_size_ha) ~ log_field_size  + surrf_mean_log*federal_st + propAg1000 +  new_IDKTYP, y=log(df$farm_size_ha), data=df)
summary.regsubsets(subsets)
