df <- df_sample_sample

#df <- df[which(df$matched_sa==TRUE),]
#df <- df[-which(df$fieldSizeM <= 2),]
df$farm_size_ha <- df$farm_size

df$ID_KTYP <- as.factor(df$ID_KTYP)
df$new_IDKTYP <- as.factor(df$ID_KTYP)
df$federal_st <- as.factor(df$federal_st)
#df <- sample_frac(df, 0.3)
df$farm_size %>% max()
df$log_surrf_mean

################################################################################
# model_surrf_lognorm_2
# TODO multilevel model: Intercept = mean of normal distribution and sd = sigma? 

prior_1 = c(prior(normal(4.5, 1.1), class = "Intercept"), 
            prior(normal(1, 0.5), class = "b", coef = "log_field_size"), # positive effect assumed
            #prior(normal(1, 0.5), class = "b", coef = "log_surrf_mean"), # positive effect assumed
            prior(normal(0, 0.5), class = "b", coef = "propAg1000"), # effect  in both directiosn possible
            prior(normal(0, 0.5), class ='b', coef='avgTRI1000'), # effect  in both directiosn possible
            prior(normal(0, 0.5), class ='b', coef='SQRAvrg'), # effect  in both directiosn possible
            prior(normal(0, 0.5), class ='b'), # this is the prior for the IDKTYP interaction with fieldsize; each croptype can have different effect
            prior(normal(1, 0.5), lb=0, class = "sd", group='federal_st'),  # mit avg farm size Bayern vs TH begründen
            prior(normal(0, 0.5), lb=0, class = "sd", group='new_IDKTYP'), # mean to to zero because not necessarily different
            prior(normal(1, 0.5), lb=0, class = "Intercept",  dpar="sigma"), 
            prior(exponential(5), lb=0, class = "sd", dpar='sigma'),
            prior(normal(-0.1, 0.2), class = "b",  dpar="sigma")) # larger fields, less uncertainty)

curve(dexp(x, rate = 4), from=0, to=5, col='blue')
curve(dnorm(x, 0, 0.26), from=-1, to=1)

dnorm(seq(1, 10, 0.1), 7, 2) %>% quantile(c(0.1, 0.9))

curve(dlnorm(x, meanlog = 1, sdlog = 2), from=0, to=5000)
# BB: 242; LS: 73; TH: 216; BV: 30; SA: 270 -> Prior für Intercept; 
# log(30) = 3.4 ; log(270) = 5.6 -> normal(4.5, 1) deckt beide ab
# multilevel varying slope of the mean parameter-> "sd", group='federal_st' = normal(1, 1)

model_formula <- bf(farm_size_ha ~ 1 + log_field_size + log_field_size:new_IDKTYP + 
                      log_surrf_mean + 
                      propAg1000 + avgTRI1000 + SQRAvrg + 
                      (1 | federal_st) + (1 | new_IDKTYP),
                    sigma ~ log_field_size + (1 | federal_st))
brms::get_prior(model_formula, data=df, family = lognormal(link='identity'))

model_surrf_lognorm_2 <- brm(model_formula, 
                           data = df, 
                           family = lognormal(link='identity'), 
                           cores=4, chains=4, prior=prior_1, iter=2000) 
model_surrf_lognorm_2 <- add_criterion(model_surrf_lognorm_2, c("loo"))

summary(model_surrf_lognorm_2)
save(model_surrf_lognorm_2, file='05_Bayesian_model/models/model_surrf_lognorm_2.rda')
load('05_Bayesian_model/models/model_surrf_lognorm_2.rda')

plot(model_surrf_lognorm_2)
plot(conditional_effects(model_surrf_lognorm_2), points = TRUE)
model_surrf_lognorm_2$prior
tab_model(model_surrf_lognorm_2)
pp_check(model_surrf_lognorm_2, type="scatter", ndraws =9) + xlim(0, 1000) + ylim(0, 1000)

jpeg("F:/Max/Field_s_Farm_s/figures/PrP_check_exp4.jpeg", 
     
     width = 20, height = 15, quality = 100, units = "cm",res= 300,
     
     type = "cairo")

pp_check(model_surrf_lognorm_2, type="dens_overlay", ndraws =100) + 
  xlim(0, 500) + ylim(0, 0.05) + ggtitle('Prior predictive check')
dev.off()
pp_check(model_surrf_lognorm_2, type="hist", ndraws =9) + xlim(-100, 1000) 


preds_m <- posterior_predict(model_surrf_lognorm_2)
range(preds_m)
loo_compare(model_surrf_lognorm_2, model_surrf_lognorm) 

vars <- ranef(model_surrf_lognorm_2, summary = FALSE) %>% as.data.frame() %>% exp() %>% c() 
hist(vars$state.BV.Intercept, 100)

ll <- model_surrf_lognorm_2 %>%
  log_lik() %>% as_tibble()
ll <- ll %>%
  mutate(sums     = rowSums(.),
         deviance = -2 * sums)

ll %>% ggplot(aes(x = deviance)) + geom_histogram()
################################################################################
# model_surrf_lognorm_3
# TODO how does the model compare to the other full model? (loo)
# TODO how to interpret the interaction terms? Do they make sense
# TODO posterior samples from group level; sigma == logscale
prior_1 = c(prior(normal(4.5, 1.1), class = "Intercept"), 
            prior(normal(1, 0.5), class = "b", coef = "log_field_size"), # positive effect assumed
            prior(normal(1, 0.5), class = "b", coef = "surrf_mean_log"), # positive effect assumed
            prior(normal(0, 0.5), class = "b", coef = "propAg1000"), # effect  in both directiosn possible
            #prior(normal(0, 0.5), class ='b', coef='avgTRI1000'), # effect  in both directiosn possible
            prior(normal(0, 0.5), class ='b', coef='SQRAvrg'), # effect  in both directiosn possible
            prior(normal(0, 0.5), class ='b'), # this is the prior for the IDKTYP interaction with fieldsize; each croptype can have different effect
            prior(normal(1, 0.5), lb=0, class = "sd", group='federal_st'),  # mit avg farm size Bayern vs TH begründen
            prior(normal(0, 0.5), lb=0, class = "sd", group='new_IDKTYP'), # mean to to zero because not necessarily different
            prior(normal(1, 0.5), lb=0, class = "Intercept",  dpar="sigma"), 
            prior(exponential(5), lb=0, class = "sd", dpar='sigma'),
            prior(normal(-0.1, 0.2), class = "b",  dpar="sigma")) # larger fields, less uncertainty)


#df$farm_size_ha <- (df$farm_size * 0.0001) + 1


model_formula <- bf(farm_size_ha ~ 1 + log_field_size + log_field_size:new_IDKTYP + 
                      surrf_mean_log + 
                      propAg1000 + SQRAvrg + 
                      (1 | federal_st) + (1 | new_IDKTYP),
                    sigma ~ log_field_size + (1 | federal_st))

brms::get_prior(model_formula, data=df, family = lognormal(link='identity'))

model_surrf_lognorm_3 <- brm(model_formula, 
                             data = df, 
                             family = lognormal(link='identity'), 
                             cores=4, chains=4 , prior=prior_1)
# TODO
#loo comparison ohne sdelevation und SQR

model_surrf_lognorm_3 <- add_criterion(model_surrf_lognorm_3, c("loo"))
summary(model_surrf_lognorm_3)
save(model_surrf_lognorm_3, file='05_Bayesian_model/models/model_surrf_lognorm_3.rda')
load('05_Bayesian_model/models/model_surrf_lognorm_3.rda')

plot(model_surrf_lognorm_3)
plot(conditional_effects(model_surrf_lognorm_3), points = TRUE)

tab_model(model_surrf_lognorm_3)

pp_check(model_surrf_lognorm_3, ndraws = 11) + xlim(0, 1000)
  coord_cartesian(xlim = c(0, 2000), ylim=c(0, 0.05))


ranef(model_surrf_lognorm_3)
vars <- ranef(model_surrf_lognorm_3, summary = FALSE) %>% as.data.frame() %>% exp() %>% c() 
hist(vars$state.TH.Intercept, 100)
model_surrf_lognorm_3$ranef
posterior_summary(model_surrf_lognorm_3)
################################################################################
# model_surrf_lognorm_4
prior_1 = c(prior(normal(4.5, 1.1), class = "Intercept"), 
            prior(normal(1, 0.5), class = "b", coef = "log_field_size"), # positive effect assumed
            prior(normal(1, 0.5), class = "b", coef = "surrf_mean_log"), # positive effect assumed
            prior(normal(0, 0.5), class = "b", coef = "propAg1000"), # effect  in both directiosn possible
            #prior(normal(0, 0.5), class ='b', coef='avgTRI1000'), # effect  in both directiosn possible
            prior(normal(0, 0.5), class ='b', coef='SQRAvrg'), # effect  in both directiosn possible
            prior(normal(0, 0.5), class ='b'), # this is the prior for the IDKTYP interaction with fieldsize; each croptype can have different effect
            prior(normal(1, 0.5), lb=0, class = "sd", group='federal_st'),  # mit avg farm size Bayern vs TH begründen
            prior(normal(0, 0.5), lb=0, class = "sd", group='new_IDKTYP'), # mean to to zero because not necessarily different
            prior(normal(1, 0.5), lb=0, class = "Intercept",  dpar="sigma"), 
            prior(exponential(5), lb=0, class = "sd", dpar='sigma'),
            prior(normal(-0.1, 0.2), class = "b",  dpar="sigma")) # larger fields, less uncertainty)


model_formula <- bf(farm_size_ha ~ 1 + log_field_size:new_IDKTYP + 
                      log_field_size + surrf_mean_log*propAg1000 + 
                      SQRAvrg + 
                      (1 | federal_st) + (1 | new_IDKTYP),
                    sigma ~ log_field_size + (1 | federal_st))

brms::get_prior(model_formula, data=df, family = lognormal(link='identity'))

model_surrf_lognorm_4 <- brm(model_formula, 
                             data = df, 
                             family = lognormal(link='identity'), 
                             cores=4, chains=4 , prior=prior_1)
# TODO
#loo comparison ohne sdelevation und SQR

model_surrf_lognorm_4 <- add_criterion(model_surrf_lognorm_4, c("loo"))
summary(model_surrf_lognorm_4)
save(model_surrf_lognorm_4, file='05_Bayesian_model/models/model_surrf_lognorm_4.rda')
load('05_Bayesian_model/models/model_surrf_lognorm_4.rda')

plot(model_surrf_lognorm_4)
plot(conditional_effects(model_surrf_lognorm_4), points = TRUE)

tab_model(model_surrf_lognorm_4)
jpeg("F:/Max/Field_s_Farm_s/figures/PP_check_model4.jpeg", 
     
     width = 20, height = 15, quality = 100, units = "cm",res= 300,
     
     type = "cairo")

pp_check(model_surrf_lognorm_4, ndraws = 11) + xlim(0, 1000)
dev.off()

coord_cartesian(xlim = c(0, 2000), ylim=c(0, 0.05))


ranef(model_surrf_lognorm_4)
vars <- ranef(model_surrf_lognorm_4, summary = FALSE) %>% as.data.frame() %>% exp() %>% c() 
hist(vars$state.TH.Intercept, 100)
model_surrf_lognorm_4$ranef
posterior_summary(model_surrf_lognorm_4)

################################################################################
# model_surrf_lognorm_5

prior_1 = c(prior(normal(4.5, 1.1), class = "Intercept"), 
            prior(normal(1, 0.5), class = "b", coef = "log_field_size"), # positive effect assumed
            prior(normal(1, 0.5), class = "b", coef = "surrf_mean_log"), # positive effect assumed
            prior(normal(0, 0.5), class = "b", coef = "propAg1000"), # effect  in both directiosn possible
            #prior(normal(0, 0.5), class ='b', coef='avgTRI1000'), # effect  in both directiosn possible
            prior(normal(0, 0.5), class ='b', coef='SQRAvrg'), # effect  in both directiosn possible
            prior(normal(0, 0.5), class ='b'), # this is the prior for the IDKTYP interaction with fieldsize; each croptype can have different effect
            prior(normal(1, 0.5), lb=0, class = "sd", group='federal_st'),  # mit avg farm size Bayern vs TH begründen
            prior(normal(0, 0.5), lb=0, class = "sd", group='new_IDKTYP'), # mean to to zero because not necessarily different
            prior(normal(1, 0.5), lb=0, class = "Intercept",  dpar="sigma"), 
            prior(exponential(5), lb=0, class = "sd", dpar='sigma'),
            prior(normal(-0.1, 0.2), class = "b",  dpar="sigma")) # larger fields, less uncertainty)


model_formula <- bf(farm_size_ha ~ 1 + log_field_size:new_IDKTYP + 
                      log_field_size + surrf_mean_log*propAg1000 + 
                      SQRAvrg + avgTRI0 +
                      (1 | federal_st) + (1 | new_IDKTYP),
                    sigma ~ log_field_size + (1 | federal_st))

brms::get_prior(model_formula, data=df, family = lognormal(link='identity'))

                 
model_surrf_lognorm_5 <- brm(model_formula, 
                             data = df, 
                             family = lognormal(link='identity'), 
                             cores=4, chains=4, prior=prior_1)
prior_summary(model_surrf_lognorm_5)
pp_check(model_surrf_lognorm_5) + xlim(0, 1000)

model_surrf_lognorm_5 <- add_criterion(model_surrf_lognorm_5, c("loo"))
summary(model_surrf_lognorm_5)
save(model_surrf_lognorm_5, file='05_Bayesian_model/models/model_surrf_lognorm_5.rda')
load('05_Bayesian_model/models/model_surrf_lognorm_5.rda')

plot(model_surrf_lognorm_5)
plot(conditional_effects(model_surrf_lognorm_5), points = TRUE)

tab_model(model_surrf_lognorm_5)
pp_check(model_surrf_lognorm_5) + xlim(0, 1000)

ranef(model_surrf_lognorm_5)
vars_ranef <- ranef(model_surrf_lognorm_5, summary = FALSE) %>% as.data.frame() %>% exp() %>% c() 

df_ranef <-  as.data.frame(do.call(cbind, vars_ranef)) %>% gather()
ggplot(df_ranef %>% filter(grepl("sigma",key)), aes(x=value, col=as.factor(key))) + 
  geom_histogram(bins=50) + facet_wrap(~key)


ll <- model_surrf_lognorm_5 %>%
  log_lik() %>% as_tibble()
ll <- ll %>%
  mutate(sums     = rowSums(.),
         deviance = -2 * sums)

ll %>% ggplot(aes(x = deviance)) + geom_histogram()

################################################################################
# model_surrf_lognorm_6

prior_1 = c(prior(normal(4.5, 1.1), class = "Intercept"), 
            prior(normal(0, 0.5), class = "b", coef = "log_field_size"), # positive effect assumed
            prior(normal(0, 0.5), class = "b", coef = "surrf_mean_log"), # positive effect assumed
            prior(normal(0, 0.5), class = "b", coef = "propAg1000"), # effect  in both directiosn possible
            #prior(normal(0, 0.5), class ='b', coef='avgTRI1000'), # effect  in both directiosn possible
            prior(normal(0, 0.5), class ='b', coef='SQRAvrg'), # effect  in both directiosn possible
            prior(normal(0, 0.5), class ='b'), # this is the prior for the IDKTYP interaction with fieldsize; each croptype can have different effect
            prior(normal(1, 0.5), lb=0, class = "sd"),  # mit avg farm size Bayern vs TH begründen
            #prior(normal(0, 0.5), lb=0, class = "sd", group='new_IDKTYP'), # mean to to zero because not necessarily different
            prior(normal(1, 0.5), lb=0, class = "Intercept",  dpar="sigma"), 
            prior(exponential(5), lb=0, class = "sd", dpar='sigma'),
            prior(normal(-0.1, 0.2), class = "b",  dpar="sigma")) # larger fields, less uncertainty)


model_formula <- bf(farm_size_ha ~ 1 + log_field_size:new_IDKTYP:federal_st + 
                      log_field_size + surrf_mean_log*propAg1000 + 
                      SQRAvrg + 
                      (log_field_size | federal_st:new_IDKTYP),
                    sigma ~ log_field_size + (1 | federal_st))


model_surrf_lognorm_test <- brm(model_formula, 
                             data = df, 
                             family = lognormal(link='identity'), 
                             cores=4, chains=4, prior=prior_1)

model_surrf_lognorm_test <- add_criterion(model_surrf_lognorm_test, c("loo"))
summary(model_surrf_lognorm_test)
save(model_surrf_lognorm_6, file='05_Bayesian_model/models/model_surrf_lognorm_6.rda')
load('05_Bayesian_model/models/model_surrf_lognorm_6.rda')

plot(model_surrf_lognorm_6)
plot(conditional_effects(model_surrf_lognorm_6), points = TRUE) 

tab_model(model_surrf_lognorm_6)
pp_check(model_surrf_lognorm_6) + xlim(0, 1000)

ranef(model_surrf_lognorm_6)

vars_ranef <- ranef(model_surrf_lognorm_6, summary = FALSE) %>% as.data.frame() %>% exp() %>% c() 

df_ranef <-  as.data.frame(do.call(cbind, vars_ranef)) %>% gather()
ggplot(df_ranef %>% filter(grepl("sigma",key)), aes(x=value, col=as.factor(key))) + 
  geom_histogram(bins=50) + facet_wrap(~key)


ggplot(df, aes(sdElevatio, log(farm_size_ha), col=state)) + geom_point() + geom_smooth(method='lm')

################################################################################
# model_surrf_lognorm_7

prior_1 = c(prior(normal(4.5, 1.1), class = "Intercept"), 
            prior(normal(0, 0.5), class = "b", coef = "log_field_size"), # positive effect assumed
            prior(normal(0, 0.5), class = "b", coef = "surrf_mean_log"), # positive effect assumed
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
                      log_field_size + surrf_mean_log*propAg1000 + 
                      SQRAvrg + 
                      (0 + log_field_size | federal_st:new_IDKTYP) + 
                      (1 | federal_st:new_IDKTYP),
                    sigma ~ log_field_size + (1 | federal_st:new_IDKTYP))


model_surrf_lognorm_7<- brm(model_formula, 
                                data = df, 
                                family = lognormal(link='identity'), 
                                cores=4, chains=4, prior=prior_1, sample_prior = 'only')


jpeg("F:/Max/Field_s_Farm_s/figures/PP_check_model7.jpeg", 
     
     width = 20, height = 15, quality = 100, units = "cm",res= 300,
     
     type = "cairo")
pp_check(model_surrf_lognorm_7, ndraws = 40) + xlim(0, 750)
dev.off()
model_surrf_lognorm_7 <- add_criterion(model_surrf_lognorm_7, c("loo"))

save(model_surrf_lognorm_7, file='05_Bayesian_model/models/model_surrf_lognorm_7.rda')
load('05_Bayesian_model/models/model_surrf_lognorm_7.rda')
prior_summary(model_surrf_lognorm_7)

plot(model_surrf_lognorm_7)

summary(model_surrf_lognorm_7)
tab_model(model_surrf_lognorm_7)
pp_check(model_surrf_lognorm_7) + xlim(0, 500)


vars_fixef <- fixef(model_surrf_lognorm_7)["log_field_size"] %>% mean()
vars_ranef <- ranef(model_surrf_lognorm_7, summary = FALSE) %>% as.data.frame()  %>% c() #%>% exp()

df_ranef <-  as.data.frame(do.call(cbind, vars_ranef)) %>% gather()
ggplot(df_ranef %>% filter(grepl("SU.log_field_size",key)), aes(x=value, col=as.factor(key))) + 
  geom_histogram(bins=50) + facet_wrap(~key)
unique(df_ranef$key) 

################################################################################
# model_surrf_lognorm_8

prior_1 = c(prior(normal(4.5, 1.1), class = "Intercept"), 
            prior(normal(0, 0.5), class = "b", coef = "log_field_size"), # positive effect assumed
            prior(normal(0, 0.5), class = "b", coef = "surrf_mean_log"), # positive effect assumed
            prior(normal(0, 0.5), class = "b", coef = "propAg1000"), # effect  in both directiosn possible
            #prior(normal(0, 0.5), class ='b', coef='avgTRI1000'), # effect  in both directiosn possible
            prior(normal(0, 0.5), class ='b', coef='SQRAvrg'), # effect  in both directiosn possible
            prior(normal(0, 0.5), class ='b'), # this is the prior for the IDKTYP interaction with fieldsize; each croptype can have different effect
            prior(normal(1, 0.5), lb=0, class = "sd"),  # mit avg farm size Bayern vs TH begründen
            #prior(normal(0, 0.5), lb=0, class = "sd", group='new_IDKTYP'), # mean to to zero because not necessarily different
            prior(normal(1, 0.5), lb=0, class = "Intercept",  dpar="sigma"), 
            prior(exponential(5), lb=0, class = "sd", dpar='sigma'),
            prior(normal(-0.1, 0.2), class = "b",  dpar="sigma")) # larger fields, less uncertainty)
#log_field_size:new_IDKTYP:federal_st + 

model_formula <- bf(farm_size_ha ~ 1 + 
                      log_field_size + surrf_mean_log*propAg1000 + 
                      SQRAvrg + 
                      (0 + log_field_size | federal_st:new_IDKTYP) + 
                      (1 | federal_st:new_IDKTYP),
                    sigma ~ log_field_size + (log_field_size | federal_st:new_IDKTYP))


model_surrf_lognorm_8 <- brm(model_formula, 
                             data = df, 
                             family = lognormal(link='identity'), 
                             cores=4, chains=4, prior=prior_1)

model_surrf_lognorm_8 <- add_criterion(model_surrf_lognorm_8, c("loo"))
summary(model_surrf_lognorm_8)
save(model_surrf_lognorm_8, file='05_Bayesian_model/models/model_surrf_lognorm_8.rda')
load('05_Bayesian_model/models/model_surrf_lognorm_8.rda')
prior_summary(model_surrf_lognorm_8)

plot(model_surrf_lognorm_8)
plot(conditional_effects(model_surrf_lognorm_8), points = TRUE) 

tab_model(model_surrf_lognorm_8)
pp_check(model_surrf_lognorm_8) + xlim(0, 1000)


vars_fixef <- fixef(model_surrf_lognorm_8)["log_field_size"] %>% mean()
vars_ranef <- ranef(model_surrf_lognorm_8, summary = FALSE) %>% as.data.frame()  %>% c() #%>% exp()

df_ranef <-  as.data.frame(do.call(cbind, vars_ranef)) %>% gather()
ggplot(df_ranef %>% filter(grepl("SU.log_field_size",key)), aes(x=value, col=as.factor(key))) + 
  geom_histogram(bins=50) + facet_wrap(~key)
unique(df_ranef$key) 


################################################################################
# model_surrf_lognorm_9

prior_1 = c(prior(normal(4.5, 1.1), class = "Intercept"), 
            #prior(normal(0, 0.5), class = "b", coef = "log_field_size"), # positive effect assumed
            prior(normal(0, 0.5), class = "b", coef = "surrf_mean_log"), # positive effect assumed
            prior(normal(0, 0.5), class = "b", coef = "propAg1000"), # effect  in both directiosn possible
            #prior(normal(0, 0.5), class ='b', coef='avgTRI1000'), # effect  in both directiosn possible
            prior(normal(0, 0.5), class ='b', coef='SQRAvrg'), # effect  in both directiosn possible
            prior(normal(0, 0.5), class ='b'), # this is the prior for the IDKTYP interaction with fieldsize; each croptype can have different effect
            prior(normal(1, 0.5), lb=0, class = "sd"),  # mit avg farm size Bayern vs TH begründen
            #prior(normal(0, 0.5), lb=0, class = "sd", group='new_IDKTYP'), # mean to to zero because not necessarily different
            prior(normal(1, 0.5), lb=0, class = "Intercept",  dpar="sigma"), 
            prior(exponential(5), lb=0, class = "sd", dpar='sigma'),
            prior(normal(-0.1, 0.2), class = "b",  dpar="sigma")) # larger fields, less uncertainty)
#log_field_size:new_IDKTYP:federal_st + 

model_formula <- bf(farm_size_ha ~ 1 + 
                      s(log_field_size) + surrf_mean_log*propAg1000 + 
                      SQRAvrg + 
                      (0 + log_field_size | federal_st:new_IDKTYP) + 
                      (1 | federal_st:new_IDKTYP),
                    sigma ~ s(log_field_size) + (log_field_size | federal_st:new_IDKTYP))


model_surrf_lognorm_9 <- brm(model_formula, 
                             data = df, 
                             family = lognormal(link='identity'), 
                             cores=4, chains=4, prior=prior_1)

model_surrf_lognorm_9 <- add_criterion(model_surrf_lognorm_9, c("loo"))
summary(model_surrf_lognorm_9)
save(model_surrf_lognorm_9, file='05_Bayesian_model/models/model_surrf_lognorm_9.rda')
load('05_Bayesian_model/models/model_surrf_lognorm_9.rda')
prior_summary(model_surrf_lognorm_9)

plot(model_surrf_lognorm_9)
plot(conditional_effects(model_surrf_lognorm_9), points = TRUE) 

tab_model(model_surrf_lognorm_9)
pp_check(model_surrf_lognorm_9) + xlim(0, 1000)


vars_fixef <- fixef(model_surrf_lognorm_9)["log_field_size"] %>% mean()
vars_ranef <- ranef(model_surrf_lognorm_9, summary = FALSE) %>% as.data.frame()  %>% c() #%>% exp()

df_ranef <-  as.data.frame(do.call(cbind, vars_ranef)) %>% gather()
ggplot(df_ranef %>% filter(grepl("SU.log_field_size",key)), aes(x=value, col=as.factor(key))) + 
  geom_histogram(bins=50) + facet_wrap(~key)
unique(df_ranef$key) 


################################################################################
# plotting
m <- df %>% sample_frac(1) %>% 
glm(log_farm_size ~ log_field_size + SQRAvrg + avgTRI1000 +  federal_st + new_IDKTYP, family=gaussian(link='identity'), data=.)
summary(m)
df$log_field_size
ggplot(df, aes(avgTRI1000,log_farm_size, col= federal_st)) + geom_point() + geom_smooth(method = 'lm')
ggplot(df, aes(SQRAvrg,x=federal_st )) + geom_boxplot() + geom_smooth(method = 'lm')

m <- df %>% sample_frac(1) %>% 
  lm(log_farm_size ~  FormerDDRmember + log_field_size*surrf_mean_log+proportAgr , data=.) 
m_1 <- df %>% sample_frac(1) %>% 
  lm(log_farm_size ~ FormerDDRmember * proportAgr*log_field_size*surrf_mean_log , data=.) 
m_2 <- df %>% sample_frac(1) %>% 
  lm(log_farm_size ~ log_field_size  + FormerDDRmember + proportAgr+ log_field_size*surrf_mean_log , data=.) 
m_3 <- df %>% sample_frac(1) %>% 
  lm(log_farm_size ~ log_field_size  + FormerDDRmember + proportAgr+ log_field_size +surrf_mean_log , data=.) 
m_4 <- df %>% sample_frac(1) %>% 
  lm(log_farm_size ~ log_field_size  + FormerDDRmember + log_field_size*surrf_mean_log , data=.) 
m_5 <- df %>% sample_frac(1) %>% 
  lm(log_farm_size ~ log_field_size + FormerDDRmember , data=.) 
m_6 <- df %>% sample_frac(1) %>% 
  lm(log_farm_size ~ log_field_size + FormerDDRmember , data=.) 
m_7 <- df %>% sample_frac(1) %>% 
  lm(log_farm_size ~ surrf_mean_log +FormerDDRmember, data=.) 
summary(m)
summary(m_7)

AIC(m, m_1, m_2, m_3, m_4, m_5, m_6, m_7) %>% arrange(AIC)
plotss <- plot_model(model_full, type = "int", term=c("log_field_size", "surrf_mean_log"), mdrt.values = "minmax" )
plotss
rm(plotss)
summary(m)
AIC(m)
new_data <- data.frame(log_field_size = seq(min(df$log_field_size), 
                                            max(df$log_field_size), 
                                            length.out = 100), 
                       proportAgr = rep(min(df$proportAgr), 100),
                       surrf_mean_log = rep(max(df$surrf_mean_log), 100),
                       FormerDDRmember = as.factor(rep(0, 100)))

new_data$pred <- predict(m, newdata = new_data)

ggplot(new_data, aes(log_field_size, pred)) + geom_line() + 
  geom_point(data=df %>% sample_frac(0.2) , aes(log_field_size, log_farm_size))



  