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

df_sample_large <- df_sub_1 %>% sample_frac(0.25)
df_sample_large[which(df_sample_large$fieldSizeM < 100),]
write.csv(df_sample_large, '05_Bayesian_model/data/sample_large_025.csv')
df_sample_large <- read.csv('05_Bayesian_model/data/sample_large_025.csv')


model_surrf_lognorm_7_full<- brm(model_formula, 
                            data = df_sample_large, 
                            family = lognormal(link='identity'), 
                            cores=4, chains=4, prior=prior_1)

model_surrf_lognorm_7_full %>% summary()

tab_model(model_surrf_lognorm_7_full)


save(model_surrf_lognorm_7_full, file='05_Bayesian_model/models/model_surrf_lognorm_7_full.rda')
load('05_Bayesian_model/models/model_surrf_lognorm_7_full.rda')

# Extract parameters for the lognormal distribution for the specific predictor values
lognormal_params <- conditional_effects(m, newdata = new_data, terms = "mu_sigma")
pp_check(model_surrf_lognorm_7_full, ndraws = 100) + xlim(0, 1000)
