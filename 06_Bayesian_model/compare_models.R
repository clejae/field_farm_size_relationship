
########################################
# compare models
model_mvlvl_intercept_elev$criteria

loo_compare(model_simple, model_simple_homsk, model_simple_elev,
            model_simple_elev_nfk, model_mvlvl_intercept, model_mlvlv,
            model_mvlvl_intercept_elev, model_simple_slope, model_simple_elev_intact,
            criterion = c('loo')) %>% 
  print(simplify = F)


loo_compare(model_surrf_lognorm_2, model_surrf_lognorm_3, model_surrf_lognorm_4,
             model_surrf_lognorm_5, model_surrf_lognorm_6, model_surrf_lognorm_7,
            model_surrf_lognorm_8, model_surrf_lognorm_9,
            criterion = c('loo')) %>% print(simplify = F)

# TODO add ID_KTYP model
loo_compare(model_simple_slope, model_surrf_2,model_surrf_3, model_surrf_4, model_full, criterion = c('loo')) %>% 
  print(simplify = T)
model_slope$formula
model_mlvlv$fit

model_weights(model_slope, model_simple, weights = "waic") #%>% round(digits = 3)

