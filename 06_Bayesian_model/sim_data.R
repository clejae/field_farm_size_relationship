library(dplyr)
library(ggplot2)

################################################################################
# create simulated data 
# simulate field sizes in hectare
fs_sim <- runif(10000, 1, 200) 

# assign random farm IDs
farm_id <- rnorm(10000, 100, 200) %>% round(0)


#farm_id <- runif(10000, 1, 100) %>% round(0)

farm_id %>% unique() %>% length()
# create data frame
df_sim <- data.frame(fs = fs_sim, farm_id = farm_id)

# calculate farm size by summing all fields of a farm
grouped <- df_sim %>% group_by(farm_id) %>% 
  summarise(farm_size = sum(fs)) 

# add farm size column to dataframe
df_sim = merge(df_sim, grouped)
# calculate the number of fields per farm 
grouped <- df_sim %>% group_by(farm_id) %>% 
  count()
df_sim = merge(df_sim, grouped)


################################################################################

# average number of fields per farm
mean(df_sim$n)
# Plot FSFSR for the simulated data
ggplot(df_sim, aes((fs), (farm_size))) + geom_point() + 
  geom_smooth(method = 'lm') + 
  xlab('field size ha') + ylab('farm size ha') + ggtitle('simulated data')

lm(farm_size ~ fs, data=df_sim) %>% summary()

# -> Positive relationship between the two variables
################################################################################
# subtracting field size from farm size
df_sim$farm_size <- df_sim$farm_size - df_sim$fs
# Plot FSFSR after subtraction of field size from farm size
ggplot(df_sim, aes((fs), (farm_size))) + geom_point() + 
  geom_smooth(method = 'lm') + 
  xlab('field size m2') + ylab('farm size m2') + ggtitle('simulated data')
lm(farm_size ~ fs, data=df_sim) %>% summary()

# -> No positive relationship between the two variables anymore
