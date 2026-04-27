rm(list = ls()) ; graphics.off()

require(reshape2)
require(ggplot2)
require(dplyr)

require(glmnet)

setwd("//wsl.localhost/Ubuntu/home/acaillebotte/projects/packages/sdg4varselect/work/chap1")


# =======================================#
#             LOADING DATA               #
# =======================================#
fn <- paste0("simulation_files/PKMEM_N300_J12_P1000_data.csv")

all_data = read.csv2(fn, sep = ";", dec = ".") 


fn <- paste0("simulation_files/PKMEM_N300_J12_P1000_cov.csv")

all_cov = read.csv2(fn, sep = ";", dec = ".") 


# data %>% ggplot(aes(times, Y)) + 
#   geom_line(aes(col = factor(id))) + 
#   theme(legend.position = "None")


# =======================================#
#               FCT 4 NLM                #
# =======================================#

g <- function(t, D,V, phi){
  D * phi[1]/(V*phi[1] - phi[2]) *
    (exp(-phi[2]/V*t)-exp(-phi[1]*t))
}

L <- function(phi,dt){
  sum((dt$Y - g(dt$times, D = 100, V = 30, phi = phi))^2)
}

mu = c(6,8)

# =======================================#
#                  FIT                   #
# =======================================#
i = 0
p_partial <- 0.1
third_obs = unique(all_data$times)[3]
out = list()
for(p_partial in c(0.0,0.1,0.2,0.3,0.4)){
  for(i in 0:299){
    print(paste0("i = ", i, ", p_partial = ", p_partial))
    
    data <- all_data %>% filter(seed == i) %>% dplyr::select(-seed) %>% 
      filter(id <= round(p_partial*max(id)) | (times <= third_obs ))
    
    cov <- all_cov %>% filter(seed == i) %>% dplyr::select(-seed) %>%
      as.matrix
  
    # nlm(L, p = c(5,5), dt = data[which(data$id == 1),])$estimate
    phi_hat <- data %>% group_by(id) %>%
      # group_map(~ head(.x, 2L))
      group_modify(~ data.frame(phi = t(nlm(L, p = c(5,5), dt = .x)$estimate)))
    
    phi_hat <- as.matrix(phi_hat[,2:3] - mu)
    phi_hat 
    
    # =======================================#
    #                 GLMNET                 #
    # =======================================#
    resmg.cv <- cv.glmnet(cov, phi_hat, family = "mgaussian", alpha=1)
    resmg.net <- glmnet(cov, phi_hat, family = "mgaussian", alpha = 1, lambda = resmg.cv$lambda.1se)
    
    resg1.cv <- cv.glmnet(cov, phi_hat[,1], family = "gaussian", alpha=1)
    resg1.net <- glmnet(cov, phi_hat[,1], family = "gaussian", alpha = 1, lambda = resg1.cv$lambda.1se)
    resg2.cv <- cv.glmnet(cov, phi_hat[,2], family = "gaussian", alpha=1)
    resg2.net <- glmnet(cov, phi_hat[,2], family = "gaussian", alpha = 1, lambda = resg2.cv$lambda.1se)
    
    out[[i+1]] = list(
      mg = list(lambda.1se = resmg.cv$lambda.1se,
                beta_hat = resmg.net$beta),
      g = list(lambda.1se = c(resg1.cv$lambda.1se,resg2.cv$lambda.1se),
               beta_hat = list(phi.1 = resg1.net$beta), phi.2 = resg2.net$beta),
      phi_hat = phi_hat
    )
  
  }
  out[[i+1]]
  
  
  saveRDS(out, file = paste0("two_step/p", p_partial, ".rds"))
}













