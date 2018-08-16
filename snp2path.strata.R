############ Autoencoder for clustering using h2o.kmeans ######
### Author: Chrysanthi Ainali 
### Date: 28.04.2018 (last change @ 11.08.2018)
### Version: 0.1
### Comments: vizualisation code is part of http://jkunst.com/r-material/201703-DataVisualizationMeetup/code.html
############################################### 

# ws & packages -----------------------------------------------------------
rm(list = ls())


# general
library(tidyverse)
library(stringr)
library(lubridate)
library(ggplot)

# cors
library(widyr) # devtools::install_github("dgrtwo/widyr")
library(igraph)

#### H2O -  autoencoder
# The following two commands remove any previously installed H2O packages for R.

if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

# Next, we download packages that H2O depends on.
pkgs <- c("RCurl","jsonlite")
for (pkg in pkgs) {
if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}

# Now we download, install and initialize the H2O package for R.
install.packages("h2o", type="source", repos="http://h2o-release.s3.amazonaws.com/h2o/rel-wheeler/4/R")

# Finally, let's load H2O and start up an H2O cluster
library(h2o)
###  Shut down and restart H2O as shown below to use all your CPUs.
# h2o.shutdown()
h2o.init()

# vizualisation
install.packages("highcharter")
install.packages("hrbrthemes")
#install.packages("partykit")

### download and read the dataset
urlfile <- "snp.genotype.file"
rarfile <- file.path("data", basename(urlfile))

if(!file.exists(rarfile)) {
  download.file(urlfile, file.path("data", basename(urlfile)), mode = "wb")
}



## data <- read.table("data/file", sep='\t', header=TRUE)

data <- dataset2_geno.snpsub15.h2o
##dh2o <- as.h2o(data)


####### deep learning hyperparameters ######
activation_opt <- c("Rectifier", "Maxout", "Tanh")
l1_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01)
l2_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01)

hyper_params <- list(activation = activation_opt, l1 = l1_opt, l2 = l2_opt)
search_criteria <- list(strategy = "RandomDiscrete", max_runtime_secs = 600)



##### autoencoder - build and train the model #####

mix_autoenc <- h2o.deeplearning(
  x = names(dataset2_geno.snpsub15.h2o)[-1],
  training_frame = dataset2_geno.snpsub15.h2o,
  hidden = c(400, 100, 2, 100, 400),
  epochs = 50,
  activation = "Tanh",
  autoencoder = TRUE
)

#### Extract the non-linear features 
dautoenc <- h2o.deepfeatures(mix_autoenc, dataset2_geno.snpsub15.h2o, layer = 3) %>%
  as.data.frame() %>%
  mutate(samples = rownames(dataset2_geno.snpsub15.t))


colnames(dautoenc) <- c("x", "y", "samples")


#######################clustering#############################

#### H2O-KMeans #######
dkmod <- map_df(seq(1, 20, by = 3), function(k){
  mod.km <- h2o.kmeans(training_frame = as.h2o(dautoenc), k = k, x = c("x", "y"))
  mod.km@model$model_summary
})


dkmod <- dkmod %>%
  mutate(wc_ss = within_cluster_sum_of_squares/total_sum_of_squares,
         bt_ss = between_cluster_sum_of_squares/total_sum_of_squares)

## plot the number of clusters that are created in order to decide for the best
plot(dkmod$number_of_clusters, dkmod$wc_ss, "line", xlab = "clusters", ylab="WC_SS")


## final clustering to define the grouping
mod_km <- h2o.kmeans(training_frame = as.h2o(dautoenc), k = 4, x = c("x", "y"))

### grouping ###
dautoenc.gr <- dautoenc %>%
  mutate(group = as.vector(h2o.predict(object = mod_km, newdata = as.h2o(.))),
         group = as.numeric(group) + 1,
         group = paste("grupo", group))

	

#### Visualisation
## plot the groups
ggplot(dautoenc.gr,  aes(x = x, y = y)) +
 geom_point(aes(col=group))




