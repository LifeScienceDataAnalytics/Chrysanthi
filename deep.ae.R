#################################################
## Author: Chrysanthi Ainali
## Date: 22.03.2018
## last updated: 14.05.2018
################################################

### step 1:collection of autoencoders where each one learns the underlying manifold of a group of similar objects
## running deep learning model in H2O, autoencoders are trained simultaneously

# install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o/latest_stable_R")))

## upgrading h2o packages
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

###
library(h2o)
# call h2o - use of all available cores
h2o.init(nthreads = -1)
h2o.no_progress()  # Disable progress bars for Rmd


### example dataset
# This step takes a few seconds bc we have to download the data from the internet...
train_file <- "https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/train.csv.gz"
test_file <- "https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/test.csv.gz"
train <- h2o.importFile(train_file)
test <- h2o.importFile(test_file)

y <- "C785"  #response column: digits 0-9
x <- setdiff(names(train), y)  #vector of predictor column names

# Since the response is encoded as integers, we need to tell H2O that
# the response is in fact a categorical/factor column.  Otherwise, it 
# will train a regression model instead of multiclass classification.
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])


######### parameters definition ###############
# x:  is a vector containing names of the columns with training data
# y: the response column used as the dependent variable
# training_frame: specify the dataset used to build the model
# validation_frame: specify the dataset used to evaluate the accuracy of the model (optional)
# nfolds: specify the number of folds for cross-validation
# distribution: specify loss function - can take the values ‘bernoulli’, ‘multinomial’, ‘poisson’, ‘gamma’, ‘tweedie’, ‘laplace’, ‘huber’ or 
# ‘gaussian’, while ‘AUTO’ automatically picks a parameter based on the data
# activation: possible values are ‘Tanh’, ‘TanhWithDropout’, ‘Rectifier’, ‘RectifierWithDropout’, ‘Maxout’ or ‘MaxoutWithDropout’
# sparse: a boolean value denoting a high degree of zeros
# autoencoder: specifies that we need a deep autoencoder instead of a feed-forward Network 
# input_dropout_ratio: Specify the input layer dropout ratio to improve generalization. Suggested values are 0.1 or 0.2
# standardize: if TRUE then automatically is normalising the dataset
# epochs: specify the number of times to iterate the dataset 
# hidden: specify the hidden layer sizes
# l1: specify the L1 regularization to add stability and improve generalization (minimizing the error function)
################################################


###### Simple model ######
# train the model with 2 hidden layers of 20 neurons each

model_fit1 <- h2o.deeplearning(x=x,  y=y,  
			 # model_id= model_fit1,
			  training_frame=train, 
			  validation_frame=test,  
			  distribution="multinomial",  
			  activation="RectifierWithDropout",
                          hidden=c(20,20),
			  input_dropout_ratio=0.2, 
			  sparse=TRUE, 
 			  l1=1e-5, 
                          #stopping_rounds = 0,  # disable early stopping
			  epochs=200)

### Train a DNN with early stopping ### 
# turn on early stopping and specify the stopping criterion - use 5 fold cross validation

model_fit2 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
			    validation_frame = test,
                            model_id = "modl_fit2",
                            distribution="multinomial",
                            activation="RectifierWithDropout",
                            input_dropout_ratio=0.2,
                            sparse=TRUE,
                            l1=1e-5,
                            epochs = 50,
                            hidden = c(20,20),
                            nfolds = 5,                            #used for early stopping
                            score_interval = 1,                    #used for early stopping
                            stopping_rounds = 5,                   #used for early stopping
                            stopping_metric = "misclassification", #used for early stopping
                            stopping_tolerance = 1e-3,             #used for early stopping
                            seed = 1)

# Performance comparison 
dl_perf1 <- h2o.performance(model = model_fit1, newdata = test)
dl_perf2 <- h2o.performance(model = model_fit2, newdata = test)


# Retreive test set MSE
h2o.mse(dl_perf1)
h2o.mse(dl_perf2) 

# Get the CV models from the `dl_fit3` object
cv_models <- sapply(model_fit2@model$cross_validation_models, 
                    function(i) h2o.getModel(i$name))

# Plot the scoring history over time
plot(cv_models[[1]], 
     timestep = "epochs", 
     metric = "classification_error")


###### Hyperparameter optimization ######
## First define a grid of Deep Learning hyperparameters and specify the search_criteria

activation_opt <- c("Rectifier", "Maxout", "Tanh", "RectifierWithDropout")
#activation_opt <- c("Rectifier", "TahnWithDropout", "Tanh", "RectifierWithDropout")
l1_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01)
l2_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01)

hyper_params <- list(activation = activation_opt, l1 = l1_opt, l2 = l2_opt)
search_criteria <- list(strategy = "RandomDiscrete", max_runtime_secs = 600)

dl_grid <- h2o.grid("deeplearning", x = x, y = y,
                    grid_id = "dl_grid",
                    training_frame = train,
                    validation_frame = test,
		    nfolds=3,
                    seed = 1,
		   # autoencoder = TRUE,
                    hidden = c(20,20),
                    hyper_params = hyper_params,
                    search_criteria = search_criteria)


# evaluate performance
dl_gridperf <- h2o.getGrid(grid_id = "dl_grid", 
                           sort_by = "accuracy", 
                           decreasing = TRUE)
print(dl_gridperf)

# Grab the model_id for the top DL model, chosen by validation error
best_dl_model_id <- dl_gridperf@model_ids[[1]]
best_dl <- h2o.getModel(best_dl_model_id)

# evaluate the model performance
best_dl_perf <- h2o.performance(model = best_dl, newdata = test)
h2o.mse(best_dl_perf)


#### Run DNN with autoencoder applying unsupervised training first and then supervised
## Split the training data into two pieces: one that will be used for unsupervised pre-training and 
## the other that will be used for supervised training

splits <- h2o.splitFrame(train, 0.5, seed = 1)

# first part of the data, without labels for unsupervised learning
train_unsupervised <- splits[[1]]

# second part of the data, with labels for supervised learning
train_supervised <- splits[[2]]

dim(train_supervised)
dim(train_unsupervised)

### train autoencoder model
hidden <- c(128, 64, 128)
ae_model <- h2o.deeplearning(x = x, 
                             training_frame = train_unsupervised,
                             model_id = "mnist_autoencoder",
                             ignore_const_cols = FALSE,
                             activation = "Tanh",  # Tanh is good for autoencoding
                             hidden = hidden,
                             autoencoder = TRUE)


## use pre-trained model for supervised DNN
## use of the weights from the unsupervised autoencoder model
fit1 <- h2o.deeplearning(x = x, y = y,
                         training_frame = train_supervised,
                         ignore_const_cols = FALSE,
                         hidden = hidden,
                         pretrained_autoencoder = "mnist_autoencoder")
perf1 <- h2o.performance(fit1, newdata = test)
h2o.mse(perf1)



## Dimension reduction 
# use of h2o.deepfeatures()



###############################
# disconnect with h2o
# h2o.shutdown()


### step 2:

