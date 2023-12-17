library(tidyverse)
library(vroom)
library(tidymodels)
library(themis)
library(discrim)
library(embed)

AmazonTraining <- vroom("Desktop/STAT348/AmazonEmployeeAccess/train.csv")
AmazonTest <- vroom("Desktop/STAT348/AmazonEmployeeAccess/test.csv")

#CHANGE ACTION to a FACTOR
AmazonTraining <- AmazonTraining%>%
  mutate(ACTION = as.factor(ACTION))





myAmazonReceta <-
  recipe(ACTION~., data=AmazonTraining) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(all_outcomes(), neighbors = 5)

# prep <- prep(myAmazonReceta)
# bake <- bake(prep, new_data = AmazonTraining)






#-----------LOGISTIC REGRESSION-----------#

logmod <- logistic_reg() %>%
  set_engine("glm")


# set up the workflow

logistic_wf <- workflow() %>%
  add_recipe(myAmazonReceta) %>%
  add_model(logmod) %>%
  fit(data = AmazonTraining)


logistic_preds <- predict(logistic_wf, new_data = AmazonTest,
                          type = "prob")

# prep for kaggle

logistic_output <- tibble(id = AmazonTest$id, Action = logistic_preds$.pred_1)

vroom_write(logistic_output, "logisticPreds.csv", delim = ",")





#-----------PENALIZED LOGISTIC REGRESSION-----------#


# set up model and workflow
penalized_model <- logistic_reg(mixture = tune(), penalty = tune()) %>%
  set_engine("glmnet")

penalized_wf <- 
  workflow() %>%
  add_recipe(myAmazonReceta) %>%
  add_model(penalized_model)

## set up a tuning grid
tuning_grid <-
  grid_regular(penalty(),
               mixture(),
               levels = 5)

## split into folds
folds <- vfold_cv(AmazonTraining, v = 5, repeats = 1)

# Cross Validation

CV_results <-
  penalized_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# finding best tuning parameter values

best_tune <-
  CV_results %>%
  select_best("roc_auc")

# finalize wf and get preds

final_wf <-
  penalized_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = AmazonTraining)

penalized_preds <-
  final_wf %>%
  predict(new_data = AmazonTest, type = "prob")

#Prep for Kaggle

penalized_logistic_output <- tibble(id = AmazonTest$id, Action = penalized_preds$.pred_1)

vroom_write(penalized_logistic_output, "Penalized_Preds.csv", delim = ",")


#-------------RANDOM FORESTS--------------#


# set up model and workflow
randomforestmodel <- rand_forest(mtry = tune(), min_n = tune(),
                      trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

randomforestwf <- 
  workflow() %>%
  add_recipe(myAmazonReceta) %>%
  add_model(randomforestmodel)

#tuning grid
tuning_grid <-
  grid_regular(mtry(range = c(1,9)),
               min_n(),
               levels = 5)

# split into folds
folds <- vfold_cv(AmazonTraining, v = 5, repeats = 1)

# run cross-validation

CV_results <-
  randomforestwf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

#tuning parameters

best_tune <-
  CV_results %>%
  select_best("roc_auc")

# finalize wf and get preds

finalrf_wf <-
  randomforestwf %>%
  finalize_workflow(best_tune) %>%
  fit(data = AmazonTraining)

rf_preds <-
  finalrf_wf %>%
  predict(new_data = AmazonTest, type = "prob")

# prepare for kaggle, output csv file

rf_values <- tibble(id = AmazonTest$id, Action = rf_preds$.pred_1)

vroom_write(rf_values, "RandomForestAmazonPreds.csv", delim = ",")






#----------NAIVE BAYES----------#




nb_model <-
  naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")


nb_wf <-
  workflow() %>%
  add_recipe(myAmazonReceta) %>%
  add_model(nb_model)

## set up a tuning grid
tuning_grid <-
  grid_regular(Laplace(),
               smoothness(),
               levels = 5)

## split into folds
folds <- vfold_cv(AmazonTraining, v = 5, repeats = 1)

# run cv

CV_results <-
  nb_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# find best tuning parameter values

best_tune <-
  CV_results %>%
  select_best("roc_auc")

# finalize wf and get preds

final_wf <-
  nb_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = AmazonTraining)

nb_preds <-
  final_wf %>%
  predict(new_data = AmazonTest, type = "prob")

# prepare and export preds to csv for kaggle

nb_output <- tibble(id = AmazonTest$id, Action = nb_preds$.pred_1)

vroom_write(nb_output, "AmazonNaiveBayesPreds.csv", delim = ",")





#-----------K-NEAREST NEIGHBORS-----------#


#set up model and workflow
knn_model <-
  nearest_neighbor(neighbors = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")


knn_wf <-
  workflow() %>%
  add_recipe(myAmazonReceta) %>%
  add_model(knn_model)

#tuning grid
tuning_grid <-
  grid_regular(neighbors(),
               levels = 5)

#split into folds
folds <- vfold_cv(AmazonTraining, v = 5, repeats = 1)

#run cross validation

CV_results <-
  knn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# find best tuning parameter values

best_tune <-
  CV_results %>%
  select_best("roc_auc")

# finalize wf and get preds

final_wf <-
  knn_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = AmazonTraining)

knn_preds <-
  final_wf %>%
  predict(new_data = AmazonTest, type = "prob")

# prepare and export preds to csv for kaggle

knn_values <- tibble(id = AmazonTest$id, Action = knn_preds$.pred_1)


vroom_write(knn_values, "KnnAmazonPredictions", delim = ",")





#-----------Support Vector Machines-----------#


SVM_mod <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")


svm_workflow <- 
  workflow() %>%
  add_recipe(myAmazonReceta) %>%
  add_model(SVM_mod)

#set up a tuning grid
tuning_grid <-
  grid_regular(rbf_sigma(),
               cost(),
               levels = 5)

#split into folds
folds <- vfold_cv(AmazonTraining, v = 5, repeats = 1)

#run cross validation

CV_results <-
  svm_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

#optimize tuning parameters

best_tune <-
  CV_results %>%
  select_best("roc_auc")

# finalize wf and get preds

finalsvm_wf <-
  svm_workflow %>%
  finalize_workflow(best_tune) %>%
  fit(data = AmazonTraining)

svm_preds <-
  finalsvm_wf %>%
  predict(new_data = AmazonTest, type = "prob")

#Create CSV for kaggle preds

svm_values <- tibble(id = AmazonTest$id, Action = svm_preds$.pred_1)
vroom_write(svm_output, "SVMAmazonPredictions.csv", delim = ",")

