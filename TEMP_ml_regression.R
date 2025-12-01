


library(tidymodels)

set.seed(222)


# Train–test split (75% / 25%)
split_Bochum <- initial_split(df_Bochum_KL, prop = 3/4)
df_Train <- training(split_Bochum)
df_Test  <- testing(split_Bochum)


# Preprocessing recipe (normalize + PCA)

rcp_Norm <- 
  recipe(evapo_r ~ ., data = df_Train) |>
  step_YeoJohnson(all_predictors()) |> 
  step_normalize(all_predictors())


# linear regression --------------

mdl_Mlm <- 
  linear_reg() |>
  set_engine("lm")


# Combine recipe + model

wflow_Norm_Mlm <- 
  workflow() |>
  add_recipe(rcp_Norm) |>
  add_model(mdl_Mlm)


# Fit the workflow on the training data

fit_Norm_Mlm <- 
  wflow_Norm_Mlm |>
  fit(data = df_Train)


# Predict on the test set

pred_Mlm <- 
  predict(fit_Norm_Mlm, df_Test) |>
  bind_cols(df_Test |> select(evapo_r))
metrics(pred_Mlm, truth = evapo_r, estimate = .pred)


# Predicted vs Observed plot

ggplot(pred_Mlm, aes(x = evapo_r, y = .pred)) +
  geom_point(alpha = 0.6, color = color_RUB_blue) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = color_TUD_pink) +
  labs(
    x = "Observed",
    y = "Predicted"
  )






## Ridge model (alpha = 0) -------------

mdl_Ridge <- 
  linear_reg(
    penalty = 0.1,   # lambda
    mixture = 0      # alpha = 0 -> ridge
  ) |>
  set_engine("glmnet")


# Combine recipe + model

wflow_Norm_Ridge <- 
  workflow() |>
  add_recipe(rcp_Norm) |>
  add_model(mdl_Ridge)


# Fit

fit_Norm_Ridge <- 
  wflow_Norm_Ridge |>
  fit(df_Train)


# Predict

pred_Ridge <- 
  predict(fit_Norm_Ridge, df_Test) |>
  bind_cols(df_Test |> select(evapo_r))


# Metrics

metrics(pred_Ridge, truth = evapo_r, estimate = .pred)


## Lasso Regression (L1) ----------

mdl_Lasso <- 
  linear_reg(
    penalty = 0.1,   # lambda
    mixture = 1      # alpha = 1 -> lasso
  ) |>
  set_engine("glmnet")

wflow_Norm_Lasso <- 
  workflow() |>
  add_recipe(rcp_Norm) |>
  add_model(mdl_Lasso)

fit_Norm_Lasso <- 
  wflow_Norm_Lasso |> fit(df_Train)

pred_Lasso <- 
  predict(fit_Norm_Lasso, df_Test) |>
  bind_cols(df_Test |> select(evapo_r))

metrics(pred_Lasso, truth = evapo_r, estimate = .pred)

## Elastic Net Regression (intermediate α) ------------

mdl_Enet <- 
  linear_reg(
    penalty = 0.1,  # lambda
    mixture = 0.5   # alpha between 0 and 1
  ) |>
  set_engine("glmnet")

wflow_Norm_Enet <- 
  workflow() |>
  add_recipe(rcp_Norm) |>
  add_model(mdl_Enet)

fit_Norm_Enet <- 
  wflow_Norm_Enet |> fit(df_Train)

pred_Enet <- 
  predict(fit_Norm_Enet, df_Test) |>
  bind_cols(df_Test |> select(evapo_r))

metrics(pred_Enet, truth = evapo_r, estimate = .pred)

ggplot(pred_Ridge, aes(x = evapo_r, y = .pred)) +
  geom_point(alpha = 0.6, color = color_RUB_blue) +
  geom_abline(intercept = 0, slope = 1, 
              linetype = "dashed", color = color_TUD_pink) +
  labs(
    x = "Observed",
    y = "Predicted",
    title = "Ridge Regression Predictions"
  ) +
  theme_minimal()

# random forest model ------------

mdl_RF <- 
  rand_forest(
    trees = 500,        # you can adjust
    mtry  = NULL,       # default = sqrt(p)
    min_n = 5           # default minimum node size
  ) |>
  set_engine("ranger") |> 
  set_mode("regression")


# Combine recipe + model

wflow_Norm_RF <- 
  workflow() |>
  add_recipe(rcp_Norm) |>
  add_model(mdl_RF)


# Fit the workflow on the training data

fit_Norm_RF <- 
  wflow_Norm_RF |>
  fit(data = df_Train)


# Predict on the test set

pred_RF <- 
  predict(fit_Norm_RF, df_Test) |>
  bind_cols(df_Test |> select(evapo_r))


# Evaluate model performance

metrics(pred_RF, truth = evapo_r, estimate = .pred)


ggplot(pred_RF, aes(x = evapo_r, y = .pred)) +
  geom_point(alpha = 0.6, color = color_RUB_blue) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    color = color_TUD_pink
  ) +
  labs(
    x = "Observed",
    y = "Predicted"
  )

# decision tree ------

mdl_DT <- 
  decision_tree(
    cost_complexity = 0.01,   # cp, can tune
    tree_depth      = NULL,   # max depth, can tune
    min_n           = 5       # minimum node size
  ) |>
  set_engine("rpart") |>
  set_mode("regression")


# Combine recipe + model

wflow_Norm_DT <- 
  workflow() |>
  add_recipe(rcp_Norm) |>
  add_model(mdl_DT)


# Fit the workflow on the training data

fit_Norm_DT <- 
  wflow_Norm_DT |>
  fit(data = df_Train)


# Predict on the test set

pred_DT <- 
  predict(fit_Norm_DT, df_Test) |>
  bind_cols(df_Test |> select(evapo_r))


# Evaluate model performance

metrics(pred_DT, truth = evapo_r, estimate = .pred)


ggplot(pred_DT, aes(x = evapo_r, y = .pred)) +
  geom_point(alpha = 0.6, color = color_RUB_blue) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = color_TUD_pink) +
  labs(
    x = "Observed",
    y = "Predicted",
    title = "Decision Tree Predictions"
  ) +
  theme_minimal()





# SVM regression -------------

mdl_SVM <- 
  svm_rbf(       # Radial basis function kernel
    cost  = 1,   # C parameter, can tune
    rbf_sigma = 0.1  # Kernel width, can tune
  ) |>
  set_engine("kernlab") |>
  set_mode("regression")


# Combine recipe + model

wflow_Norm_SVM <- 
  workflow() |>
  add_recipe(rcp_Norm) |>
  add_model(mdl_SVM)


# Fit the workflow on the training data

fit_Norm_SVM <- 
  wflow_Norm_SVM |>
  fit(data = df_Train)


# Predict on the test set

pred_SVM <- 
  predict(fit_Norm_SVM, df_Test) |>
  bind_cols(df_Test |> select(evapo_r))


# Evaluate model performance

metrics(pred_SVM, truth = evapo_r, estimate = .pred)

ggplot(pred_SVM, aes(x = evapo_r, y = .pred)) +
  geom_point(alpha = 0.6, color = color_RUB_blue) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = color_TUD_pink) +
  labs(
    x = "Observed",
    y = "Predicted",
    title = "SVM Regression Predictions"
  ) +
  theme_minimal()





# Define XGBoost regression model ----------

mdl_XGB <- 
  boost_tree(
    trees = 500,        # number of trees
    tree_depth = 6,     # maximum depth of each tree
    learn_rate = 0.05,  # learning rate
    loss_reduction = 0, # gamma parameter
    min_n = 5           # minimum node size
  ) |>
  set_engine("xgboost") |>
  set_mode("regression")


# Combine recipe + model

wflow_Norm_XGB <- 
  workflow() |>
  add_recipe(rcp_Norm) |>
  add_model(mdl_XGB)


# Fit the workflow on the training data

fit_Norm_XGB <- 
  wflow_Norm_XGB |>
  fit(data = df_Train)


# Predict on the test set

pred_XGB <- 
  predict(fit_Norm_XGB, df_Test) |>
  bind_cols(df_Test |> select(evapo_r))


# Evaluate model performance

metrics(pred_XGB, truth = evapo_r, estimate = .pred)

ggplot(pred_XGB, aes(x = evapo_r, y = .pred)) +
  geom_point(alpha = 0.6, color = color_RUB_blue) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = color_TUD_pink) +
  labs(
    x = "Observed",
    y = "Predicted",
    title = "XGBoost Regression Predictions"
  ) +
  theme_minimal()




# fully connected neural network (DNN) -------------

mdl_ANN <- 
  mlp(
    hidden_units = 32,  # neurons in the hidden layer
    penalty = 0.0,      # optional L2 regularization
    dropout = 0.2
  ) |>
  set_engine("nnet") |>
  set_mode("regression")



# Combine recipe + model

wflow_Norm_ANN <- 
  workflow() |>
  add_recipe(rcp_Norm) |>
  add_model(mdl_ANN)


# Fit the workflow on the training data

fit_Norm_ANN <- 
  wflow_Norm_ANN |>
  fit(data = df_Train)


# Predict on the test set

pred_ANN <- 
  predict(fit_Norm_ANN, df_Test) |>
  bind_cols(df_Test |> select(evapo_r))


# Evaluate model performance

metrics(pred_ANN, truth = evapo_r, estimate = .pred)

ggplot(pred_ANN, aes(x = evapo_r, y = .pred)) +
  geom_point(alpha = 0.6, color = color_RUB_blue) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = color_TUD_pink) +
  labs(
    x = "Observed",
    y = "Predicted",
    title = "ANN / DNN Regression Predictions"
  ) +
  theme_minimal()


# CROSS-validation -------
## folds ---------
set.seed(123)
cv_folds <- vfold_cv(df_Train, v = 10)

## Model with tunable hyperparameters ------------------------------
mdl_EN_tune <- 
  linear_reg(
    penalty = tune(),   # lambda
    mixture = tune()    # alpha
  ) |>
  set_engine("glmnet")

## Workflow -------------------------

wflow_EN_tune <- 
  workflow() |>
  add_recipe(rcp_Norm) |>
  add_model(mdl_EN_tune)

## Hyperparameter grid -------------

grid_EN <- grid_regular(
  penalty(range = c(-4, 1)),   # 10^-4 to 10^1
  mixture(range = c(0, 1)),    # ridge to lasso
  levels = 6
)

## Cross-validated tuning ----------

set.seed(123)
tune_results <- 
  tune_grid(
    wflow_EN_tune,
    resamples = cv_folds,
    grid = grid_EN,
    metrics = metric_set(rmse, mae, rsq),
    control = control_grid(save_pred = TRUE)
  )

## Show best hyperparameters -------

show_best(tune_results, metric = "rmse")

## Finalize model with best hyperparameters -------------------------

best_params <- select_best(tune_results, metric = "rmse")

wflow_final <- finalize_workflow(wflow_EN_tune, best_params)

fit_final <- fit(wflow_final, df_Train)

## Predict on test data ------------

pred_final <- 
  predict(fit_final, df_Test) |>
  bind_cols(df_Test |> select(evapo_r))

metrics(pred_final, truth = evapo_r, estimate = .pred)



# tunable Random Forest model -----------
# 1. Cross-validation folds
set.seed(123)
cv_folds <- vfold_cv(df_Train, v = 10)
# 2. Define tunable Random Forest model
# 1. Model specification with tuning parameters
mdl_RF_Tune <- 
  rand_forest(
    trees = tune(),   # number of trees
    mtry  = tune(),   # number of predictors sampled at split
    min_n = tune()    # minimum node size
  ) |>
  set_engine("ranger") |>
  set_mode("regression")

# 2. Workflow (recipe + model)
wflow_RF_Tune <- 
  workflow() |>
  add_recipe(rcp_Norm) |>
  add_model(mdl_RF_Tune)

# 3. Hyperparameter grid - extract from workflow, not model
param_RF_Tune <- extract_parameter_set_dials(wflow_RF_Tune)

# 4. Update mtry parameter with the actual number of predictors
param_RF_Tune <- param_RF_Tune |>
  update(mtry = mtry(range = c(1, ncol(df_Train) - 1)))  

# 5. Create grid
grid_RF_Reg <- grid_regular(
  param_RF_Tune,
  levels = 3
)
# 5. Cross-validated tuning
set.seed(123)
tuned_RF <- 
  tune_grid(
    wflow_RF_Tune,
    resamples = cv_folds,
    grid = grid_RF_Reg,
    metrics = metric_set(rmse, rsq, mae),
    control = control_grid(save_pred = TRUE)
  )
# 6. Best hyperparameters
show_best(tuned_RF, metric = "rmse")
bestp_RF_Tune <- select_best(tuned_RF, metric = "rmse")
# 7. Final model with tuned hyperparameters
wflow_RF_Final <- 
  finalize_workflow(wflow_RF_Tune, bestp_RF_Tune)

fit_RF_Final <- fit(wflow_RF_Final, df_Train)

pred_RF_Final <- 
  predict(fit_RF_Final, df_Test) |>
  bind_cols(df_Test |> select(evapo_r))

metrics(pred_RF_Final, truth = evapo_r, estimate = .pred)


ggplot(pred_RF_Final, aes(x = evapo_r, y = .pred)) +
  geom_point(alpha = 0.6, color = color_RUB_blue) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    color = color_TUD_pink
  ) +
  labs(
    x = "Observed",
    y = "Predicted"
  )

# Tunable Decision Tree model ------------------------------------

# 1. Cross-validation folds (you already created cv_folds)

# 2. Define tunable Decision Tree model
mdl_tree_tune <- 
  decision_tree(
    cost_complexity = tune(),   # pruning parameter (CCP)
    tree_depth      = tune(),   # maximum depth
    min_n           = tune()    # minimum number of data points in a node
  ) |>
  set_engine("rpart") |>
  set_mode("regression")

# 3. Workflow (recipe + model)
wflow_tree_tune <- 
  workflow() |>
  add_recipe(rcp_Norm) |>
  add_model(mdl_tree_tune)

# 4. Hyperparameter grid
set.seed(123)
grid_tree <- grid_random(
  cost_complexity(range = c(-5, -0.5)),  # .00001 to .3 (log scale)
  tree_depth(range = c(2, 30)),
  min_n(range = c(2, 40)),
  size = 30     # number of random grid combinations
)

# 5. Cross-validated tuning
set.seed(123)
tree_tune_results <- 
  tune_grid(
    wflow_tree_tune,
    resamples = cv_folds,
    grid = grid_tree,
    metrics = metric_set(rmse, rsq, mae),
    control = control_grid(save_pred = TRUE)
  )

# 6. Best hyperparameters
show_best(tree_tune_results, metric = "rmse")
best_tree_params <- select_best(tree_tune_results, metric = "rmse")

# 7. Final model with tuned hyperparameters
wflow_tree_final <- 
  finalize_workflow(wflow_tree_tune, best_tree_params)

fit_tree_final <- fit(wflow_tree_final, df_Train)

# Predict
pred_tree_final <- 
  predict(fit_tree_final, df_Test) |>
  bind_cols(df_Test |> select(evapo_r))

# Metrics
metrics(pred_tree_final, truth = evapo_r, estimate = .pred)

# Plot predicted vs observed
ggplot(pred_tree_final, aes(x = evapo_r, y = .pred)) +
  geom_point(alpha = 0.6, color = color_RUB_blue) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    color = color_TUD_pink
  ) +
  labs(
    x = "Observed",
    y = "Predicted"
  )


# Tunable SVM model (RBF kernel) --------------------------------

# 1. Cross-validation folds (already created: cv_folds)

# 2. Define tunable SVM model
mdl_SVM_Tune <-
  svm_rbf(
    cost      = tune(),   # C penalty
    rbf_sigma = tune()    # kernel width
  ) |>
  set_engine("kernlab") |>
  set_mode("regression")

# 3. Workflow
wflow_SVM_Tune <-
  workflow() |>
  add_recipe(rcp_Norm) |>
  add_model(mdl_SVM_Tune)

# 4. Hyperparameter grid
param_SVM_Tune <- extract_parameter_set_dials(wflow_SVM_Tune)

set.seed(123)
grid_SVM_Random <- grid_random(param_SVM_Tune,
  size = 50
)

# 5. Cross-validated tuning
set.seed(123)
tuned_SVM <-
  tune_grid(
    wflow_SVM_Tune,
    resamples = cv_folds,
    grid = grid_SVM_Random,
    metrics = metric_set(rmse, rsq, mae),
    control = control_grid(save_pred = TRUE)
  )

# 6. Best hyperparameters
show_best(tuned_SVM, metric = "rmse")
bestp_SVM_Tune <- select_best(tuned_SVM, metric = "rmse")

# 7. Final SVM model
wflow_SVM_Final <-
  finalize_workflow(wflow_SVM_Tune, bestp_SVM_Tune)

fit_SVM_Final <- fit(wflow_SVM_Final, df_Train)

# 8. Predict
pred_SVM_Final <-
  predict(fit_SVM_Final, df_Test) |>
  bind_cols(df_Test |> select(evapo_r))

# 9. Metrics
metrics(pred_SVM_Final, truth = evapo_r, estimate = .pred)

# 10. Plot predicted vs observed
ggplot(pred_SVM_Final, aes(x = evapo_r, y = .pred)) +
  geom_point(alpha = 0.6, color = color_RUB_blue) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    color = color_TUD_pink
  ) +
  labs(
    x = "Observed",
    y = "Predicted"
  )

# GBOST --------
library(xgboost)

# 1. Model specification with tunable hyperparameters
mdl_GB_Tune <-
  boost_tree(
    trees = tune(),        # number of trees
    learn_rate = tune(),   # learning rate
    mtry = tune(),         # number of predictors per split
    tree_depth = tune(),   # max depth
    min_n = tune(),        # min node size
    loss_reduction = tune()  # gamma
  ) |>
  set_engine("xgboost") |>
  set_mode("regression")

# 2. Workflow
wflow_GB_Tune <-
  workflow() |>
  add_recipe(rcp_Norm) |>
  add_model(mdl_GB_Tune)

# 3. Parameter set
param_GB_Tune <- extract_parameter_set_dials(wflow_GB_Tune) |> finalize(df_Train)


# 4. Bayesian tuning
set.seed(123)
tuned_GB <-
  tune_bayes(
    wflow_GB_Tune,
    resamples = cv_folds,
    param_info = param_GB_Tune,
    initial = 10,  # initial random points
    iter = 30,     # number of Bayesian iterations
    metrics = metric_set(rmse, rsq, mae),
    control = control_bayes(
      verbose = TRUE,
      save_pred = TRUE
    )
  )

# 5. Best hyperparameters
show_best(tuned_GB, metric = "rmse")
bestp_GB_Tune <- select_best(tuned_GB, metric = "rmse")

# 6. Final GB model
wflow_GB_Final <-
  finalize_workflow(wflow_GB_Tune, bestp_GB_Tune)

fit_GB_Final <- fit(wflow_GB_Final, df_Train)

# 7. Predict
pred_GB_Final <-
  predict(fit_GB_Final, df_Test) |>
  bind_cols(df_Test |> select(evapo_r))

# 8. Metrics
metrics(pred_GB_Final, truth = evapo_r, estimate = .pred)

# 9. Plot predicted vs observed
ggplot(pred_GB_Final, aes(x = evapo_r, y = .pred)) +
  geom_point(alpha = 0.6, color = color_RUB_blue) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    color = color_TUD_pink
  ) +
  labs(
    x = "Observed",
    y = "Predicted"
  )

