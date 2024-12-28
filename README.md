# Cancer Classification Pipeline with Feature Selection and Machine Learning
 This repository proposes a comprehensive workflow for feature selection and machine learning-based breast cancer classification. The pipeline is built for reliable feature selection and model evaluation across internal, external, and independent validation datasets. A detailed explanation of the procedure, complete with pertinent code snippets, is provided below.
 ## Workflow Overview
 This workflow follows these main steps:


### 1.	Data Preparation
 * Split the **meta dataset** (61 Normal, 71 Cancer) into:  
    * **External Training Set:** (80%) for feature selection and training  
    * **External Testing Set:** (20%) for model evaluation  
 * 	Use an independent **validation dataset** (42 Normal, 42 Cancer) for additional evaluation
   # Load datasets
   ```
metaset <- read.csv("meta_dataset.csv")
validation <- read.csv("validation_dataset.csv")
```

# Split meta dataset into training and testing sets (80/20)
```
set.seed(1234)
library(caret)
index <- createDataPartition(metaset$Target, p = 0.8, list = FALSE)
Train_1 <- metaset[index,]
Test_1 <- metaset[-index,]
```
### 2.	Feature Selection
*	Select robust features using two methods: 
    * **VarSelRF** (Variable Selection with Random Forest)  
    * **SHAP** (SHapley Additive exPlanations)

# Feature selection with VarSelRF  
```
library(randomForest)  
library(varSelRF)

set.seed(42)  
Train_1_genes <- Train_1[, -1]  
Train_1_target <- as.factor(Train_1$Target)  
varSelRF_model <- varSelRF(Train_1_genes,
                           Train_1_target,
                           c.sd = 1, 
                           mtryFactor = 1, 
                           ntree = 5000,
                           ntreeIterat = 2000,
                           vars.drop.num = NULL,
                           vars.drop.frac = 0.2,
                           whole.range = TRUE, 
                           recompute.var.imp = FALSE,
                           verbose = FALSE,
                           returnFirstForest = TRUE, 
                           fitted.rf = NULL, 
                           keep.forest = FALSE)
# Selected features  
var_features <- varSelRF_model$selected.vars
write.csv(var, file = "Selected_features_varselRF.csv")
```
# Feature selection with SHAP  
```
library(xgboost)
library(SHAPforxgboost)
# Train an XGBoost model for SHAP
xgb_params <- list(eta = 0.01, max_depth = 10, eval_metric = "auc")
xgb_model <- xgboost(data = as.matrix(Train_1_genes),
                     label = as.numeric(as.character(Train_1_target)),
                     nrounds = 50,
                     params = xgb_params,
                     verbose = 1)

shap_values <- shap.values(xgb_model, 
                           X_train = as.matrix(Train_1_genes))
# Compute SHAP values  
shap_a <- as.data.frame(shap_values$mean_shap_score)
SHAP_a <- tibble::rownames_to_column(shap_a, "Features")
shap <- SHAP_a$Features[SHAP_a$`shap_values$mean_shap_score` > 0]
write.csv(shap, file = "Selected_features_SHAP.csv")
``` 

### 3.	Data Splitting  
* Created trimmed datasets using features selected by each method  
* Split the trimmed external training set into:  
    * **Internal Training Set:** (70%) for model training  
    * **Internal Testing Set:** (30%) for internal evaluation
 # Split external training set into internal training/testing sets (70/30) 
 ```
set.seed(5678)
train_index <- createDataPartition(Train_1_target, 
                                   p = 0.7, 
                                   list = FALSE)
Train_2 <- Train_1[train_index, ]
Test_2 <- Train_1[-train_index, ]
Train_2_target <- as.factor(Train_2$Target)
Test_2_target <- as.factor(Test_2$Target)
```
# Prepare Train2, Test2, Test1 data using varselRF features   
```
Train_2_varselRF <- Train_2[,..var]
Test_2_varselRF <- Test_2[,..var]
Test_1_varselRF <- Test_1[,..var]
Validation_varselRF <- validation[, ..var]

## Create a list of test sets  
test_sets_varSelRF <- list(
  internal = list(Train_data = Train_2_varselRF, 
                  Test_data = Test_2_varselRF, 
                  Train_target = Train_2_target,
                  Test_target = Test_2_target),
  external = list(Train_data = Train_2_varselRF, 
                  Test_data = Test_1_varselRF, 
                  Train_target = Train_2_target, 
                  Test_target = Test_1_target),
  validation = list(Train_data = Train_2_varselRF, 
                    Test_data = Validation_varselRF,
                    Train_target = Train_2_target, 
                    Test_target = Validation_target)
)
```
# Prepare Train2, Test2, Test1 data using SHAP features 
```
Train_2_SHAP <- Train_2[,..shap]
Test_2_SHAP <- Test_2[,..shap]
Test_1_SHAP <- Test_1[,..shap]
Validation_SHAP <- validation[, ..shap]

# Create a list of test sets
test_sets_SHAP <- list(
  internal = list(Train_data = Train_2_SHAP,
                  Test_data = Test_2_SHAP, 
                  Train_target = Train_2_target, 
                  Test_target = Test_2_target),
  external = list(Train_data = Train_2_SHAP,
                  Test_data = Test_1_SHAP,
                  Train_target = Train_2_target, 
                  Test_target = Test_1_target),
  validation = list(Train_data = Train_2_SHAP,
                    Test_data = Validation_SHAP, 
                    Train_target = Train_2_target, 
                    Test_target = Validation_target)
)
``` 
### 4.	Classification
* Train machine learning models on the internal training set, using selected features
* Evaluate models on the internal testing set, external testing set, and validation dataset  
Machine learning models include:
    * Random Forest (RF)
    * Artificial Neural Networks (ANN)
    * Support Vector Machines (SVM) with radial kernel
    * Support Vector Machines (SVM) with polynomial kernel
  # Train and Evaluate Models
```
# Prepare a list to store results for all models across all test sets
model_names <- c("RandomForest", "ANN", "SVM_Radial", "SVM_Polynomial")
# Initialize a list to store model-specific results
model_results <- list()
all_results <- list()

set.seed(7890)
# Loop through models
for (model_name in model_names) {
cat("\nTraining and evaluating", model_name, "...\n")
  
# Select model identifier for caret
model_id <- switch(model_name,
                   "RandomForest" = "rf",
                    "ANN" = "nnet",
                    "SVM_Radial" = "svmRadial",
                    "SVM_Polynomial" = "svmPoly")
```
# Train and Evaluate Models with varSelRF datasets
```
trained_model_varSelRF <- caret::train(x = test_sets_varSelRF$internal$Train_data,
                                y = test_sets_varSelRF$internal$Train_target,
                                method = model_id,
                                tuneLength = 5,
                                trControl = trainControl(method = "cv", 
                                                         number = 10,
                                                         verboseIter = TRUE))
  
# Loop through each test set (internal, external, validation)
for (test_set_name in names(test_sets_varSelRF)) {
# Get the corresponding test set data and target
test_data <- test_sets_varSelRF[[test_set_name]]$Test_data
test_target <- test_sets_varSelRF[[test_set_name]]$Test_target
    
# Predict on the test set
predictions <- predict(trained_model_varSelRF, test_data)
    
# Calculate confusion matrix
cm <- confusionMatrix(predictions, test_target)
    
# Calculate AUC
auc_value <- auc(as.numeric(test_target), as.numeric(predictions))

```
# Train and Evaluate Models with SHAP datasets
```
trained_model_SHAP <- caret::train(x = test_sets_SHAP$internal$Train_data,
                                y = test_sets_SHAP$internal$Train_target,
                                method = model_id,
                                tuneLength = 5,
                                trControl = trainControl(method = "cv", 
                                                         number = 10,
                                                         verboseIter = TRUE))
  
# Loop through each test set (internal, external, validation)
for (test_set_name in names(test_sets_SHAP)) {
# Get the corresponding test set data and target
test_data <- test_sets_SHAP[[test_set_name]]$Test_data
test_target <- test_sets_SHAP[[test_set_name]]$Test_target
    
# Predict on the test set
predictions <- predict(trained_model_SHAP, test_data)
    
# Calculate confusion matrix
cm <- confusionMatrix(predictions, test_target)
    
# Calculate AUC
auc_value <- auc(as.numeric(test_target), as.numeric(predictions))
```

### 5.	Model Evaluation
Evaluate models using: 
* Accuracy
* Sensitivity
* Specificity
* Recall
*	F1 Score
*	AUC
  
# Store results
```
    model_results[[test_set_name]] <- list(
      confusion_matrix = cm,
      auc = auc_value
    )
  }
  
  # Add model-specific results to the overall results
  all_results[[model_name]] <- model_results
}

##### Organize Results into a Data Frame #####
results_df <- do.call(rbind, lapply(names(all_results), function(model_name) {
  model_results <- all_results[[model_name]]
  
  do.call(rbind, lapply(names(model_results), function(test_set_name) {
    cm <- model_results[[test_set_name]]$confusion_matrix
    auc_value <- model_results[[test_set_name]]$auc
    
    data.frame(
      Model = model_name,
      TestSet = test_set_name,
      Accuracy = cm$overall["Accuracy"],
      Sensitivity = cm$byClass["Sensitivity"],
      Specificity = cm$byClass["Specificity"],
      Precision = cm$byClass["Pos Pred Value"],
      Recall = cm$byClass["Recall"],
      F1 = cm$byClass["F1"],
      AUC = auc_value,
      stringsAsFactors = FALSE
    )
  }))
}))

# Print final results
print(results_df)

# Save results as a CSV file for future reference
write.csv(results_df, "Model_Evaluation_Results_varSelRF.csv", row.names = FALSE) ## For varSelRF Results
or
write.csv(results_df, "Model_Evaluation_Results_SHAP.csv", row.names = FALSE) ## For SHAP Results
```  
