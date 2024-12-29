# Load necessary libraries
library(caret)
library(randomForest)
library(nnet)
library(e1071)
library(xgboost)
library(SHAPforxgboost)
library(pROC)

#### Import datasets #####
metaset <- read.csv("meta_datset.csv")
validation <- read.csv("validation_dataset.csv")
Validation_target <- as.factor(validation$Target)

##### Split meta dataset into training and testing sets (80/20)
set.seed(1234)
index <- createDataPartition(metaset$Target, p = 0.8, list = FALSE)
Train_1 <- metaset[index, ]
Test_1 <- metaset[-index, ]

Train_1_target <- as.factor(Train_1$Target)
Test_1_target <- as.factor(Test_1$Target)
Train_1_genes <- as.data.frame(Train_1[, -1])
Test_1_genes <- as.data.frame(Test_1[, -1])

##### Feature Selection using SHAP ######
xgb_params <- list(eta = 0.01, max_depth = 10, eval_metric = "auc")
xgb_model <- xgboost(data = as.matrix(Train_1_genes),
                     label = as.numeric(as.character(Train_1_target)),
                     nrounds = 50,
                     params = xgb_params,
                     verbose = 1)

shap_values <- shap.values(xgb_model, 
                           X_train = as.matrix(Train_1_genes))

shap_a <- as.data.frame(shap_values$mean_shap_score)
SHAP_a <- tibble::rownames_to_column(shap_a, "Features")
shap <- SHAP_a$Features[SHAP_a$`shap_values$mean_shap_score` > 0]
write.csv(shap, file = "Selected_features_SHAP.csv")

# Split training data further into 70/30 for model training and testing
set.seed(5678)
train_index <- createDataPartition(Train_1_target, 
                                   p = 0.7, 
                                   list = FALSE)
Train_2 <- Train_1[train_index, ]
Test_2 <- Train_1[-train_index, ]
Train_2_target <- as.factor(Train_2$Target)
Test_2_target <- as.factor(Test_2$Target)

##### Prepare Train2, Test2, Test1 data using SHAP features #####
Train_2_SHAP <- Train_2[,..shap]
Test_2_SHAP <- Test_2[,..shap]
Test_1_SHAP <- Test_1[,..shap]
Validation_SHAP <- validation[, ..shap]

##### Create a list of test sets #####
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
model_names <- c("RandomForest", "ANN", "SVM_Radial", "SVM_Polynomial")
# Prepare a list to store results for all models across all test sets
all_results <- list()

##### Train and Evaluate Models #####
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
  
  ##### Train model on internal training data (Train_2_SHAP)
  trained_model <- caret::train(x = test_sets$internal$Train_data,
                                y = test_sets$internal$Train_target,
                                method = model_id,,
                                tuneLength = 5,
                                trControl = trainControl(method = "cv", 
                                                         number = 10,
                                                         verboseIter = TRUE))
  
    # Initialize a list to store model-specific results
    model_results <- list()
  
    # Loop through each test set (internal, external, validation)
    for (test_set_name in names(test_sets)) {
    # Get the corresponding test set data and target
    test_data <- test_sets[[test_set_name]]$Test_data
    test_target <- test_sets[[test_set_name]]$Test_target
    
    # Predict on the test set
    predictions <- predict(trained_model, test_data)
    
    # Calculate confusion matrix
    cm <- confusionMatrix(predictions, test_target)
    
    # Calculate AUC
    auc_value <- auc(as.numeric(test_target), as.numeric(predictions))
    
    # Store results
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
write.csv(results_df, "Model_Evaluation_Results_SHAP.csv", row.names = FALSE)

