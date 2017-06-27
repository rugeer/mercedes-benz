library(dplyr)
library(caret)
library(plyr)
library(Metrics)
library(xgboost)
library(ranger)
library(nnet)
library(Metrics)
library(kernlab)
library(LiblineaR)
library(data.table)
train.raw <- train.raw[which(train.raw$y!=265.32),]
#data preparation
col_names <- colnames(train.raw[,-c(1,2)])
# Determine data types in the data set
data_types <- sapply(col_names,function(x){class(train.raw[[x]])})
unique_data_types <- unique(data_types)
# Separate attributes by data type
DATA_ATTR_TYPES <- lapply(unique_data_types,function(x){ names(data_types[data_types == x])})
names(DATA_ATTR_TYPES) <- unique_data_types
#################################################################################################################
#taking a random sample to build level 0 models
set.seed(123)
train.rows <- sample(nrow(train.raw), nrow(train.raw)*0.8)
#################################################################################################################
#Create Level 0 Model Feature Sets
# Feature Set 1 - Boruta Confirmed and tentative Attributes
prepL0FeatureSet1 <- function(df) {
  id <- df$ID
  y <- df$y
  predictor_vars <- col_names
  predictors <- df[predictor_vars]
  # for numeric set missing values to -1 for purposes
  num_attr <- intersect(predictor_vars,DATA_ATTR_TYPES$integer)
  for (x in num_attr){
    predictors[[x]][is.na(predictors[[x]])] <- -1
  }
  # for character  atributes set missing value
  char_attr <- intersect(predictor_vars,DATA_ATTR_TYPES$character)
  for (x in char_attr){
    predictors[[x]][is.na(predictors[[x]])] <- "*MISSING*"
    predictors[[x]] <- factor(predictors[[x]])
  }
  
  return(list(id=id,y=y,predictors=predictors))
}
L0FeatureSet1 <- list(train=prepL0FeatureSet1(train.raw),
                      test=prepL0FeatureSet1(test.raw))
common_features <- intersect(colnames(L0FeatureSet1$test$predictors), colnames(L0FeatureSet1$train$predictors))
L0FeatureSet1$train$predictors <- L0FeatureSet1$train$predictors[,c(common_features)]
L0FeatureSet1$test$predictors <- L0FeatureSet1$test$predictors[,c(common_features)]


###########################################################################################################################
# Feature Set 2 (xgboost) - Boruta Confirmed Attributes
prepL0FeatureSet2 <- function(df) {
  id <- df$ID
  if (class(df$y) != "NULL") {
    y <- df$y
  } else {
    y <- NULL
  }
  
  predictor_vars <- col_names
  predictors <- df[predictor_vars]
  # for numeric set missing values to -1 for purposes
  num_attr <- intersect(predictor_vars,DATA_ATTR_TYPES$integer)
  for (x in num_attr){
    predictors[[x]][is.na(predictors[[x]])] <- NA
  }
  # for character  atributes set missing value
  char_attr <- intersect(predictor_vars,DATA_ATTR_TYPES$character)
  for (x in char_attr){
    predictors[[x]][is.na(predictors[[x]])] <- NA
    predictors[[x]] <- as.factor(predictors[[x]])
  }
  predictors <- model.matrix(~., data=predictors)[,-1]
  return(list(id=id,y=y,predictors=predictors))
}


L0FeatureSet2 <- list(train=prepL0FeatureSet2(train.raw),
                      test=prepL0FeatureSet2(test.raw))

common_features <- intersect(colnames(L0FeatureSet2$test$predictors), colnames(L0FeatureSet2$train$predictors))
L0FeatureSet2$train$predictors <- L0FeatureSet2$train$predictors[,c(common_features)]
L0FeatureSet2$test$predictors <- L0FeatureSet2$test$predictors[,c(common_features)]

