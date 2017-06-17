library(dplyr)
library(caret)
library(plyr)
library(Metrics)
train.raw <- train_2
test.raw <- test_2
#data preparation
col_names <- colnames(train.raw[,-c(1,2)])
# Determine data types in the data set
data_types <- sapply(col_names,function(x){class(train.raw[[x]])})
unique_data_types <- unique(data_types)
# Separate attributes by data type
DATA_ATTR_TYPES <- lapply(unique_data_types,function(x){ names(data_types[data_types == x])})
names(DATA_ATTR_TYPES) <- unique_data_types
# create folds for training
set.seed(13)
data_folds <- createFolds(train.raw$y, k=5)
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

###########################################################################################################################
#Level 0 Model Training
#train model on one data fold
trainOneFold <- function(this_fold,feature_set) {
  # get fold specific cv data
  cv.data <- list()
  cv.data$predictors <- feature_set$train$predictors[this_fold,]
  cv.data$ID <- feature_set$train$id[this_fold]
  cv.data$y <- feature_set$train$y[this_fold]
  # get training data for specific fold
  train.data <- list()
  train.data$predictors <- feature_set$train$predictors[-this_fold,]
  train.data$y <- feature_set$train$y[-this_fold]
  set.seed(825)
  fitted_mdl <- do.call(train,
                        c(list(x=train.data$predictors,y=train.data$y),
                          CARET.TRAIN.PARMS,
                          MODEL.SPECIFIC.PARMS,
                          CARET.TRAIN.OTHER.PARMS))
  yhat <- predict(fitted_mdl,newdata = cv.data$predictors,type = "raw")
  score <- rmse(cv.data$y,yhat)
  ans <- list(fitted_mdl=fitted_mdl,
              score=score,
              predictions=data.frame(ID=cv.data$ID,yhat=yhat,y=cv.data$y))
  return(ans)
}

# make prediction from a model fitted to one fold
makeOneFoldTestPrediction <- function(this_fold,feature_set) {
  fitted_mdl <- this_fold$fitted_mdl
  
  yhat <- predict(fitted_mdl,newdata = feature_set$test$predictors,type = "raw")
  
  return(yhat)
}