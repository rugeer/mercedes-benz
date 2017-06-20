#xgboost model
# set caret training parameters
CARET.TRAIN.PARMS <- list(method="svmLinear")   

CARET.TUNE.GRID <-  expand.grid(C=c(0.25,0.26))

MODEL.SPECIFIC.PARMS <- list(verbose=1) 
# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE
)

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                tuneGrid=CARET.TUNE.GRID,
                                metric="RMSE")
# final model fit
svm_mdl <- do.call(train,
                   c(list(x=L0FeatureSet2$train$predictors,y=L0FeatureSet2$train$y),
                     CARET.TRAIN.PARMS,
                     MODEL.SPECIFIC.PARMS,
                     CARET.TRAIN.OTHER.PARMS))
###########################################################################################################################
#shouldnt need this
test.traincontrol <- trainControl(method="none",
                                  verboseIter=FALSE,
                                  classProbs=FALSE)

test.parameters <- list(trControl=test.traincontrol,
                        tuneGrid=svm_mdl$bestTune,
                        metric="RMSE")

svm_mdl_final <- do.call(train,
                         c(list(x=L0FeatureSet2$train$predictors,y=L0FeatureSet2$train$y),
                           CARET.TRAIN.PARMS,
                           MODEL.SPECIFIC.PARMS,
                           test.parameters))

test_svm_yhat <- predict(svm_mdl_final,newdata = L0FeatureSet2$test$predictors,type = "raw")
svm_submission <- cbind(ID=L0FeatureSet2$test$id,y=test_svm_yhat)
write.csv(svm_submission,file="svm_sumbission.csv",row.names=FALSE)
