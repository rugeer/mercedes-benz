#ensemble model1
ensemble_submission <- cbind(ID=L0FeatureSet2$test$id,
                             y=((test_xgb_yhat+test_gbm_yhat+2*test_rngr_yhat)/4))
write.csv(ensemble_submission,file="ensemble_sumbission.csv",row.names=FALSE)
#ensemble model2
train_data <- cbind(xgb=xgb_train_pred,gbm=gbm_train_pred,
                    rngr=rngr_train_pred)
y_train <-  L0FeatureSet1$train$y
test_data <- cbind(xgb=xgb_test_pred,gbm=gbm_test_pred,
                   rngr=rngr_test_pred)
#gbm model
# set caret training parameters
set.seed(123)
CARET.TRAIN.PARMS <- list(method="ranger")   

m <- as.integer(sqrt(ncol(train_data)))
CARET.TUNE.GRID <-  expand.grid(mtry=runif(10,1,2)*m)

MODEL.SPECIFIC.PARMS <- list(verbose=1, num.trees=600) 

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(
  method = "cv",
  number = 10,
  verboseIter = TRUE
  #savePredictions = 'final' # To save out of fold predictions for best parameter combinantions
)

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                tuneGrid=CARET.TUNE.GRID,
                                metric="RMSE")

rngr_mdl_ensemble <- do.call(train,
                    c(list(x=train_data,y=y_train),
                      CARET.TRAIN.PARMS,
                      MODEL.SPECIFIC.PARMS,
                      CARET.TRAIN.OTHER.PARMS))
#predictions for submission
test_gbm_ensemble_yhat <- predict(rngr_mdl_ensemble,newdata = test_data,type = "raw")
test_gbm_ensemble_submission <- cbind(ID=L0FeatureSet2$test$id,y=test_gbm_ensemble_yhat)
write.csv(test_gbm_ensemble_submission ,file="test_lm_ensemble_submission.csv",row.names=FALSE)

# set caret training parameters
set.seed(123)
CARET.TRAIN.PARMS <- list(method="nnet")
CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                metric="Rsquared")
MODEL.SPECIFIC.PARMS <- list(verbose=TRUE,linout=TRUE,trace=FALSE) #NULL # Other model specific parameters

mdl_ensemble <- do.call(train,
                             c(list(x=train_data,y=y_train),
                               CARET.TRAIN.PARMS,
                               CARET.TRAIN.OTHER.PARMS,
                               MODEL.SPECIFIC.PARMS))
mdl_ensemble$results


test_rngr_ensemble_yhat <- predict(mdl_ensemble,newdata = test_data,type = "raw")
test_rngr_ensemble_submission <- cbind(ID=L0FeatureSet2$test$id,y=test_rngr_ensemble_yhat)
write.csv(test_rngr_ensemble_submission ,file="test_nnet2_ensemble_submission.csv",row.names=FALSE)

