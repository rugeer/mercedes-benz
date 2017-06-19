#xgboost model
# set caret training parameters
CARET.TRAIN.PARMS <- list(method="xgbTree")   

CARET.TUNE.GRID <-  expand.grid(nrounds=200, 
                                max_depth=10, 
                                eta=runif(2, 0.085,0.13), 
                                gamma=runif(1,0,0.1), 
                                colsample_bytree=c(1,0.4), 
                                min_child_weight=1,
                                subsample=1)

MODEL.SPECIFIC.PARMS <- list(verbose=1) 
# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(
  method = "cv",
  number = 2,
  verboseIter = TRUE,
  early_stopping_rounds=3
)

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                tuneGrid=CARET.TUNE.GRID,
                                metric="RMSE")
# final model fit
xgb_mdl <- do.call(train,
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
                        tuneGrid=xgb_mdl$bestTune,
                        metric="RMSE")

xgb_mdl_final <- do.call(train,
                         c(list(x=L0FeatureSet2$train$predictors,y=L0FeatureSet2$train$y),
                           CARET.TRAIN.PARMS,
                           MODEL.SPECIFIC.PARMS,
                           test.parameters))

test_xgb_yhat <- predict(xgb_mdl_final,newdata = L0FeatureSet2$test$predictors,type = "raw")
xgb_submission <- cbind(ID=L0FeatureSet2$test$id,y=test_xgb_yhat)
write.csv(xgb_submission,file="xgb_sumbission.csv",row.names=FALSE)
