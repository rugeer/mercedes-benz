#xgboost model
set.seed(123)
# set caret training parameters
CARET.TRAIN.PARMS <- list(method="xgbTree")   

CARET.TUNE.GRID <-  expand.grid(#nrounds=sample(c(20:100),20), 
                                max_depth=4, 
                                nrounds=62,
                                #eta=runif(10, 0.08,0.12), 
                                eta=0.08869209,
                                gamma=7.55123, 
                                #colsample_bytree=runif(10,0.2,0.8)
                                colsample_bytree=0.6913006, 
                                #min_child_weight=sample(c(1:10),10),
                                min_child_weight=7,
                                subsample=0.4)

MODEL.SPECIFIC.PARMS <- list(verbose=1) 
# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(
  method = "cv",
  number = 10,
  verboseIter = TRUE,
  savePredictions = 'final' # To save out of fold predictions for best parameter combinantions
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
                        #optimal parameters already chosen though
                        #tuneGrid=CARET.TUNE.GRID ,
                        metric="RMSE")

xgb_mdl_final <- do.call(train,
                         c(list(x=L0FeatureSet2$train$predictors,y=L0FeatureSet2$train$y),
                           CARET.TRAIN.PARMS,
                           MODEL.SPECIFIC.PARMS,
                           test.parameters))
###########################################################################################################################
#predictions for level1 model
xgb_train_pred <- xgb_mdl$pred$pred[order(xgb_mdl$pred$rowIndex)]
#Predicting probabilities for the test data
xgb_test_pred <- predict(xgb_mdl, newdata = L0FeatureSet2$test$predictors,type = "raw")
###########################################################################################################################
#predictions for submission
test_xgb_yhat <- predict(xgb_mdl_final,newdata = L0FeatureSet2$test$predictors,type = "raw")
xgb_submission <- cbind(ID=L0FeatureSet2$test$id,y=test_xgb_yhat)
write.csv(xgb_submission,file="xgb_sumbission.csv",row.names=FALSE)
