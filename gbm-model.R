# set caret training parameters
set.seed(123)
CARET.TRAIN.PARMS <- list(method="gbm")   
CARET.TUNE.GRID <-  expand.grid(#n.trees=c(100,110,90),
                                n.trees=90,
                                #interaction.depth=c(10,8,12), 
                                interaction.depth=8,
                                #shrinkage=c(0.1,0.09,0.12),
                                shrinkage=0.09,
                                #n.minobsinnode=c(10,8,12)
                                n.minobsinnode=8)
MODEL.SPECIFIC.PARMS <- list(verbose=1) #NULL # Other model specific parameters
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
gbm_mdl <- do.call(train,
                   c(list(x=L0FeatureSet1$train$predictors,y=L0FeatureSet1$train$y),
                     CARET.TRAIN.PARMS,
                     MODEL.SPECIFIC.PARMS,
                     CARET.TRAIN.OTHER.PARMS))
###########################################################################################################################
#shouldnt need this either
test.traincontrol <- trainControl(method="none",
                                 verboseIter=FALSE,
                                 classProbs=FALSE)

test.parameters <- list(trControl=test.traincontrol,
                        #normaly tuneGrid=gbm_mdl$bestTune
                        #already found optimal parameteres
                                tuneGrid=CARET.TUNE.GRID,
                                metric="RMSE")
gbm_mdl_final <- do.call(train,
                   c(list(x=L0FeatureSet1$train$predictors,y=L0FeatureSet1$train$y),
                     CARET.TRAIN.PARMS,
                     MODEL.SPECIFIC.PARMS,
                     test.parameters))

###########################################################################################################################
#predictions for level1 model
gbm_train_pred <- gbm_mdl$pred$pred[order(gbm_mdl$pred$rowIndex)]
#Predicting probabilities for the test data
gbm_test_pred <- predict(gbm_mdl, newdata = L0FeatureSet1$test$predictors,type = "raw")
###########################################################################################################################
#predictions for submission
test_gbm_yhat <- predict(gbm_mdl_final,newdata = L0FeatureSet1$test$predictors,type = "raw")
gbm_submission <- cbind(ID=L0FeatureSet1$test$id,y=test_gbm_yhat)
write.csv(gbm_submission,file="gbm_sumbission.csv",row.names=FALSE)
###########################################################################################################################


