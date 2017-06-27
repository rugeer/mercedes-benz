# set caret training parameters
set.seed(123)
CARET.TRAIN.PARMS <- list(method="ranger")   

m <- as.integer(sqrt(ncol(L0FeatureSet1$train$predictors)))
CARET.TUNE.GRID <-  expand.grid(mtry=31.919)
                               
MODEL.SPECIFIC.PARMS <- list(verbose=1, num.trees=600) 

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
rngr_mdl <- do.call(train,
                    c(list(x=L0FeatureSet1$train$predictors,y=L0FeatureSet1$train$y),
                      CARET.TRAIN.PARMS,
                      MODEL.SPECIFIC.PARMS,
                      CARET.TRAIN.OTHER.PARMS))
###########################################################################################################################
#might not need these, since caret automatically fits to the whole data set
test.traincontrol <- trainControl(method="none",
                                  verboseIter=FALSE,
                                  classProbs=FALSE)

test.parameters <- list(trControl=test.traincontrol,
                        #normally tuneGrid=expand.grid(rngr_mdl$bestTune)
                        #optimal parameters already chosen though
                        tuneGrid=CARET.TUNE.GRID ,
                        metric="RMSE")

rngr_mdl_final <- do.call(train,
                         c(list(x=L0FeatureSet1$train$predictors,y=L0FeatureSet1$train$y),
                           CARET.TRAIN.PARMS,
                           MODEL.SPECIFIC.PARMS,
                           test.parameters))

###########################################################################################################################
#Predicting the out of fold prediction probabilities for training data
rngr_train_pred <- rngr_mdl$pred$pred[order(rngr_mdl$pred$rowIndex)]
#Predicting probabilities for the test data
rngr_test_pred <- predict(rngr_mdl, newdata = L0FeatureSet1$test$predictors,type = "raw")
###########################################################################################################################
#predictions for submission
test_rngr_yhat <- predict(rngr_mdl_final,newdata = L0FeatureSet1$test$predictors,type = "raw")
rngr_submission <- cbind(ID=L0FeatureSet1$test$id,y=test_rngr_yhat)
write.csv(rngr_submission,file="rngr_sumbission.csv",row.names=FALSE)
