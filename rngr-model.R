# set caret training parameters
CARET.TRAIN.PARMS <- list(method="ranger")   

m <- as.integer(sqrt(ncol(L0FeatureSet1$train$predictors)))
CARET.TUNE.GRID <-  expand.grid(mtry=31.919)
                               
MODEL.SPECIFIC.PARMS <- list(verbose=1, num.trees=600) 

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(
  method = "cv",
  number = 10,
  verboseIter = TRUE                                                        
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
                        tuneGrid=expand.grid(rngr_mdl$bestTune),
                        metric="RMSE")

rngr_mdl_final <- do.call(train,
                         c(list(x=L0FeatureSet1$train$predictors,y=L0FeatureSet1$train$y),
                           CARET.TRAIN.PARMS,
                           MODEL.SPECIFIC.PARMS,
                           test.parameters))

test_rngr_yhat <- predict(rngr_mdl_final,newdata = L0FeatureSet1$test$predictors,type = "raw")
rngr_submission <- cbind(ID=L0FeatureSet1$test$id,y=test_rngr_yhat)
write.csv(rngr_submission,file="rngr_sumbission.csv",row.names=FALSE)
