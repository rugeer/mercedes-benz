# set caret training parameters
set.seed(123)
CARET.TRAIN.PARMS <- list(method="gbm")   
CARET.TUNE.GRID <-  expand.grid(n.trees=c(40,50,60,70,80), 
                                interaction.depth=12, 
                                shrinkage=c(0.06,0.08,0.09,0.1,1.1) ,
                                n.minobsinnode=10)
MODEL.SPECIFIC.PARMS <- list(verbose=1) #NULL # Other model specific parameters
# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE                                                        
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
                                tuneGrid=gbm_mdl$bestTune,
                                metric="RMSE")
gbm_mdl_final <- do.call(train,
                   c(list(x=L0FeatureSet1$train$predictors,y=L0FeatureSet1$train$y),
                     CARET.TRAIN.PARMS,
                     MODEL.SPECIFIC.PARMS,
                     test.parameters))

test_gbm_yhat <- predict(gbm_mdl_final,newdata = L0FeatureSet1$test$predictors,type = "raw")
gbm_submission <- cbind(ID=L0FeatureSet1$test$id,y=test_gbm_yhat)
write.csv(gbm_submission,file="gbm_sumbission.csv",row.names=FALSE)
###########################################################################################################################