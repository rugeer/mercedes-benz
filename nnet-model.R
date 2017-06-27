# putting all the models together using neural networks
# create predictions for each model for the train set

gbm_yhat <- predict(gbm_mdl_final,newdata = L0FeatureSet1$train$predictors,type = "raw")
xgb_yhat <- predict(xgb_mdl_final,newdata = L0FeatureSet2$train$predictors,type = "raw")
rngr_yhat <- predict(rngr_mdl_final,newdata = L0FeatureSet1$train$predictors,type = "raw")

# put the data together
L1FeatureSet <- list()
L1FeatureSet$ID <- L0FeatureSet1$train$id
L1FeatureSet$y <- L0FeatureSet1$train$y
predictors <- data.frame(gbm_yhat,xgb_yhat,rngr_yhat)
predictors_rank <- t(apply(predictors,1,rank))
colnames(predictors_rank) <- paste0("rank_",names(predictors))
L1FeatureSet$predictors <- predictors #cbind(predictors,predictors_rank)

L1FeatureSet$test$id <- gbm_submission[,"ID"]
L1FeatureSet$test$predictors <- data.frame(gbm_yhat=test_gbm_yhat,
                                           xgb_yhat=test_xgb_yhat,
                                           rngr_yhat=test_rngr_yhat)

# neural net model
# set caret training parameters
CARET.TRAIN.PARMS <- list(method="nnet") 
best.tune <- l1_nnet_mdl$bestTune
CARET.TUNE.GRID <-  expand.grid(size=5,decay=0.04943298)  # NULL provides model specific default tuning parameters

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(
  method = "cv",
  number = 10,
  verboseIter = TRUE
)
  
CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                tuneGrid=CARET.TUNE.GRID,
                                metric="RMSE", maxit=200,subsample=0.6)

MODEL.SPECIFIC.PARMS <- list(verbose=TRUE,linout=TRUE,trace=FALSE) #NULL # Other model specific parameters


# train the model
set.seed(825)
l1_nnet_mdl <- do.call(train,c(list(x=L1FeatureSet$predictors,y=L1FeatureSet$y),
                               CARET.TRAIN.PARMS,
                               MODEL.SPECIFIC.PARMS,
                               CARET.TRAIN.OTHER.PARMS))

###########################################################################################################################
#might not need these, since caret automatically fits to the whole data set
test.traincontrol <- trainControl(method="none",
                                  verboseIter=FALSE,
                                  classProbs=FALSE)

test.parameters <- list(trControl=test.traincontrol,
                        tuneGrid=l1_nnet_mdl$bestTune,
                        metric="RMSE")

l1_nnet_mdl_final <- do.call(train,c(list(x=L1FeatureSet$predictors,y=L1FeatureSet$y),
                                  CARET.TRAIN.PARMS,
                                  MODEL.SPECIFIC.PARMS,
                                  test.parameters))

test_l1_nnet_yhat <- predict(l1_nnet_mdl_final,newdata = L1FeatureSet$test$predictors,type = "raw")
nnet_submission <- cbind(ID=L1FeatureSet$test$id,y=test_l1_nnet_yhat)
colnames(nnet_submission)[2] <- 'y'
write.csv(nnet_submission,file="nnet_submission.csv",row.names=FALSE)
