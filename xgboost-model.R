###########################################################################################################################
#xgboost model
# set caret training parameters
CARET.TRAIN.PARMS <- list(method="xgbTree")   
CARET.TUNE.GRID <-  expand.grid(nrounds=800, 
                                max_depth=10, 
                                eta=0.03, 
                                gamma=0.1, 
                                colsample_bytree=0.4, 
                                min_child_weight=1,
                                subsample=1)
MODEL.SPECIFIC.PARMS <- list(verbose=0) #NULL # Other model specific parameters
# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method="none",
                                 verboseIter=FALSE,
                                 classProbs=FALSE)
CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                tuneGrid=CARET.TUNE.GRID,
                                metric="RMSE")
# generate Level 1 features
xgb_set <- llply(data_folds,trainOneFold,L0FeatureSet2)
# final model fit
xgb_mdl <- do.call(train,
                   c(list(x=L0FeatureSet2$train$predictors,y=L0FeatureSet2$train$y),
                     CARET.TRAIN.PARMS,
                     MODEL.SPECIFIC.PARMS,
                     CARET.TRAIN.OTHER.PARMS))

# CV Error Estimate
cv_y <- do.call(c,lapply(xgb_set,function(x){x$predictions$y}))
cv_yhat <- do.call(c,lapply(xgb_set,function(x){x$predictions$yhat}))
rmse(cv_y,cv_yhat)
cat("Average CV rmse:",mean(do.call(c,lapply(xgb_set,function(x){x$score}))))
test_xgb_yhat <- predict(xgb_mdl,newdata = L0FeatureSet2$test$predictors,type = "raw")
xgb_submission <- cbind(ID=L0FeatureSet2$test$id,y=test_xgb_yhat)

write.csv(xgb_submission,file="xgb_sumbission.csv",row.names=FALSE)
