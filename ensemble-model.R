#ensemble model
ensemble_submission <- cbind(ID=L0FeatureSet2$test$id,
                             y=((test_xgb_yhat+test_gbm_yhat+2*test_rngr_yhat)/4))
write.csv(ensemble_submission,file="ensemble_sumbission.csv",row.names=FALSE)
