rm(list= ls())
#-------------------------------------#
train <- read.csv("train.csv")
test <- read.csv("test.csv")
str(train)
str(test)

y_train <- train["target"]
y_train
x_train <- train[2:33]
x_train
x_test <- test[2:33]
x_test
str(x_train)
str(x_test)

####library####
library(ggplot2)
library(dplyr)
library(MASS)
library(nnet)
library(rpart)
library(caret)
library(Metrics)
library(MLmetrics)

#install.packages(c("tidyverse","dplyr","caret","Metrics","ggplot2","xgboost","data.table","mltools"))
library(tidyverse,warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE)
library(dplyr,warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE)
library(caret,warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE)
library(Metrics,warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE)
library(xgboost,warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE)
library(data.table,warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE) # fread()
library(mltools,warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE) # one_hot()
###### 


total_data <- rbind(x_train,x_test)
total_data
table(y_train)


#########crossvalidation xg
set.seed(200)
x_train_cv<-data.matrix(x_train)
y_train_cv<-data.matrix(y_train)


xgb_cv<-xgb.cv(data=x_train_cv,label=y_train_cv,
               nfold=5,nrounds=1000,early_stopping_rounds = 10,
               objective='multi:softprob',metrics='mlogloss',num_class=4,prediction = T,print_every_n = 10,
               params=list(eta=0.05,max_depth=8,subsample=0.8,colsample_bytree=0.8,stratified=T)) 
xgb_cv$evaluation_log


#####xgboost modeling

train_x<-data.matrix(x_train)
train_x
train_y<-data.matrix(y_train)
xgb_train<-xgb.DMatrix(data=train_x,label=train_y)

train_x
watchlist <- list(train=xgb_train)

xgb_fit <- xgb.train(data = xgb_train, 
                     eta=0.05, 
                     max_depth=8, subsample=0.8,colsample_bytree=0.8,
                     nrounds= 385,# xgb_cv 寃곌낵, Best iteration
                     objective= "multi:softprob",
                     eval_metric= "mlogloss",num_class=4,       
                     watchlist=watchlist,
                     print_every_n = 10
)


y_true <- y_train
xgb_test<-data.matrix(x_test)
xgb_pred_test<-predict(xgb_fit,xgb_test)
xgb_pred_test_total<-matrix(xgb_pred_test,nrow=4) %>% t() %>% data.frame()
xgb_pred_test_total
xgb_pred_test_total
xgb_pred_test_total$prediction <-apply(xgb_pred_test_total,1,function(x) colnames(xgb_pred_test_total)[which.max(x)])
xgb_pred_test_total$prediction <- ifelse(xgb_pred_test_total$prediction == "X1",0,
                                         ifelse(xgb_pred_test_total$prediction == "X2",1,
                                                ifelse(xgb_pred_test_total$prediction == "X3",2,3)))

xgb_pred_test_total$prediction



##########submission
submission <- read.csv("sample_submission.csv")
submission
submission$target <- xgb_pred_test_total$prediction
submission
write.csv(submission,"C:/Users/admin/Desktop/visual/sample.csv")

