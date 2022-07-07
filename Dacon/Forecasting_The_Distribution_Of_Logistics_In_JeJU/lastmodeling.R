rm(list =ls())
#######data open###
train_data <-read.csv("last.csv")
str(train_data)

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


z <-substr(train_data$id_1,1,2)
train_data$code_si <-z
train_data 


train_data$growth_index <- ifelse(is.na(train_data$growth_index), 7.10, train_data$growth_index )
train_data$work_index <- ifelse(is.na(train_data$work_index), 61.23, train_data$work_index)

train_data <- train_data %>% mutate(arr=ifelse(substr(id_1,1,2)==50,1,0))


train_data$id_1 <- format(train_data$id_1,scientific = F)
train_data$id_2 <- format(train_data$id_2,scientific = F)


z <-substr(train_data$id_1,1,10)
train_data$code_dong <-z
train_data 

y <-substr(train_data$id_1,11,16)
train_data$etc <- y
train_data$etc


y <-substr(train_data$id_2,11,16)






train_data$etc2 <-y
train_data$etc2
train_data$big_category
train_data$category <- as.character(factor(train_data$category, levels= unique(train_data$category)))
train_data$big_category <- as.character(factor(train_data$big_category, levels= unique(train_data$big_category)))
head(train_data$category)

x <- substr(train_data$id_1,1,10)
head(x)
train_data$code_dong <-x
train_data$code_dong

x <- substr(train_data$id_2,1,10)
train_data$code_dong2 <-x


train_data$number


head(train_data)

t <- substr(train_data$id_2,1,2)
train_data$code_si2 <- t

t <- substr(train_data$id_2,1,2)
train_data$code_si2 <- t

train_data$code_gu2 <- train_data$code_gu_id2


train_data <- train_data %>% mutate(region=(ifelse(code_si!=code_si2, 0, 1)))



#total_data <- total_data %>% filter(growth_index < 3000) # 이상치 제거

train_data$work_index <- (train_data$work_index-mean(train_data$work_index))/sd(train_data$work_index) # (x-mean(x))/sd(x) 사용

train_data$city_index <- (train_data$city_index-mean(train_data$city_index))/sd(train_data$city_index) # (x-mean(x))/sd(x) 사용

boxcox(train_data$growth_index+70.49~1) #box cox변환을 위한 lambda 값 확보 -> lambda = 0 -> log를 취해준다.

train_data$growth_index <- log(train_data$growth_index+70.49) ##

q1 <- quantile(train_data$growth_index)
q2 <- quantile(train_data$city_index)

train_data <- train_data %>% mutate(growth_o=ifelse(growth_index < q1[2], 1,
                                                    ifelse(growth_index > q1[4], 3, 2)))
train_data <- train_data %>% mutate(work_o=ifelse(work_index < q2[2], 1,
                                                  ifelse(work_index > q2[4], 3, 2)))
boxplot(train_data$growth_o)
boxplot(train_data$growth_index)

train_data <- train_data %>% dplyr::mutate(arr=ifelse(substr(id_1,1,2)==50,1,0))
train_data$work_o <- as.character(train_data$work_o)
train_data$growth_o <- as.character(train_data$work_o)
train_data$big_category_factor <- as.character(train_data$big_category_factor)
head(train_data)
table(train_data$number)
table(train_data$code_si)
str(train_data$code_si)

str(train_data)
train_data <-train_data %>%  dplyr::select(number,code_si, code_gu, code_dong,etc, code_si2,code_gu2,code_dong2,etc2, category, big_category_factor, population, work_o,growth_o,city_index,region)
str(train_data)

write.csv(train_data,"data.csv")
#boxcox(train_data$number~1)
#train_data$number <- ((train_data$number^(-2)-1)/(-2))
#train_data$number


plot(x=data_tr$code_dong, y = data_tr$number)


N <- nrow(train_data)
n.tr <- round(0.7*N)
tr <- sort(sample(1:N, n.tr, replace=F))
data_tr <- train_data[tr,]
data_te <- train_data[-tr,]

str(data_tr)
table(data_tr$number)
options(max.print=50) #조절해가면서 빼고 -> 빼고 ->빼고
model <- lm(number~code_dong+work_o+category+etc,data=data_tr,singular.ok = TRUE )
summary(model)
anova(model)
#install.packages("car")
library(car)
str(data_tr)
gvif(model)
chisq <- chisq.test(table(as.matrix(iris[,-5])))
chisq

###one hot encoding
data_tr$category<- one_hot(data_tr$category)
data_tr$work_o<- one_hot(data_tr$work_o)
data_tr$code_dong <- one_hot(data_tr$code_dong)
data_tr$region <- one_hot(data_tr$region)
data_tr$etc <- one_hot(data_tr$etc)


data_te$category<- one_hot(data_te$category)
data_te$region<- one_hot(data_te$region)
data_te$work_o<- one_hot(data_te$work_o)
data_te$code_dong <- one_hot(data_te$code_dong)
data_te$etc <- one_hot(data_te$etc)



data_tr <- data_tr %>%dplyr::select(number,category,work_o,region,code_dong,etc)
data_te <- data_te %>%dplyr::select(number,category,work_o,region,code_dong,etc)
data_tr
data_te
####cross validation

set.seed(200)
x_train_cv<-data.matrix(data_tr[,-1])
y_train_cv<-data.matrix(data_tr$number)


xgb_cv<-xgb.cv(data=x_train_cv,label=y_train_cv,
               nfold=5,nrounds=1000,early_stopping_rounds = 10,
               objective='reg:squaredlogerror',metrics='rmsle',
               params=list(eta=0.05,max_depth=8,subsample=0.8,colsample_bytree=0.8,stratified=T)) 
xgb_cv$evaluation_log


#####xgboost

train_x<-data.matrix(data_tr[,-1])
train_x
train_y<-data.matrix(data_tr$number)
xgb_train<-xgb.DMatrix(data=train_x,label=train_y)

train_x
watchlist <- list(train=xgb_train)

xgb_fit <- xgb.train(data = xgb_train, 
                     eta=0.05, 
                     max_depth=8, subsample=0.8,colsample_bytree=0.8,
                     nrounds= 157,# xgb_cv 결과, Best iteration
                     objective= "reg:squaredlogerror",
                     eval_metric= "rmsle",       
                     watchlist=watchlist,
                     print_every_n = 10
)


y_true <- data_te$number
xgb_test<-data.matrix(data_te[,-1])
xgb_pred_test<-predict(xgb_fit,xgb_test)

pred3 <- xgb_pred_test
rmse = caret::RMSE(data_te$number,pred3)
rmse
y_pred <-pred3
rmsle <- function(y_true, y_pred){
  sqrt(mean((log1p(y_true) - log1p(y_pred))^2))
  }
# create random ground truth values
y_true <- 2^runif(10, 0, 10)
cat('Actual Values: \n', y_true, '\n')

# create noisy predictions
y_pred <- rnorm(10, 15+y_true, 20)
cat('Predicted Values: \n', y_pred, '\n')

# calculate error
cat(sprintf('RMSLE: %0.5f', rmsle(y_true, y_pred)))


##### random forest

library(randomForest)
model <- randomForest(number~work_o+region+category+code_dong+etc, data= data_tr, importance= TRUE)
importance(model)

test_rf <- data_te[,-1]
test_rf
rf.pred <- predict(model,test_rf)
rmse = caret::RMSE(rf.pred,data_te$number)
rmse

y_pred <- rf.pred
y_true <- data_te$number
postResample(rf.pred,y_true)

rmsle <- function(y_true, y_pred){
  sqrt(mean((log1p(y_true) - log1p(y_pred))^2))
  }

y_true <- 2^runif(10, 0, 10)
cat('Actual Values: \n', y_true, '\n')

y_pred <- rnorm(10, 15+y_true, 20)
cat('Predicted Values: \n', y_pred, '\n')


cat(sprintf('RMSLE: %0.5f', rmsle(y_true, y_pred)))



####

#install.packages("devtools")
#devtools::install_github('catboost/catboost', subdir = 'catboost/R-package')

#install.packages("knitr")
library(catboost)
library(knitr)
data_tr <- data_tr %>%dplyr::select(number,category,work_o,region,code_dong,etc)
data_te <- data_te %>%dplyr::select(number,category,work_o,region,code_dong,etc)

data_tr$region <- as.factor(data_tr$region)
data_tr$category <- as.factor(data_tr$category)
data_tr$work_o <- as.factor(data_tr$work_o)
data_tr$code_dong <- as.factor(data_tr$code_dong)
data_tr$etc <- as.factor(data_tr$etc)

data_te$region <- as.factor(data_te$region)
data_te$category <- as.factor(data_te$category)
data_te$work_o <- as.factor(data_te$work_o)
data_te$code_dong <- as.factor(data_te$code_dong)
data_te$etc <- as.factor(data_te$etc)
str(data_tr)
set.seed(200)

y_train <- data_tr$number
x_train <- data_tr[,-1]
y_valid <- data_te$number
x_valid <- data_te[,-1]



train_pool <- catboost.load_pool(data = x_train, label = y_train)
test_pool <- catboost.load_pool(data = x_valid, label = y_valid)

params <- list(iterations=500,
               learning_rate=0.01,
               depth=10,
               loss_function='RMSE',
               eval_metric='RMSE',
               random_seed = 55,
               od_type='Iter',
               metric_period = 50,
               od_wait=20,
               use_best_model=TRUE)
ct.model <- catboost.train(learn_pool = train_pool,params = params)
ct_pred=catboost.predict(ct.model,test_pool)

rmse = caret::RMSE(ct_pred,data_te$number)
rmse

y_pred <- ct_pred
y_true <- data_te$number

rmsle <- function(y_true, y_pred){
  sqrt(mean((log1p(y_true) - log1p(y_pred))^2))
}

y_true <- 2^runif(10, 0, 10)
cat('Actual Values: \n', y_true, '\n')

y_pred <- rnorm(10, 15+y_true, 20)
cat('Predicted Values: \n', y_pred, '\n')


cat(sprintf('RMSLE: %0.5f', rmsle(y_true, y_pred)))

####lightgbm
library(Matrix)
#install.packages("lightgbm")
library(lightgbm)

data_tr$region <- as.character(data_tr$region)
data_tr$category <- as.character(data_tr$category)
data_tr$work_o <- as.character(data_tr$work_o)
data_tr$code_dong <- as.character(data_tr$code_dong)
data_tr$etc <- as.character(data_tr$etc)

data_te$region <- as.character(data_te$region)
data_te$category <- as.character(data_te$category)
data_te$work_o <- as.character(data_te$work_o)
data_te$code_dong <- as.character(data_te$code_dong)
data_te$etc <- as.character(data_te$etc)


data_tr <- data_tr %>%dplyr::select(number,category,work_o,region,code_dong,etc)
data_te <- data_te %>%dplyr::select(number,category,work_o,region,code_dong,etc)

x_test <- as.matrix(data_tr$number)
x_train <- as.matrix(data_tr[,-1])
y_test <- as.matrix(data_te$number)
y_train <- as.matrix(data_te[,-1])



params <- list(iterations=500,
               learning_rate=0.01,
               depth=10,
               objective = "regression",
               metric='mae',
               use_best_model=TRUE)

lgbid <- lightgbm(params = params,x_train,x_test,nrounds = 500,early_stopping_rounds = 10)

lgbid.pred <- predict(lgbid,data = y_train)
lgbid.pred

rmse = caret::RMSE(lgbid.pred,data_te$number)
rmse

y_pred <- lgbid.pred
y_true <- data_te$number

rmsle <- function(y_true, y_pred){
  sqrt(mean((log1p(y_true) - log1p(y_pred))^2))
}
?rnorm()
y_true <- 2^runif(10, 0, 10)
cat('Actual Values: \n', y_true, '\n')

y_pred <- rnorm(10, 15+y_true, 20)
cat('Predicted Values: \n', y_pred, '\n')


cat(sprintf('RMSLE: %0.5f', rmsle(y_true, y_pred)))
