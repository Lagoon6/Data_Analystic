rm(list = ls())
data(iris)
library(nnet)
library(caret)
idx <- createDataPartition(iris$Species, p= 0.7, list = F)
train_data <- iris[idx,]
test_data <- iris[-idx,]

#logistic regressor model  #y level = 3
model <- multinom(Species~., data = train_data)
model
summary(model)
fitted(model)
pred <- predict(model,test_data)
xtabs(~pred+test_data$Species)
confusionMatrix(pred, test_data$Species)


multi.result <-multinom(Species~., data= train_data)
# over fitting
multi.fitting_rate <- predict(multi.result, train_data)
# 예측률
multi.pred <- predict(multi.result, test_data)
# 훈련도(fitting rate)
table(multi.fitting_rate, train_data$Species)
# 예측률
table(multi.pred, test_data$Species)
confusionMatrix(multi.pred, test_data$Species)


#decision tree model
library(rpart)
model2 <- rpart(Species~., data =train_data)
pred2 <- predict(model2, test_data, type = "class")
confusionMatrix(pred2, test_data$Species)



#randomforest model
#install.packages("randomForest")
library(randomForest)
rf.result <- randomForest(Species~., data = train_data)
rf.result
#훈련도 fitting rate
rf.fr <- predict(rf.result, train_data, type ="response")
table(rf.fr, train_data$Species)
#예측률
rf.pred <- predict(rf.result, test_data, type ="response")
table(rf.pred, test_data$Species)

confusionMatrix(rf.pred, test_data$Species)
#-----중요도: RandomForest, XGBoost, Bruta, stepAIC, 가설검정
rf.importance <- randomForest(Species~., data= iris, importance = T)
importance(rf.importance)
