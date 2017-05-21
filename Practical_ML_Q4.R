# Practical ML Q-4

## Question-1
library(ElemStatLearn)

data(vowel.train)
data(vowel.test)

vowel.train$y<-as.factor(vowel.train$y)
vowel.test$y<-as.factor(vowel.test$y)

set.seed(33833)
library(randomForest)
library(caret)
model.rf<-train(y~., data = vowel.train, method = "rf", prox=TRUE)
predrf<-predict(model.rf, vowel.test)
cmrf<-confusionMatrix(predrf, vowel.test$y)
cmrf

set.seed(33833)
model.gbm<-train(y~., data = vowel.train, method = "gbm", verbose=FALSE)
predgbm<-predict(model.gbm, vowel.test)
cmgbm<-confusionMatrix(predgbm, vowel.test$y)
cmgbm

# Combined accuracy model or what is the acuuracy with agreement
datacombined<-data.frame(predrf, predgbm, y=vowel.test$y)
fitcomb<-train(y ~.,data=datacombined)
predfitcomb<-predict(fitcomb, vowel.test)
cmcomb<-confusionMatrix(predfitcomb, vowel.test$y)
cmcomb

L3 <- LETTERS[1:3]
fac <- sample(L3, 10, replace = TRUE)
(d <- data.frame(x = 1, y = 1:10, fac = fac))

sink("Diverted Output.txt")
newdata<-data.frame("index"<-numeric())
for (i in 1:nrow(datacombined)) {
  print(datacombined[i,])
  newdata[(nrow(newdata)+1),]<-datacombined[i,1]
  print(newdata)
}
sink()

# We can also find the accuracy using a function and doing it manually
findata<-data.frame("predrf"=numeric(), "predgbm"=numeric(), "y"=numeric())
for (i in 1:nrow(datacombined)){
  #print(datacombined[i,1])
  datacombined[i,2]
  if(identical(datacombined[i,1], datacombined[i,2])){
    findata[(nrow(findata)+1),] <- datacombined[i,]
  }
}
head(findata)

# Finding the Count to get the accuracy
count=0
for (i in 1:nrow(findata)){
  if (identical(findata[i,2], findata[i,3])) {
    count<-count+1
  }
}
count

accuracy<-(count/nrow(findata))*100
accuracy

sink("Diverted Output.txt")
## Question-2
library(caret)
library(gbm)
set.seed(3433)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)

adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

# Three models
set.seed(62433)
modfit.rf<-train(diagnosis~., method = "rf", data=training, prox=TRUE)
predict.rf<-predict(modfit.rf, testing)
cmrf<-confusionMatrix(predic.rf,testing$diagnosis)
cmrf

set.seed(62433)
modfit.gbm<-train(diagnosis~., method = "gbm", data=training, verbose=FALSE)
predict.gbm<-predict(modfit.gbm, testing)
cmgbm<-confusionMatrix(predict.gbm,testing$diagnosis)
cmgbm

## Question-3
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain<-createDataPartition(concrete$CompressiveStrength, p=3/4)[[1]]
training=concrete[inTrain,]
testing=concrete[-inTrain,]
set.seed(233)
fit<-train(CompressiveStrength~.,data=training,method="lasso")
predict.lasso<-predict(fit,testing)
plot.enet(fit$finalModel,xvar = "penalty", use.color = TRUE)






set.seed(62433)
modfit.lda<-train(diagnosis~., method = "lda", data=training)
predict.lda<-predict(modfit.lda, testing)
cmlda<-confusionMatrix(predict.lda,testing$diagnosis)
cmlda

set.seed(62433)
# Stacking all the predictors together using randomforests
combined.pred<-data.frame(predict.rf, predict.gbm, predict.lda, diagnosis=testing$diagnosis)
modfit.rfnew<-train(diagnosis~., method="rf", data=combined.pred, prox=TRUE)
predict.combined<-predict(modfit.rfnew,testing)
cmstacked<-confusionMatrix(predict.combined,testing$diagnosis)
cmstacked
sink()


