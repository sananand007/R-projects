# Practical ML QUiz-3

library(AppliedPredictiveModeling)
data(segmentationOriginal)
library(caret)

#str(segmentationOriginal)
#head(segmentationOriginal$Case)

#Subset The data to a training set and testing set Based on the case variable 

training.set<-segmentationOriginal[which(segmentationOriginal$Case=='Train'),]
testing.set<-segmentationOriginal[which(segmentationOriginal$Case=='Test'),]


#Set the seed to 125 and fit a CART model with the rpart method using all predictor variables and default caret settings.
library(rpart)
library(ggplot2)
library(rpart.plot)
library(rpart.utils)
set.seed(125)
modFit.cart<-train(Class~., data=training.set, method="rpart")
rpart.plot(modFit.cart$finalModel)

#Olive Oil Data
library(pgmm)
data(olive)
olive = olive[,-1]

head(olive)

#Fit a classification tree where Area is the outcome variable. Then predict the value of area for the following data frame using the tree command with all defaults

require(tree)
fit.tree1<-tree(Area ~ .,data = olive)
summary(fit.tree1)

newdata<-as.data.frame(t(colMeans(olive)))

plot(fit.tree1)
text(fit.tree1, all=TRUE)

newdata

predict(fit.tree1, newdata)

# South Africa heart desease data set
library(ElemStatLearn)
data(SAheart)
set.seed(8464)
train=sample(1:dim(SAheart)[1], size=dim(SAheart)[1]/2, replace = F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]

summary(train)

set.seed(1234)

#fit a logistic regression model (method="glm", be sure to specify family="binomial") with Coronary Heart Disease (chd) 
#as the outcome and age at onset, current alcohol consumption, obesity levels, cumulative tabacco, type-A behavior, and low density lipoprotein cholesterol as predictors.

modelfit.logit<-train(chd~age+alcohol+obesity+tobacco+typea+ldl,model="glm",data=trainSA,family="binomial")
missClass = function(values,prediction){sum(((prediction > 0.5)*1) != values)/length(values)}
missClass(trainSA$chd, predict(modelfit.logit,trainSA))
missClass(trainSA$chd, predict(modelfit.logit,testSA))


# Vowel

library(ElemStatLearn)
data(vowel.train)
data(vowel.test)

vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)


str(vowel.train)

set.seed(33833)

library(randomForest)
library(caret)
modelfit.randf<-randomForest(y~., data=vowel.train,method="rf",prox=TRUE)
vowelvarImptrain<-varImp(modelfit.randf, scale = FALSE)



