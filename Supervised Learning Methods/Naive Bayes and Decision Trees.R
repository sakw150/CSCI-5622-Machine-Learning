############################################# Decision Trees and Naive Bayes 
library(rpart)   ## FOR Decision Trees
library(rpart.plot)
library(rattle)  ## FOR Decision Tree Vis
library(dplyr)
############################################# Portugal Wine Dataset 
PW=read.csv("~/Desktop/Undergrad/CSCI 5622 Machine Learning /Machine Learning Module 1/PortugalWineCleaned.csv", stringsAsFactors=TRUE)
DataSize=nrow(PW)
TrainingSet_Size=floor(DataSize*(3/4))
TestSet_Size = DataSize - TrainingSet_Size
# Splitting into Test and Training Dataset
set.seed(1)
Sampledata=sample(nrow(PW),TrainingSet_Size,replace=FALSE)
Training.Set=PW[Sampledata,]
Test.Set=PW[-Sampledata,]
# Checking Dimension of Testing and Training 
dim(Training.Set)
structure(Training.Set)
dim(Test.Set)
structure(Test.Set)
# Removing Labels from Test Set 
TestLabels=Test.Set$quality
Test.Set = Test.Set[ , -which(names(Test.Set) %in% c("quality"))]
TrainLabels=Training.Set$quality 

############################################# Portugal Wine Decision Tree 
str(Training.Set)
DTPW=rpart(Training.Set$quality~.,data=Training.Set,method="class")
summary(DTPW)
plotcp(DTPW) ## CP Plot 
prp(DTPW)
printcp(DTPW) # Prints CP values. Want lowest cross validation error or low xerror

# Decision Tree with CP Optimization 
library(gridExtra)
DTPWcp=rpart(Training.Set$quality~.,data=Training.Set,cp=0.014706,method="class")
prp(DTPWcp)
predictionDT=predict(DTPWcp,Test.Set,type="class")
png("CP Table.png",height=500, width=500)
DTCM=table(predictionDT,TestLabels)
grid.arrange(tableGrob(DTCM))
dev.off()
Error=1-sum(diag(DTCM))/sum(DTCM)
accuracy=1-Error
accuracy
rpart.plot(DTPWcp)
#Another Method of Pruning Tree
prunePW=prune(DTPW, cp=0.014706)
prp(prunePW)
fancyRpartPlot(prunePW)
############################################# Naive Bayes Using the Portugal Wine Dataset 
library(dplyr)
library(ggplot2)
library(tm)
library(e1071)
library(caret)
# Gaussian Naive Bayes for Portugal Wine Dataset
naivebayes=naiveBayes(Training.Set,TrainLabels, laplace=1) 
predictionWD=predict(naivebayes,Test.Set)
(CM=table(predictionWD,TestLabels))
png("R Naive Bayes.png",height=500, width=500)
grid.arrange(tableGrob(CM))
dev.off()
Error=1-sum(diag(CM))/sum(CM)
accuracy=1-Error
print(accuracy)
#############################################  Naive Bayes Using the Wine Enthusiast Dataset
library(quanteda.textmodels)
library(quanteda)
library(tm)
library(caTools)
WE=read.csv("~/Desktop/Undergrad/CSCI 5622 Machine Learning /Machine Learning Module 1/WineReview(WineEnthusiast)Cleaned.csv", stringsAsFactors=TRUE)
for (i in 1:nrow(WE)){
  if (WE$points[i]<= 86){
    WE$points[i]="Bad"
  }
  else if(WE$points[i]>=88){
    WE$points[i]="Good"
  }
  else {
    WE$points[i]="Average"
  }
}
head(WE)
summary(WE)
WE = subset(WE, select = c('country','points','price','variety','description'))
DataSizeWE=nrow(WE)
TrainingSet_SizeWE=floor(DataSizeWE*(3/4))
TestSet_SizeWE = DataSizeWE - TrainingSet_SizeWE
SampledataWE=sample(nrow(WE),TrainingSet_SizeWE,replace=FALSE)
Training.SetWE=as.data.frame(WE[SampledataWE,])
Test.SetWE=as.data.frame(WE[-SampledataWE,])
# Checking Dimension of Testing and Training 
dim(Training.SetWE)
dim(Test.SetWE)
# Remove Labels 
TestLabelsWE=Test.SetWE$points
Test.SetWE = Test.SetWE[ , -which(names(Test.SetWE) %in% c("points"))]
TrainLabelsWE=Training.SetWE$points
############################################# Text Cleaning for Naive Bayes
# Corpus and Corpus Cleaning
test_corpus=Corpus(VectorSource(Test.SetWE$description))
train_corpus=Corpus(VectorSource(Training.SetWE$description))
# Lower
test_corp=tm_map(test_corpus,tolower)
train_corp=tm_map(train_corpus,tolower)
# Remove Number 
test_corp=tm_map(test_corp,removeNumbers)
train_corp=tm_map(train_corp,removeNumbers)
# Remove Stop Words 
test_corp=tm_map(test_corp,removeWords,stopwords("english"))
train_corp=tm_map(train_corp,removeWords,stopwords("english"))
# Remove Punctuation
train_corp=tm_map(train_corp,removePunctuation)
test_corp=tm_map(test_corp,removePunctuation)
# Remove White Space 
test_corp=tm_map(test_corp,stripWhitespace)
train_corp=tm_map(train_corp,stripWhitespace)
inspect(train_corp[2])
train_corp = names(unlist(train_corp))
test_corp=names(unlist(test_corp))
textnb=textmodel_nb(dfm(tokens(train_corp)),TrainLabelsWE, smooth=1, distribution="multinomial")
NBtextpredict=predict(textnb,newdata=dfm(tokens(test_corp[-length(test_corp)])),force=TRUE)
CMWE=table(TestLabelsWE, NBtextpredict)
png("WE Naive Bayes.png",height=500, width=500)
grid.arrange(tableGrob(CMWE))
dev.off()
Error=1-sum(diag(CMWE))/sum(CMWE)
accuracy=1-Error
print(accuracy)

