library(tokenizers)
library(stopwords)
library(tidyverse)
library(arules)
library(arulesViz)
library(ggplot2)
library(stringr)
library(syuzhet)
library(ursa)
library(Matrix)
library(tcltk)
Tdata=read.csv("~/Desktop/Machine Learning Module 1/NEWSAPICleaned2.csv")
head(Tdata)
nrow(Tdata)
Tdata=data.frame(Tdata)
# Skip This Portion Since We Already Converted To Transactional Data 
sink("~/Desktop/Machine Learning Module 1/TransactionFile.csv")
for(i in 1:nrow(Tdata)){
  Tokens=tokenizers::tokenize_words(Tdata$description[i],stopwords=stopwords::stopwords("en"),
                        lowercase=TRUE,strip_numeric=TRUE,strip_punct=TRUE,simplify=TRUE)
  cat(unlist(str_squish(Tokens)),"\n",sep=",")
}
sink()
# ARM
Transactions=read.transactions("~/Desktop/Machine Learning Module 1/TransactionFile.csv",
                               rm.duplicates = FALSE,
                               format="basket",
                               sep=",")
inspect(Transactions)
summary(Transactions)
Transrules=arules::apriori(Transactions,
                           parameter=list(support=0.01,confidence=0.7))
inspect(Transrules)
sortedrules=sort(Transrules,by="support",decreasing=TRUE)
inspect(sortedrules)
plot(sortedrules)
library(tcltk)
plot(sortedrules, method="graph", engine="interactive",shading="support")
plot(sortedrules, method="graph", engine="htmlwidget")
itemFrequencyPlot(Transactions, topN=20, type="absolute")
