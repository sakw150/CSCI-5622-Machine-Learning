#!/usr/bin/env python
# coding: utf-8

# In[94]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import random as rd
from sklearn.metrics import confusion_matrix
from sklearn import metrics 
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
PWdata=pd.read_csv("~/Desktop/Undergrad/CSCI 5622 Machine Learning /Machine Learning Module 1/PortugalWineCleaned.csv")
# Let Bad=0, Average=1, Good=2
quality={'Bad':0, 'Average':1, 'Good':2}
PWdata['quality']=PWdata['quality'].map(quality)
PWdata.head()
# Split into Test and Training Data 
rd.seed(1)
Train,Test=train_test_split(PWdata, test_size=0.2)
print(Train.shape)
print(Test.shape)
Testlabel=Test['quality'] # Test Label 
TestPW=Test.drop(['quality'],axis=1) # Remove Label
Trainlabel=Train['quality'] # Train Label 
TrainPW=Train.drop(['quality'],axis=1) # Remove Label 
TrainPW.head()


# In[105]:


rd.seed(1)
DT=DecisionTreeClassifier()
DTPW=DT.fit(TrainPW,Trainlabel)
predict=DT.predict(TestPW)
print("Original", metrics.accuracy_score(Testlabel,predict))
features=['fixed.acidity','volatile.acidity','citric.acid','residual.sugar','chlorides','total.sulfur.dioxide','density','pH','sulphates','alcohol','quality']
tree.plot_tree(DTPW, feature_names=features)
plt.savefig("PythonTreeOriginal.pdf", format="pdf")

GINIDT = DecisionTreeClassifier(criterion='gini', max_depth=3)
GINIDT.fit(TrainPW, Trainlabel)
pred = GINIDT.predict(TestPW)
print('Gini', accuracy_score(Testlabel, pred))
tree.plot_tree(GINIDT, feature_names=features)
plt.savefig("PythonTreeGINIDepth3.pdf", format="pdf")
print(confusion_matrix(Testlabel,pred))


ENTDT= DecisionTreeClassifier(criterion='entropy', max_depth=27)
ENTDT.fit(TrainPW, Trainlabel)
predENT = ENTDT.predict(TestPW)
print('Entropy', accuracy_score(Testlabel, predENT))
tree.plot_tree(ENTDT, feature_names=features)
plt.savefig("PythonTreeEntropyDepth.pdf", format="pdf")
print(confusion_matrix(Testlabel,predENT))
       


# In[100]:


#Pruning Tree Using Max Depth 
i=30
for i in range(1,i):
    GINIDT = DecisionTreeClassifier(criterion='gini', max_depth=i)
    GINIDT.fit(TrainPW, Trainlabel)
    pred = GINIDT.predict(TestPW)
    print('Gini', accuracy_score(Testlabel, pred))
j=30
for j in range(1,j):
    ENTDT= DecisionTreeClassifier(criterion='entropy', max_depth=j)
    ENTDT.fit(TrainPW, Trainlabel)
    predENT = ENTDT.predict(TestPW)
    print('Entropy', accuracy_score(Testlabel, predENT))
    
    


# In[102]:


# Gaussian Naive Bayes Portugal Wine Dataset
from sklearn.naive_bayes import GaussianNB
GNB=GaussianNB()
GNBfit=GNB.fit(TrainPW,Trainlabel)
GNBpredict=GNBfit.predict(TestPW)
confusion_matrixGNB=metrics.confusion_matrix(Testlabel,GNBpredict)
print(confusion_matrixGNB)
Accuracy = metrics.accuracy_score(Testlabel,GNBpredict)
print(Accuracy)

# Multinomial Naive Bayes Portugal Wine Dataset
from sklearn.naive_bayes import MultinomialNB
MultNB=MultinomialNB()
MultNBfit=MultNB.fit(TrainPW,Trainlabel)
MultNBpredict=MultNBfit.predict(TestPW)
confusion_matrixMult=metrics.confusion_matrix(Testlabel,MultNBpredict)
print(confusion_matrixMult)
Accuracy = metrics.accuracy_score(Testlabel,MultNBpredict)
print(Accuracy)


# In[ ]:




