#!/usr/bin/env python
# coding: utf-8

# In[47]:


###################### Portugal Wine Dataset
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import random as rd
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
PWdata=pd.read_csv("~/Desktop/Undergrad/CSCI 5622 Machine Learning /Machine Learning Module 1/PWDiscretized.csv")
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


# In[48]:


# Linear Function 
#linSVM=SVC(C=20, kernel='linear',verbose=True, gamma="auto")
#linSVM.fit(TrainPW,Trainlabel)
#linpredict=linSVM.predict(TestPW)
#linconfusion=confusion_matrix(Testlabel,linpredict)
#print(linconfusion)
#Accuracy = metrics.accuracy_score(Testlabel,linpredict)
#print(Accuracy)

# Radial Function 
radialSVM=SVC(C=20, kernel='rbf',verbose=True, gamma="auto")
radialSVM.fit(TrainPW,Trainlabel)
radialpredict=radialSVM.predict(TestPW)
radialconfusion=confusion_matrix(Testlabel,radialpredict)
print(radialconfusion)
Accuracy = metrics.accuracy_score(Testlabel,radialpredict)
print(Accuracy)

# Sigmoidal Function
sigSVM=SVC(C=20, kernel='sigmoid',verbose=True, gamma="auto")
sigSVM.fit(TrainPW,Trainlabel)
sigpredict=sigSVM.predict(TestPW)
sigconfusion=confusion_matrix(Testlabel,sigpredict)
print(sigconfusion)
Accuracy = metrics.accuracy_score(Testlabel,sigpredict)
print(Accuracy)


# In[43]:


###################### Chemical Analysis Dataset
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import random as rd
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
CAdata=pd.read_csv("~/Desktop/Undergrad/CSCI 5622 Machine Learning /Machine Learning Module 1/CADiscretized.csv") 
# Split into Test and Training Data 
rd.seed(1)
TrainCA,TestCA=train_test_split(CAdata, test_size=0.2)
print(TrainCA.shape)
print(TestCA.shape)
TestlabelCA=TestCA['Alcohol'] # Test Label 
TestCA=TestCA.drop(['Alcohol'],axis=1) # Remove Label
TrainlabelCA=TrainCA['Alcohol'] # Train Label 
TrainCA=TrainCA.drop(['Alcohol'],axis=1) # Remove Label 
TrainCA.head()
CAdata.head()


# In[46]:


# Linear Function 
linearSVM=SVC(C=10, kernel='linear',verbose=True, gamma="auto")
linearSVM.fit(TrainCA,TrainlabelCA)
linearpredictCA=linearSVM.predict(TestCA)
linearconfusionCA=confusion_matrix(TestlabelCA,linearpredictCA)
print(linearconfusionCA)
Accuracy = metrics.accuracy_score(TestlabelCA,linearpredictCA)
print(Accuracy)


# In[44]:


# Radial Function 
radialSVM=SVC(C=10, kernel='rbf',verbose=True, gamma="auto")
radialSVM.fit(TrainCA,TrainlabelCA)
radialpredictCA=radialSVM.predict(TestCA)
radialconfusionCA=confusion_matrix(TestlabelCA,radialpredictCA)
print(radialconfusionCA)
Accuracy = metrics.accuracy_score(TestlabelCA,radialpredictCA)
print(Accuracy)


# In[45]:


# Sigmoidal Function
sigSVM=SVC(C=20, kernel='sigmoid',verbose=True, gamma="auto")
sigSVM.fit(TrainCA,TrainlabelCA)
sigpredictCA=sigSVM.predict(TestCA)
sigconfusionCA=confusion_matrix(TestlabelCA,sigpredictCA)
print(sigconfusionCA)
Accuracy = metrics.accuracy_score(TestlabelCA,sigpredictCA)
print(Accuracy)

