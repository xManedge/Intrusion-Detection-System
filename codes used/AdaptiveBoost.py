# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 20:39:04 2019

@author: Manoj V
"""

#importing the required modules
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
#Loading the dataset

#url = "20071006.xlsx"
#dataset = pd.read_excel(url)
#
#dataset=dataset.drop(['Source_IP_Address','Destination_IP_Address','Start_Time'], axis=1)

train, test = train_test_split(dataset,test_size=0.4)


#Pre-processing
labelEncoder = LabelEncoder()
labelEncoder.fit(train['Service'])
train['Service'] = labelEncoder.transform(train['Service'])
labelEncoder.fit(test['Service'])
test['Service'] = labelEncoder.transform(test['Service'])
labelEncoder.fit(train['Flag'])
train['Flag'] = labelEncoder.transform(train['Flag'])
labelEncoder.fit(test['Flag'])
test['Flag'] = labelEncoder.transform(test['Flag'])
labelEncoder.fit(train['Duration.1'])
train['Duration.1'] = labelEncoder.transform(train['Duration.1'])
labelEncoder.fit(test['Duration.1'])
test['Duration.1'] = labelEncoder.transform(test['Duration.1'])

#Converting the dataset to Data and Target
X = np.array(train.drop(['Label'], 1).astype(float))
y = np.array(train['Label'])
X_test = np.array(test.drop(['Label'], 1).astype(float))
y_test = np.array(test['Label'])

#Fitting the model
abc = AdaBoostClassifier(n_estimators=50,learning_rate=1)
model= abc.fit(X, y)

#Predicting the labels for Test Data
y_pred = model.predict(X_test)

#Metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("Error metrics")
print("mean squared error",metrics.mean_squared_error(y_test,y_pred))
print("mean absolute error",metrics.mean_absolute_error(y_test,y_pred))
