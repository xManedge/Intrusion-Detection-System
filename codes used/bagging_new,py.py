# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:59:49 2019

@author: Manoj V
"""


#import the necessary libraries and user definied functions 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from Libraries import Load, CFS, LabelEncode, metric_analysis, target
import datetime

#Loading the dataset
url = ['all.xlsx','20071006.xlsx','20081204.xlsx','20090903.xlsx','20111104.xlsx','20120505.xlsx','20131101.xlsx','20141103.xlsx','20151105.xlsx']
for i in url:
    a=datetime.datetime.now()
    Dataset = Load(i)
    
    #Label Encoding 
    Dataset['Service'] = LabelEncode(Dataset,'Service')
    Dataset['Flag'] = LabelEncode(Dataset,'Flag')
    Dataset['Duration.1']=LabelEncode(Dataset,'Duration.1')
    
    #Splitting the dataset into test and train
    train, test = train_test_split(Dataset,test_size=0.4)
    
    #Storing the target column seperately and discarding the column from the dataset
    y_train, train = target(train)
    y_test, test = target(test)
    
    #Applying CFS to the algorithm
    selected_columns = CFS(train)
    train = train[selected_columns]
    test = test[selected_columns]
    
    #Perfomring the bagging operation
    cart = DecisionTreeClassifier()
    bac = BaggingClassifier(base_estimator=cart,n_estimators=50)
    model= bac.fit(train, y_train)
    
    #Predicting for test data
    y_pred = model.predict(test)
    
#    metric_analysis(y_test, y_pred)
    print(datetime.datetime.now().replace(microsecond=0)-a)

