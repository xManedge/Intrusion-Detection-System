# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:44:22 2019

@author: Manoj V
"""

#Adaboost algorithm for Machine Learning

#import the necessary libraries and user definied functions 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from Libraries import Load, LabelEncode, CFS, metric_analysis, target
import datetime
#Loading the dataset

url = ['all.xlsx','20071006.xlsx','20081204.xlsx','20090903.xlsx','20111104.xlsx','20120505.xlsx','20131101.xlsx','20141103.xlsx','20151105.xlsx']
for i in url:
    a=datetime.datetime.now().replace(microsecond=0)    
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
    
    #applying CFS on the train data and selecting columns recomended by CFS
    selected_columns = CFS(train)
    train = train[selected_columns]
    test = test[selected_columns]
    
    
    #Performing the required adaptive boost learning 
    adaptive_boost = AdaBoostClassifier(n_estimators=50,learning_rate=1)
    model= adaptive_boost.fit(train, y_train)
    
    #Predicting the values for test data
    y_pred = model.predict(test)
    
    #Performing the required analysis
#    metric_analysis(y_test,y_pred)
    
    print(datetime.datetime.now().replace(microsecond=0)-a)
