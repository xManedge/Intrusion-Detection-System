# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 22:48:01 2019

@author: Manoj V
"""

from Libraries import Load, CFS, LabelEncode, metric_analysis, target
from sklearn.model_selection import train_test_split


url = ['all.xlsx','20071006.xlsx','20081204.xlsx','20090903.xlsx','20111104.xlsx','20120505.xlsx','20131101.xlsx','20141103.xlsx','20151105.xlsx']


for i in url: 
    Dataset = Load(i)
    Dataset['Service'] = LabelEncode(Dataset,'Service')
    Dataset['Flag'] = LabelEncode(Dataset,'Flag')
    Dataset['Duration.1']=LabelEncode(Dataset,'Duration.1')
    
    train, test = train_test_split(Dataset,test_size=0.4)
    
    y_train, train = target(train)
    print('Name : ' + i)
    print(CFS(train))

