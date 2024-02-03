# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 21:37:51 2019

@author: Manoj V
"""

#Correlation Feature Seleciton 

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

url = "all.xlsx"
dataset = pd.read_excel(url)

dataset=dataset.drop(['Source_IP_Address','Destination_IP_Address','Start_Time'], axis=1)
#label Encode here 
train, test = train_test_split(dataset,test_size=0.4)


#train.dropna()
#test.dropna()

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


y_train=np.array(train['Label']).astype(float)
train=train.drop('Label',axis=1)

y_test=np.array(test['Label']).astype(float)
test=test.drop('Label',axis=1)

corr=train.corr()
#sns.heatmap(corr)


columns = np.full((corr.shape[0],), True, dtype=bool)


for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.6:
            if columns[j]:
                columns[j] = False

selected_columns = train.columns[columns]

x_train = train[selected_columns]
x_test=test[selected_columns]











