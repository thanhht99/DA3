# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 23:01:34 2020

@author: Admin
"""
import pandas as pd # import thư viện pandas để dùng dataframe
from pandas import read_csv, DataFrame
import random
import numpy as np #import thư viện numpy
import matplotlib.pyplot as plt #import thư viện matplotlip
import seaborn as sns

adults = pd.read_csv('adult.data.csv',names=['Age','workclass','fnlwgt'
                                             ,'education','education_num'
                                             ,'marital_status','occupation'
                                             ,'relationship','race','sex'
                                             ,'capital_gain','capital_loss'
                                             ,'hours_per_week','native_country','label'])
adults_test = pd.read_csv('adult.data.csv',names=['Age','workclass','fnlwgt'
                                                  ,'education','education_num'
                                                  ,'marital_status','occupation'
                                                  ,'relationship','race','sex'
                                                  ,'capital_gain','capital_loss'
                                                  ,'hours_per_week','native_country','label'])


train_data = adults.drop('label',axis=1)

test_data = adults_test.drop('label',axis=1)

data = train_data.append(test_data)

label = adults['label'].append(adults_test['label'])

print(data.head())
print(' ')
print('--------------------------')
print(' ')

full_dataset = adults.append(adults_test)

print(label.head())
print(' ')
print('--------------------------')
print(' ')

data_binary = pd.get_dummies(data)

print(data_binary.head())
print(' ')
print('--------------------------')
print(' ')


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_binary,label)

print(x_train)
print(' ')
print('--------------------------')
print(' ')
# print(np.any(np.isnan(x_test)))
print(x_test)
print(' ')
print('--------------------------')
print(' ')
print(y_train)
print(' ')
print('--------------------------')
print(' ')
print(y_test)
print(' ')
print('--------------------------')
print(' ')

performance = []

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

GNB = GaussianNB()

# Binary data
GNB.fit(x_train,y_train)
train_score = GNB.score(x_train,y_train)
test_score = GNB.score(x_test,y_test)
print(f'Gaussian Naive Bayes : Training score - {train_score} - Test score - {test_score}')

performance.append({'algorithm':'Gaussian Naive Bayes', 'training_score':train_score, 'testing_score':test_score})