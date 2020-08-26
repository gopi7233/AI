# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:23:31 2020

@author: thummago
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import pickle 

dataset = pd.read_csv('book1.csv')

dataset['Experience '].fillna(0, inplace=True)

dataset['Test_score'].fillna(dataset['Test_score'].mean(), inplace=True)

X = dataset.iloc[:, :3]


def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]


X['Experience '] = X['Experience '].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


regressor.fit(X, y)

pickle.dump(regressor, open('model.pkl', 'wb'))

model = pickel.load(open('model.pkl', 'rb,))
print(model.predict([[2, 9, 6]]))

