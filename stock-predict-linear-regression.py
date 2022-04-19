#This project must install quandl (dataset) library as before

import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

#get stock price 
stockName = "WIKI/GOOGL"#"WIKI/AAPL" #"WIKI/GOOGL"
df = quandl.get(stockName)
print(df.head())

# select feature
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

#create column for calculate High Low percentage change
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0 
# calculate percentatge change percentage change 
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0  

# select 
df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]
# print(df.head())


#forecast column
forecastCol = 'Adj. Close'

#data preparation (cleaning)
df.fillna(-99999, inplace=True)

print("Lenght of dataset: ", len(df))
forecastOut = int(math.ceil(0.01*len(df)))# set number of shift days 
print("Lenght of dataset: ", len(df), " ,  forecast day: ", forecastOut)

#create label column shift 10 days or 10% with Adj. Close price 
df['label']= df[forecastCol].shift(-forecastOut) 
df.dropna(inplace=True)
print(df.head())


#define training dataset get all column for feature beside 
X =  np.array(df.drop(['label'], 1))
# define label for training
y = np.array(df['label'])

# normalization 
X = preprocessing.scale(X) 

# X= X[:,-forecastOut + 1]
y = np.array(df['label'])
print(len(X), len(y))

#split data into training, test dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# init Linerat regression model
clf = LinearRegression(n_jobs=-1) # use linear regression
# clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train) # train model 

accuracy = clf.score(X_test, y_test) 

print(accuracy)