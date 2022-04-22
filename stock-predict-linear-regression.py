#This project must install quandl (dataset) library as before
#YouTube Tutorial for Practical Machine learning 
#https://youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v 

import pandas as pd
import quandl
import math ,  datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


style.use('ggplot')

#get stock price 
stockName = "WIKI/GOOGL"#"WIKI/AAPL" #"WIKI/GOOGL"
shiftDayRatio = 0.05  # 10% for total day

df = quandl.get(stockName)
print(df.head())

# select feature
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

#create column for calculate High Low percentage change
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0 
# calculate percentatge change percentage change 
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0  

# select  price  x x x
df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]
# print(df.head())


#forecast column
forecastCol = 'Adj. Close'

#data preparation (cleaning)
df.fillna(-99999, inplace=True)

print("Lenght of dataset: ", len(df))
forecastOut = int(math.ceil(shiftDayRatio *len(df)))# set number of shift days 
print("Lenght of dataset: ", len(df), " ,  forecast day: ", forecastOut)

#create label column shift 10 days or 10% with Adj. Close price 
df['label']= df[forecastCol].shift(-forecastOut) 

print(df.head())


#define training dataset get all column for feature beside 
X =  np.array(df.drop(['label'], 1)) # drop label data at training dataset
# X =  np.array(df.drop(['label', 'Adj. Close'], 1)) # drop label and  Adj. Close price for training dataset testing 
X = preprocessing.scale(X) 
X = X[:-forecastOut]
XLately = X[-forecastOut:]



df.dropna(inplace=True)
y = np.array(df['label']) # define label for training
# X= X[:,-forecastOut + 1]
print(len(X), len(y))

#split data into training, test dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# init Linerat regression model
clf = LinearRegression(n_jobs=-1) # use linear regression
# clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train) # train model 
# save trained model
with open("stock-price-linear-regression.sav", 'wb') as f:
    pickle.dump(clf, f)

# load trained model from file 
pickleIn = open("stock-price-linear-regression.sav", 'rb')
clf = pickle.load(pickleIn)


accuracy = clf.score(X_test, y_test) 

# print(accuracy)

# validation with new data
forecastSet = clf.predict(XLately)
print(forecastSet, accuracy, forecastOut)

# add datetime handling

df['Forecast'] = np.nan # create forecast column and fill nan value

lastDate = df.iloc[-1].name #go to last column
lastUnix = lastDate.timestamp()
oneDay = 86400
nextUnix = lastUnix + oneDay 

for i in forecastSet:
    nextDate = datetime.datetime.fromtimestamp(nextUnix)
    nextUnix += oneDay
    df.loc[nextDate] = [np.nan for  _ in range(len(df.columns)-1)] + [i] #fill nan for other column
print(df.tail())
    

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()