# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 21:03:13 2018

@author: TCBGULSEREN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

data = pd.read_csv('international-airline-passengers.csv',skipfooter=5)

dataset = data.iloc[:,1].values
plt.plot(dataset)
plt.xlabel("time")
plt.ylabel("Number of Passengers")
plt.title("International Airlane Passengers")
plt.show

dataset = dataset.reshape(-1,1)
dataset = dataset.astype("float32")

#Scaling
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset)*0.50)
test_size = len(dataset)-train_size

#Data train ve test olarak ikiye bölündü
train = dataset[0:train_size,:]
test = dataset[train_size:len(dataset),:]

timesteps =10
dataX = []
dataY = []

for i in range (len(train)-timesteps-1):
    a=train[i:(i+timesteps),0]
    dataX.append(a)
    dataY.append(train[i+timesteps,0])
    
trainX = np.array(dataX)
trainY = np.array(dataY)


dataX = []
dataY = []

for i in range (len(test)-timesteps-1):
    a=test[i:(i+timesteps),0]
    dataX.append(a)
    dataY.append(test[i+timesteps,0])
    
testX = np.array(dataX)
testY = np.array(dataY)

trainX = np.reshape(trainX ,(trainX.shape[0],1,trainX.shape[1]))
testX = np.reshape(testX ,(testX.shape[0],1,testX.shape[1]))

#LSTM creation
model = Sequential()
model.add(LSTM(10,input_shape=(1,timesteps)))
model.add(Dense(units=1))

model.compile(loss = 'mean_squared_error',optimizer = 'adam')
model.fit(trainX,trainY,epochs = 50 , batch_size = 1)


#Prediction

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

#İnvert scaling
trainPredict=scaler.inverse_transform(trainPredict)
testPredict=scaler.inverse_transform(testPredict)
trainY=scaler.inverse_transform([trainY])
testY=scaler.inverse_transform([testY])

#Error hesabı
trainScore = math.sqrt(mean_squared_error(trainY[0],trainPredict[:,0]))
print('Train score:%.2f RMSE'%(trainScore))

testScore = math.sqrt(mean_squared_error(testY[0],testPredict[:,0]))
print('Test score:%.2f RMSE'%(testScore))

#Shifting and Plotting
plt.plot(scaler.inverse_transform(dataset))
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:,:]=np.nan
trainPredictPlot[timesteps:len(trainPredict)+timesteps,:] = trainPredict
plt.plot(trainPredictPlot)
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:,:]=np.nan
testPredictPlot[len(testPredict)+(timesteps*2)+2:len(dataset),:] = testPredict
plt.plot(testPredictPlot)
































    


