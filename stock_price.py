# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.optimizers import RMSprop,Adam
import pickle

dataset_train = pd.read_csv('Stock_Price_Train.csv')

#OpEN sütunu alındı
train = dataset_train.loc[:,["Open"]].values

#Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
train_scaled = scaler.fit_transform(train)

#Plotting
plt.plot(train_scaled)
plt.show()

#Datayı X_train ve Y_tarin olarak ayırma

X_train = []
Y_train = []
timesteps = 50

for i in range(timesteps,1258):
    X_train.append(train_scaled[i-timesteps:i,0])
    Y_train.append(train_scaled[i,0])

#Array haline getiriliyor
X_train , Y_train = np.array(X_train),np.array(Y_train)

#3 boyutlu hale geitirildi
X_train = np.reshape(X_train , (X_train.shape[0],X_train.shape[1],1))


#RNN modeling
regressor = Sequential()

regressor.add(SimpleRNN(units =50,activation ='tanh',return_sequences =True , input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units =50,activation ='tanh',return_sequences =True))
regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units =50,activation ='tanh',return_sequences =True))
regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units =50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer ='adam',loss='mean_squared_error' , metrics=['accuracy'])


regressor.fit(X_train,Y_train,epochs =200 ,batch_size =32)



#Test datasını alma
dataset_test= pd.read_csv('Stock_Price_Test.csv')
real_stock_prize = dataset_test.loc[:,["Open"]].values
dataset_total =pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-timesteps:].values.reshape(-1,1)
inputs = scaler.transform(inputs)


X_test = []
for i in range(timesteps,70):
    X_test.append(inputs[i-timesteps:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))


predicted_stock_prize = regressor.predict(X_test)
predicted_stock_prize= scaler.inverse_transform(predicted_stock_prize)

#Grafikleştirme
plt.plot(real_stock_prize,color='red',label = 'Real Google Stock Price')
plt.plot(predicted_stock_prize, color='blue',label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()




































































    

