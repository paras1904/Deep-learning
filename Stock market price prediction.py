import numpy
import numpy as np
import pandas as pd

df = pd.read_csv('AAPL.csv')
# print(df.head())
df1 = df.reset_index()['close']
# print(df1.head())

# import matplotlib.pyplot as plt
# plt.plot(df2)
# plt.show()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))
# print(df1.shape)

training_size = int(len(df1)*0.65)
test_size = len(df1)-training_size
train_data, test_data = df1[0:training_size],df1[training_size:len(df1),:1]

def create_dataset(dataset,time_step = 1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return np.array(dataX), np.array(dataY)

time_step = 100
X_train, y_train = create_dataset(train_data,time_step)
X_test,y_test = create_dataset(test_data,time_step)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.models import Sequential

model = Sequential([
    LSTM(50,return_sequences=True,input_shape = (100,1)),
    LSTM(50,return_sequences=True),
    LSTM(50),
    Dense(1),

])
model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64,verbose=1)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)


import math
from sklearn.metrics import mean_squared_error
rmse_train = math.sqrt(mean_squared_error(y_train,train_predict))
rmse_test = math.sqrt(mean_squared_error(y_test,test_predict))
print(rmse_train,rmse_test)
