from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import MaxPooling2D, Dropout, Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np


(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

size = (28,28,1)

model = Sequential([
    Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',use_bias=True,kernel_initializer='normal',kernel_regularizer='l1',bias_regularizer='l1',bias_initializer='zero',input_shape=size),
    MaxPooling2D(pool_size=2,strides=1),
    Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',use_bias=True,kernel_initializer='normal',kernel_regularizer='l1',bias_regularizer='l1',bias_initializer='zero'),
    MaxPooling2D(pool_size=2,strides=1),
    Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',use_bias=True,kernel_initializer='normal',kernel_regularizer='l1',bias_regularizer='l1',bias_initializer='zero'),
    MaxPooling2D(pool_size=2,strides=1),
    Conv2D(32,(3,3),strides=(1,1),padding='same',activation='relu',use_bias=True,kernel_initializer='normal',kernel_regularizer='l1',bias_regularizer='l1',bias_initializer='zero'),
    MaxPooling2D(pool_size=2,strides=1),
    Conv2D(16,(3,3),strides=(1,1),padding='same',activation='relu',use_bias=True,kernel_initializer='normal',kernel_regularizer='l1',bias_regularizer='l1',bias_initializer='zero'),
    Flatten(),
    Dense(32,activation='relu',use_bias=True,kernel_initializer='normal',bias_initializer='zero',kernel_regularizer='l1',bias_regularizer='l1'),
    Dropout(.2),
    Dense(20,activation='relu',use_bias=True,kernel_initializer='normal',bias_initializer='zero',kernel_regularizer='l1',bias_regularizer='l1'),
    Dropout(.2),
    Dense(10,activation='softmax'),
])
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
model.fit(x_train,y_train)
print(model.evaluate(x_test,y_test,batch_size=500,verbose=1))
