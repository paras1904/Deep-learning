import keras as ks
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
import numpy as np
(trainX, trainY), (testX, testY) = ks.datasets.fashion_mnist.load_data()
trainX =trainX.reshape(60000,28,28,1)
testX = testX.reshape(10000,28,28,1)
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

model = Sequential([
    Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(10,activation='softmax')
])
model.compile(optimizer="adam",metrics=['accuracy'],loss='sparse_categorical_crossentropy')
model.fit(trainX,trainY,epochs=1,verbose=2)
