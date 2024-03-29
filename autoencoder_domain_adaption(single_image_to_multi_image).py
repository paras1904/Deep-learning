# -*- coding: utf-8 -*-
"""Autoencoder Domain adaption(single image to multi image).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XCkkjFagj9B-MTos3b3217GeSI_MVgxg
"""

import numpy as np
import cv2 as cv
from google.colab.patches import cv2_imshow
from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing.image import img_to_array

def func1(path):
  img_data = []
  size = 256
  img = cv.imread(path,1)
  img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  img = cv.resize(img,(size,size))
  img_data.append(img_to_array(img))
  img_data = np.reshape(img_data,(len(img_data),size,size,3))
  img_data = img_data.astype('float32')/255.
  return img_data
path1 = '/content/chandler_autoencoder.jpg'
img_data1 = func1(path1)

path2 = '/content/joey.jpg'
img_data2 = func1(path2)

from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1

size = 256
model = Sequential([
                    Conv2D(32,(3,3),activation='relu',padding='same',kernel_regularizer=l1(0.001),input_shape = (size,size,3)),
                    MaxPooling2D((2,2),padding='same'),
                    Conv2D(8,(3,3),activation='relu',kernel_regularizer=l1(0.001),padding='same'),
                    MaxPooling2D((2,2),padding='same'),
                    Conv2D(8,(3,3),activation='relu',kernel_regularizer=l1(0.001),padding='same'),
                    MaxPooling2D((2,2),padding='same'),

                    Conv2D(8,(3,3),activation='relu',padding='same'),
                    UpSampling2D((2,2)),
                    Conv2D(8,(3,3),activation='relu',padding='same'),
                    UpSampling2D((2,2)),
                    Conv2D(32,(3,3),activation='relu',padding='same'),
                    UpSampling2D((2,2)),
                    Conv2D(3,(3,3),activation='relu',padding='same')
])

model.summary()

model.compile(optimizer='adam',loss = 'mean_squared_error')

model.fit(img_data1,img_data2,epochs = 100000,shuffle = True)

pred = model.predict(img_data1)
imshow(pred[0].reshape(size,size,3))

