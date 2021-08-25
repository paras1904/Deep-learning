# -*- coding: utf-8 -*-
"""Autoencoder Image colorisation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pk33QJtK5WzdQJBjqpcV4EOJksZAubtC

vg16 model link -> https://github.com/bnsreenu/python_for_microscopists/blob/master/other_files/colorize_autoencoder_VGG16_10000.model
"""

import numpy as np
from matplotlib.pyplot import imshow
from google.colab.patches import cv2_imshow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import cv2 as cv

path = '/content/chandler_autoencoder.jpg'

img1 = cv.imread(path,1)
cv2_imshow(img1)


img_data1 = []
size = 256
img1 = cv.resize(img1,(size,size))
img_data1.append(img_to_array(img1))
img_data1 = np.reshape(img_data1,(len(img_data1),size,size,3))
img_data1 = img_data1.astype('float32')/255.

img2 = cv.imread(path,0)
cv2_imshow(img2)

img_data2 = []
size = 256
img2 = cv.resize(img2,(size,size))
img_data2.append(img_to_array(img2))
img_data2 = np.reshape(img_data2,(len(img_data2),size,size,1))
img_data2 = img_data2.astype('float32')/255.

from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1

size = 256
model = Sequential([
                    Conv2D(32,(3,3),activation='relu',padding='same',input_shape = (size,size,1)),
                    MaxPooling2D((2,2),padding='same'),
                    Conv2D(8,(3,3),activation='relu',padding='same'),
                    MaxPooling2D((2,2),padding='same'),
                    Conv2D(8,(3,3),activation='relu',padding='same'),
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
model.fit(img_data2,img_data1,epochs = 20000,shuffle = True)

pred = model.predict(img_data2)
img = pred[0].reshape(size,size,3)
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
imshow(img)