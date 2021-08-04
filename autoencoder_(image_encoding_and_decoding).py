# -*- coding: utf-8 -*-
"""Autoencoder (image encoding and decoding).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pIjgIx-u4kBizXhJsCT-f0Lndxr0e4_3
"""

from matplotlib.pyplot import imshow
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

np.random.seed(42)
size = 256

img_data = []

img = cv2.imread('/content/chandler_autoencoder.jpg',1)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(size,size))

img_data.append(img_to_array(img))
img_array = np.reshape(img_data,(len(img_data),size,size,3))
img_array = img_array.astype('float32')/255.

model = Sequential([
                    Conv2D(512,(3,3),activation='relu',padding='same',input_shape=(size,size,3)),
                    MaxPooling2D((2,2),padding='same'),
                    Conv2D(256,(3,3),activation='relu',padding='same'),
                    MaxPooling2D((2,2),padding='same'),
                    Conv2D(128,(3,3),activation='relu',padding='same'),
                    MaxPooling2D((2,2),padding='same'),
                    Conv2D(64,(3,3),activation='relu',padding='same'),
                    MaxPooling2D((2,2),padding='same'),
                    Conv2D(32,(3,3),activation='relu',padding='same'),
                    MaxPooling2D((2,2),padding='same'),
                    Conv2D(16,(3,3),activation='relu',padding='same'),
                    MaxPooling2D((2,2),padding='same'),
                    Conv2D(8,(3,3),activation='relu',padding='same'),
                    MaxPooling2D((2,2),padding='same'), 

                    Conv2D(8,(3,3),activation='relu',padding='same'),
                    UpSampling2D((2,2)),
                    Conv2D(16,(3,3),activation='relu',padding='same'),
                    UpSampling2D((2,2)),
                    Conv2D(32,(3,3),activation='relu',padding='same'),
                    UpSampling2D((2,2)),
                    Conv2D(64,(3,3),activation='relu',padding='same'),
                    UpSampling2D((2,2)),
                    Conv2D(128,(3,3),activation='relu',padding='same'),
                    UpSampling2D((2,2)),
                    Conv2D(256,(3,3),activation='relu',padding='same'),
                    UpSampling2D((2,2)),
                    Conv2D(512,(3,3),activation='relu',padding='same'),
                    UpSampling2D((2,2)),
                    Conv2D(3,(3,3),activation='relu',padding='same'),

])


model.compile(optimizer='adam',loss='mean_squared_error',metrics = ['accuracy'])
model.summary()

"""# New Section"""

model.fit(img_array,img_array,epochs=1000,shuffle = True)

pred = model.predict(img_array)
imshow(pred[0].reshape(size,size,3))

