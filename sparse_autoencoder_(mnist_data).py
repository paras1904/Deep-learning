# -*- coding: utf-8 -*-
"""Sparse autoencoder (mnist data).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vSIBO2k_vRCLnhqs_O0ALjqmHY7PcTp3
"""

import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1
from tensorflow.keras.datasets import mnist

(x_train,_),(x_test,_) = mnist.load_data()

x_train = x_train.astype('float32')/255.
x_train = np.reshape(x_train,(len(x_train),28,28,1))

x_test = x_test.astype('float32')/255.
x_test = np.reshape(x_test,(len(x_test),28,28,1))

noise_fct = 0.5
x_train_noisy = x_train + noise_fct * np.random.normal(loc = 0.0, scale = 1.0, size = x_train.shape)
x_test_noisy = x_test + noise_fct * np.random.normal(loc = 0.0, scale = 1.0, size = x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

plt.figure(figsize=(20, 2))
for i in range(1,10):
  ax = plt.subplot(1, 10 ,i)
  plt.imshow(x_test_noisy[i].reshape(28,28),cmap = 'binary')
plt.show()

size = 28
model = Sequential([
                    Conv2D(32,(3,3),activation='relu',padding='same',input_shape = (28,28,1),kernel_regularizer=l1(0.001)),
                    MaxPooling2D((2,2),padding = 'same'),
                    Conv2D(8,(3,3),activation='relu',padding='same',kernel_regularizer=l1(0.001)),
                    MaxPooling2D((2,2),padding = 'same'),

                    Conv2D(8,(3,3),activation='relu',padding='same'),
                    UpSampling2D((2,2)),
                    Conv2D(32,(3,3),activation='relu',padding = 'same'),
                    UpSampling2D((2,2)),
                    Conv2D(1,(3,3),padding = 'same',activation = 'relu')
])

model.summary()

model.compile(optimizer='adam',loss = 'mean_squared_error')

model.fit(x_train_noisy,x_train,epochs = 10,shuffle = False,batch_size = 200)

no_noise_img = model.predict(x_test_noisy)

plt.figure(figsize=(40, 4))
for i in range(10):
    # display original
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap="binary")

    # display reconstructed (after noise removed) image
    ax = plt.subplot(3, 20, 40 + i + 1)
    plt.imshow(no_noise_img[i].reshape(28, 28), cmap="binary")

plt.show()

