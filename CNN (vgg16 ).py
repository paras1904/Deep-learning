import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense , Flatten , BatchNormalization , Conv2D , MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)

os.chdir('/home/paras/Data/dogs-vs-cats')
if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')

    for c in random.sample(glob.glob('cat*'), 500):
        shutil.move(c, 'train/cat')
    for c in random.sample(glob.glob('dog*'), 500):
        shutil.move(c, 'train/dog')
    for c in random.sample(glob.glob('cat*'), 100):
        shutil.move(c, 'test/cat')
    for c in random.sample(glob.glob('dog*'), 100):
        shutil.move(c, 'test/dog')
    for c in random.sample(glob.glob('cat*'), 50):
        shutil.move(c, 'valid/cat')
    for c in random.sample(glob.glob('dog*'), 50):
        shutil.move(c, 'valid/dog')
os.chdir('../../')

train_path = "/home/paras/Data/dogs-vs-cats/train"
test_path = "/home/paras/Data/dogs-vs-cats/test"
valid_path = "/home/paras/Data/dogs-vs-cats/valid"

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path,target_size=(224,224),classes=['cat','dog'],batch_size=15)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory = test_path,target_size=(224,224),classes=['cat','dog'],batch_size=15)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory = valid_path,target_size=(224,224),classes=['cat','dog'],batch_size=15,shuffle=False)

imgs,labels = next(train_batches)
vgg16_model = tf.keras.applications.vgg16.VGG16()
# vgg16_model.summary()
# type(vgg16_model)

model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
# model.summary()

for layer in model.layers:
    layer.trainable = False

model.add(Dense(units = 2,activation = "softmax"))
# model.summary()
model.compile(optimizer=Adam(lr=0.0001),loss="categorical_crossentropy",metrics= ['accuracy'])
model.fit(x = train_batches,validation_data = valid_batches,epochs=1,verbose = 2)
predictions = model.predict(x = test_batches,verbose=0)
print(predictions)