
import os
import caer
import canaro
import numpy as np
import cv2 as cv
import gc
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
import sklearn.model_selection as skm


IMG_SIZE = (80,80)
channels = 1
char_path = r'/home/paras/Data/simpsons_dataset/'

# Creating a character dictionary, sorting it in descending order
char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path,char)))

# Sort in descending order
char_dict = caer.sort_dict(char_dict, descending=True)
char_dict

#  Getting the first 10 categories with the most number of images
characters = []
count = 0
for i in char_dict:
    characters.append(i[0])
    count += 1
    if count >= 10:
        break
characters

# Create the training data
train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True)

# Number of training samples
print(len(train))

# Separating the array and corresponding labels
featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)


# Normalize the featureSet ==> (0,1)
featureSet = caer.normalize(featureSet)
# Converting numerical labels to binary class vectors
labels = to_categorical(labels, len(characters))


# Creating train and validation data

## NOTE:
## In the tutorial, I've use the following line
### x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio=.2)
## However, due to recent API changes in `caer`, this is now a deprecated feature.
## Instead, you can use the following line (which use's SkLearn's train-test split feature).
## Both achieve the same end result

# Do note that `val_ratio` is now `test_size`.
split_data = skm.train_test_split(featureSet, labels, test_size=.2)
x_train, x_val, y_train, y_val = (np.array(item) for item in split_data)


# Deleting variables to save memory
del train
del featureSet
del labels
gc.collect()

# Useful variables when training
BATCH_SIZE = 32
EPOCHS = 20

# Image data generator (introduces randomness in network ==> better accuracy)
datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
model = canaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE,channels=channels,output_dim=len(characters),loss='binary_crossentropy',decay=1e-6,momentum=0.9,nesterov=True,learning_rate=0.001)
print(model.summary())

from tensorflow.keras.callbacks import LearningRateScheduler
callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]

training = model.fit(train_gen,steps_per_epoch=len(x_train)//BATCH_SIZE,epochs=EPOCHS,validation_data=(x_val,y_val),validation_steps=len(y_val)//BATCH_SIZE,callbacks = callbacks_list)


def prepare(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img = cv.resize(img,IMG_SIZE)
    img = caer.reshape(img,IMG_SIZE,1)
    return img

img = cv.imread('pic_0094.jpg')
cv.imshow('img',img)
cv.waitKey(0)
pred = model.predict(prepare(img))
print(characters[np.argmax(pred[0])])