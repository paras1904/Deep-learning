import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#set seed
from numpy.random import seed
seed(10)
tf.random.set_seed(10)
print('Imported Successfully')


img_folder  = '/home/paras/Data/xray_dataset_covid19/train/'
plt.figure(figsize=(20,20))
for i in range(6):
 class_ = random.choice(os.listdir(img_folder))
 class_path= os.path.join(img_folder, class_)
 file=random.choice(os.listdir(class_path))
 image_path= os.path.join(class_path,file)
 print(image_path)
 img= mpimg.imread(image_path)
 ax=plt.subplot(1,6,(i+1))
 plt.imshow(img)
 ax.title.set_text(class_)

def create_dataset(img_folders,IMG_WIDTH,IMG_HEIGHT):
    img_data_array = []
    class_name = []
    n = 0
    for dirname, _, filenames in os.walk(img_folders):
        for filename in filenames:
            img_path = os.path.join(dirname, filename)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image,(IMG_WIDTH,IMG_HEIGHT))
            image = np.array(image)
            image = image.astype('float32')/255.
            img_data_array.append(image)
            class_=str(dirname).split("/")[-1]
            class_name.append(class_)
            n+=1
    return img_data_array, class_name,n

IMG_HEIGHT = 224
IMG_WIDTH = 224
train_path = '/home/paras/Data/xray_dataset_covid19/train/'
test_path = '/home/paras/Data/xray_dataset_covid19/test/'
train_img, train_target, n = create_dataset(train_path,IMG_WIDTH,IMG_HEIGHT)
test_img, test_target, n = create_dataset(test_path,IMG_WIDTH, IMG_HEIGHT)

target_dict={k: v for v, k in enumerate(np.unique(train_target))}
print(target_dict)
train_target= [target_dict[train_target[i]] for i in range(len(train_target))]
train_target=np.array(train_target)
train_img=np.array(train_img)
test_target= [target_dict[test_target[i]] for i in range(len(test_target))]
test_target=np.array(test_target)
test_img=np.array(test_img)

model = Sequential([
    Conv2D(64,(3,3),use_bias=True,kernel_initializer='normal',bias_initializer='zero',kernel_regularizer='l1',bias_regularizer='l1',padding='same',strides=(1,1),activation='relu',input_shape=(IMG_WIDTH,IMG_HEIGHT,3)),
    MaxPooling2D(pool_size=2,strides=1),
    Conv2D(64,(3,3),use_bias=True,kernel_initializer='normal',bias_initializer='zero',kernel_regularizer='l1',bias_regularizer='l1',padding='same',strides=(1,1),activation='relu'),
    MaxPooling2D(pool_size=2,strides=1),
    Conv2D(32,(3,3),use_bias=True,kernel_initializer='normal',bias_initializer='zero',kernel_regularizer='l1',bias_regularizer='l1',padding='same',strides=(1,1),activation='relu'),
    MaxPooling2D(pool_size=2,strides=1),
    Conv2D(16,(3,3),use_bias=True,kernel_initializer='normal',bias_initializer='zero',kernel_regularizer='l1',bias_regularizer='l1',padding='same',strides=(1,1),activation='relu'),
    MaxPooling2D(pool_size=2,strides=1),
    Flatten(),
    Dense(64,use_bias=True,kernel_initializer='normal',bias_initializer='zero',kernel_regularizer='l1',bias_regularizer='l1',activation='relu'),
    Dropout(0.24),
    Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam',loss = ['binary_crossentropy'],metrics=['accuracy'])
model.fit(train_img,train_target,epochs=5,shuffle=False)
pred = model.predict(test_img,batch_size=32)
label = [int(p>=0.5) for p in pred]
print(label)