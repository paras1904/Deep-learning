# data link = "https://urbansounddataset.weebly.com/download-urbansound8k.html"
import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

audio_dataset_path = r'/home/paras/Data/UrbanSound8K/audio/'
meta_data = pd.read_csv(r'/home/paras/Data/UrbanSound8K/metadata/UrbanSound8K.csv')
# print(meta_data.head(2))

def feature_extract(file):
    audio,rate = librosa.load(file,res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y = audio,sr=rate,n_mfcc=40)
    mfccs_sacled_features = np.mean(mfccs_features.T,axis=0)

    return mfccs_sacled_features

extracted_feature = []
for index_num,row in tqdm(meta_data.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row['fold'])+'/',str(row['slice_file_name']))
    final_class_label = row['class']
    data = feature_extract(file_name)
    extracted_feature.append([data,final_class_label])


extracted_feature_df = pd.DataFrame(extracted_feature,columns=['feature','class'])
# print(extracted_feature_df.head(5))
X = np.array(extracted_feature_df['feature'].tolist())
y = np.array(extracted_feature_df['cl ass'].tolist())
y = np.array(pd.get_dummies(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics

num_labels = y.shape[1]
model = Sequential([
    Dense(100,input_shape=(40,)),
    Activation('relu'),
    Dropout(0.5),

    Dense(200),
    Activation('relu'),
    Dropout(0.5),

    Dense(num_labels),
    Activation('softmax'),
])
model.summary()
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

from tensorflow.keras.callbacks import ModelCheckpoint
num_epochs = 200
num_batch = 45
checkpointer = ModelCheckpoint(filepath='/home/paras/Data/saved_model/audioclf.hdf5',
                               save_best_only=True,verbose=1)

model.fit(X_train,y_train,batch_size=num_batch,epochs=num_epochs,validation_data=(X_test,y_test))
print('DONE')

test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])

file_name_test = r'/home/paras/Data/7061-6-0-0.wav'
prediction_feature = feature_extract(file_name_test)
prediction_feature = prediction_feature.reshape(1,-1)
print(model.predict_classes(prediction_feature))