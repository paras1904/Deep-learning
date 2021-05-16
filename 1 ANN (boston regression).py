import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.datasets import load_boston
boston = load_boston()
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names

data["PRICE"] = boston.target
X = data.drop(['PRICE'],axis = 1)
Y = data['PRICE']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=0,
                                                 shuffle=False,
                                                 train_size=70)



from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
model = Sequential([
    Dense(13,input_dim = 13,use_bias=True,bias_regularizer='l1',kernel_regularizer='l1',bias_initializer='zero',kernel_initializer='normal',activation = 'relu'),
    Dense(26,activation='relu',use_bias = True,bias_regularizer='l1',bias_initializer='zero',kernel_regularizer='l1',kernel_initializer='normal'),
    Dense(26,activation='relu',use_bias = True,bias_regularizer='l1',bias_initializer='zero',kernel_regularizer='l1',kernel_initializer='normal'),
    Dense(13,activation='relu',use_bias=True,bias_initializer='zero',bias_regularizer='l1',kernel_initializer='normal',kernel_regularizer='l1'),
    Dense(1,activation='linear')
])
model.compile(optimizer='adam',loss='mean_squared_error',
             metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=100,shuffle=False,verbose=2)

a = model.predict([[0.00632,8.0,2.31,0.0,0.538,6.575,65.2,4.0900,1.0,296.0,15.3,396.90,4.98]])
print(a)