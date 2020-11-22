import pandas as pd

df = pd.read_csv('house.csv')
x = df[["area","bedrooms","age"]]
y = df["price"]

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam

model = Sequential([
    Dense(6,input_dim=3,activation='relu'),
    Dropout(.2,input_shape=(6,)),
    Dense(2,activation='linear')
])
model.compile(Adam(lr=0.000001),loss="mean_squared_error",metrics=['accuracy'])
model.fit(x,y,batch_size=10,epochs=10,shuffle=True,verbose=2)

a = model.predict([[2600,3,20]])
print(a)