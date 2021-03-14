import keras
from keras.datasets import fashion_mnist
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
print(x_train.shape)
train_x = x_train.reshape(-1,28,28,1)
train_x = train_x.astype(float)
train_x = train_x/255
train_y_one_hot = to_categorical(y_train)

test_x = x_test.reshape(-1,28,28,1)
test_x = test_x.astype(float)
test_x = test_x/255
test_y_one_hot = to_categorical(y_test)

model = Sequential([
    Conv2D(64,(3,3),input_shape=(28,28,1)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64,(3,3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(64),
    Dense(10),
    Activation('softmax')
])
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(lr=0.001),metrics=['accuracy'])
model.fit(train_x,train_y_one_hot,batch_size=100,epochs=2)
test_loss,test_acc = model.evaluate(test_x,test_y_one_hot)
print(test_loss)
print(test_acc)

prediction = model.predict(test_x)
print(np.argmax(np.round(prediction[1])))
plt.imshow(test_x[1].reshape(28,28))
plt.show()