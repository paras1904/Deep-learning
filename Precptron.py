import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler

train_samples = []
train_labels = []

for i in range(1000):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)

for i in range(50):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler_train_samples = scaler.fit_transform((train_samples).reshape(-1, 1))
# print(scaler_train_samples)


test_sample = []
test_label = []
for i in range(10):
    random_younger = randint(13, 64)
    test_sample.append(random_younger)
    test_label.append(1)

    random_older = randint(64, 100)
    test_sample.append(random_older)
    test_label.append(0)

for i in range(200):
    random_younger = randint(13, 64)
    test_sample.append(random_younger)
    test_label.append(0)

    random_older = randint(64, 100)
    test_sample.append(random_older)
    test_label.append(1)

test_label = np.array(test_label)
test_sample = np.array(test_sample)

scaler_test_sample = scaler.fit_transform((test_sample).reshape(-1, 1))
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam

model = Sequential([
    Dense(16, input_dim=1, activation='relu'),
    Dense(2, activation='softmax')
])
# print(model.summary())

model.compile(Adam(lr=0.0001), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.fit(scaler_train_samples, train_labels, batch_size=10, epochs=10, shuffle=True, verbose=2)

a = model.predict(x=scaler_test_sample, batch_size=10, verbose=0)

rounded_prediction = np.argmax(a, axis=-1)

for i in rounded_prediction:
    print(i)
# model.save("AI/model1")
