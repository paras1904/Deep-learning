import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

def prepare_data(timeseries_data, n_features):
    X, y =[],[]
    for i in range(len(timeseries_data)):
        end_ix = i + n_features
        if end_ix > len(timeseries_data)-1:
            break
        seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

timeseries_data = [110, 125, 133, 146, 158, 172, 187, 196, 210]
n_steps = 3
X, y = prepare_data(timeseries_data, n_steps)
# shape should be 1
print(X.shape)
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
print(X.shape)

model = Sequential()
model.add(GRU(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(GRU(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=300, verbose=0)

x_input = np.array([187, 196, 210])
temp_input = list(x_input)
lst_output = []
i = 0
while (i < 10):

    if (len(temp_input) > 3):
        x_input = np.array(temp_input[1:])
        print("{} day input {}".format(i, x_input))
        # print(x_input)
        x_input = x_input.reshape((1, n_steps, n_features))
        # print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i, yhat))
        temp_input.append(yhat[0][0])
        temp_input = temp_input[1:]
        # print(temp_input)
        lst_output.append(yhat[0][0])
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])
        i = i + 1

print(lst_output)