# multivariate data preparation
from pickle import load

import pandas
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.engine.saving import load_model
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l2
from keras.utils import np_utils
from numpy import array
from numpy import hstack
import matplotlib.pyplot as plt
import numpy as np

from sklearn import preprocessing
# from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from tensorflow_core.python import confusion_matrix
from sklearn.utils import shuffle

# split a multivariate sequence into samples
def split_sequences(X_scaler, dummy_y, n_steps):
    x, y = list(), list()
    for i in range(len(X_scaler)-1):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(X_scaler)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = X_scaler[i:end_ix, :len(X_scaler[i])], dummy_y[end_ix, :len(dummy_y[i])]
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)

def plot_loss(history):
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

def plot_acuuracy(history):
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()

# # define input sequence
dataFrame = pandas.read_csv("../data/preprocessedNew.csv")
dataset = dataFrame.values
# dataset = shuffle(dataset)
X = dataset[:, [6, 7, 8, 9, 19, 22]]
# plt.plot(X)
# plt.show()
Y = dataset[:, 5]
dummy_y = np_utils.to_categorical(Y)
print(X)
print(dummy_y[1])
min_max_scaler = preprocessing.MinMaxScaler()
X_scaler = min_max_scaler.fit_transform(X)
print(X_scaler)

# choose a number of time steps
n_steps = 2

# convert into input/output
x, y = split_sequences(X_scaler, dummy_y, n_steps)
print(x.shape, y.shape)

#train_test_validation split
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(x, y, test_size=0.2)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

# summarize the data
for i in range(len(x)):
    print(x[i], y[i])
print('value: ', x[442],y[442])
n_features = x.shape[2]
# print(n_features)

# define model
batchSize = 100
epochs = 200
model = Sequential()
# model.add(Dense(64, input_shape=(n_steps, n_features), activation='relu'))
# model.add(Dropout(0.2))
model.add(LSTM(150, activation='relu', input_shape=(n_steps, n_features), kernel_regularizer=l2(0.01), return_sequences=True))
model.add(LSTM(100, activation='relu', input_shape=(n_steps, n_features), kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(48))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fit model
history = model.fit(x, y, epochs=epochs, batch_size=batchSize, verbose=0, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=15, min_delta=0.0001, verbose=1)])
# history = model.fit(x, y, epochs=epochs, batch_size=batchSize, verbose=0, validation_split=0.2)
print(X_test.shape)
# print(x.shape)

#model summary
print(model.summary())

#model accuracy
accr = model.evaluate(X_test.reshape((X_test.shape[0], n_steps, n_features)), Y_test)
# accr = model.evaluate(x.reshape((x.shape[0], n_steps, n_features)), y)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

#save model
model.save("forecastusefinalmodel8.h5")

y_pred = model.predict_classes(X_test, batch_size=batchSize, verbose=0)
# y_pred = model.predict_classes(x, batch_size=batchSize, verbose=0)
print('predicted', y_pred)
# print(Y_test[1])
# print(y[1])
y_labels = np.argmax(Y_test, axis=1)
# y_labels = np.argmax(y, axis=1)
print('labels: ',y_labels)#222
print('LSTM model accuracy:', (accuracy_score(y_labels, y_pred)) * 100)
cm = confusion_matrix(y_labels, y_pred)
print(cm)
print(classification_report(y_labels, y_pred))
plot_loss(history)
plot_acuuracy(history)

# demonstrate prediction
# x_input = array([[0.79159829, 0.12488033, 0.28393855, 0.22040424, 0.0960452, 0.05353869], [0.71250135, 0.30932417, 0.29223186, 0.23184345, 0.0653753, 0.05903924], [0.5529381, 0.59700839, 0.49490271, 0.26195661, 0.0645682, 0.03226989]])
model = load_model('forecastusefinalmodel8.h5')
scaler = load(open('Xscaler.pkl', 'rb'))
n_steps = 2
n_features = 6
x = list()
# x_input = array([[0.649621116, 0.117640873, 0.068406625, 0.136166887, 842, 58.94], [0.720222803, 0.110312405, 0.048005606, 0.099431187, 781, 58.94], [0.704010824, 0.126155956, 0.050516172, 0.097548184, 771.5, 41.94], [0.67233705, 0.138048457, 0.062574542, 0.104199826, 871.5, 42.525], [0.607857918, 0.153220842, 0.071482456, 0.139569143, 1021, 60.26], [0.641009128, 0.140567344, 0.067888631, 0.122307, 1002, 60.26]])
# x_input = array([[0.649621116, 0.117640873, 0.068406625, 0.136166887, 842, 58.94], [0.720222803, 0.110312405, 0.048005606, 0.099431187, 781, 58.94], [0.584858469, 0.184788643, 0.071793223, 0.126608392, 801, 62.17], [0.604487678, 0.164485051, 0.068961874, 0.135387766, 1020, 62.17]])
# x_input = array([[0.850180693, 0.065641971, 0.025659068, 0.046881322, 681, 40.32], [0.626368442, 0.158185173, 0.068583374, 0.118155092, 710.5, 40.32], [0.72715031, 0.089488312, 0.048673243, 0.112054672, 771, 42.67]])
x_input = array([[0.729211356, 0.135472394, 0.04666585400000001, 0.07155067, 681.5, 40.32], [0.850180693, 0.065641971, 0.025659067999999997, 0.046881322, 681.0, 40.32], [0.6263684420000001, 0.158185173, 0.068583374, 0.11815509199999999, 710.5, 40.32]])
X_scaler = scaler.transform(x_input)
for i in range(len(X_scaler)):
    # find the end of this pattern
    end_ix = i + n_steps
    # check if we are beyond the dataset
    if end_ix > len(X_scaler):
        break
    # gather input and output parts of the pattern
    seq_x = X_scaler[i:end_ix, :len(X_scaler[i])]
    x.append(seq_x)
# x_input = x.reshape((len(x_input), 2, 6))
# print(x)
for i in x:
    xin = i.reshape(1, n_steps, n_features)
    print('In next 10 seconds person\'s state will be...')
    ynew = model.predict(xin, verbose=0)
    labels = ['Awake', 'Moderate', 'Drowsy']
    print("Predicted vector: ", ynew , " Predicted Class: ", labels[np.argmax(ynew)])