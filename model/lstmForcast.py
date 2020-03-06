# multivariate data preparation
import pandas
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l2
from keras.utils import np_utils
from numpy import array
from numpy import hstack
import matplotlib.pyplot as plt
import numpy as np

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tensorflow_core.python import confusion_matrix

# split a multivariate sequence into samples
def split_sequences(X_scaler, dummy_y, n_steps):
    x, y = list(), list()
    for i in range(len(X)-1):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(X)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = X[i:end_ix, :len(X[i])], dummy_y[end_ix, :len(dummy_y[i])]
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
X = dataset[:, [6, 7, 8, 9, 19, 22]]
Y = dataset[:, 5]
dummy_y = np_utils.to_categorical(Y)
print(X)
print(dummy_y[1])
min_max_scaler = preprocessing.MinMaxScaler()
X_scaler = min_max_scaler.fit_transform(X)
print(X_scaler)

# choose a number of time steps
n_steps = 3

# convert into input/output
x, y = split_sequences(X_scaler, dummy_y, n_steps)
print(x.shape, y.shape)

#train_test_validation split
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(x, y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

# summarize the data
for i in range(len(x)):
    print(x[i], y[i])
n_features = x.shape[2]
# print(n_features)

# define model
batchSize=100
epochs=200
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_steps, n_features), kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fit model
history = model.fit(x, y, epochs=epochs, batch_size=batchSize, verbose=0, validation_split=0.1)
print(X_test.shape)

#model summary
print(model.summary())

#model accuracy
accr = model.evaluate(X_test.reshape((X_test.shape[0], n_steps, n_features)), Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

#save model
model.save("lstm2.h5")

y_pred = model.predict_classes(X_test, batch_size=batchSize, verbose=0)
print(y_pred[1])
print(Y_test[1])
y_labels = np.argmax(Y_test, axis=1)
print(y_labels[1])
print('LSTM model accuracy:', (accuracy_score(y_labels, y_pred)) * 100)
cm = confusion_matrix(y_labels, y_pred)
print(cm)
print(classification_report(y_labels, y_pred))
plot_loss(history)
plot_acuuracy(history)

# demonstrate prediction
x_input = array([[0.79159829, 0.12488033, 0.28393855, 0.22040424, 0.0960452, 0.05353869], [0.71250135, 0.30932417, 0.29223186, 0.23184345, 0.0653753, 0.05903924], [0.5529381, 0.59700839, 0.49490271, 0.26195661, 0.0645682, 0.03226989]])
x_input = x_input.reshape((1, n_steps, n_features))
ynew = model.predict(x_input, verbose=0)
labels = ['Awake', 'Moderate', 'Drowsy']
print("Predicted vector: ", ynew , " Predicted Class: ", labels[np.argmax(ynew)])