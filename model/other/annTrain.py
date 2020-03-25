#import relevant libraries
from pickle import dump

import pandas
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.engine.saving import load_model
from keras.layers import Dense, Dropout
import numpy as np
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(7)

epochs = 200
batchSize = 150

def preprocess(df):
    dataset = df.values
    X = dataset[:, [6, 7, 8, 9, 19, 22]]
    Y = dataset[: , 5]
    dummy_y = np_utils.to_categorical(Y)
    print(dummy_y[1])
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaler = min_max_scaler.fit_transform(X)
    dump(min_max_scaler, open('scaler.pkl', 'wb'))
    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scaler, dummy_y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
    print(X_test[1])
    print(X_train.shape)
    return  X_train, X_val_and_test, Y_train, Y_val_and_test,  X_val, X_test, Y_val, Y_test

def create_model():
    model = Sequential()
    model.add(Dense(32, input_dim=6, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train(X_train, Y_train, model, epochs, batchSize):
    return model.fit(X_train, Y_train,
                 batch_size=batchSize, epochs=epochs,
                 validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

dataFrame = pandas.read_csv("../data/preprocessedNew.csv")
X_train, X_val_and_test, Y_train, Y_val_and_test, X_val, X_test, Y_val, Y_test = preprocess(dataFrame)

# input e.g. [10074.535, 2079.027, 828.732, 1558.949, 322.472, 639.5, 36.95]
# def preprocess_inferring_data(data):
#     array = []
#     for d in data:
#         array.append([d, 0, 0, 0, 0, 0, 0])
#     print(np.asarray(array))
#     return np.asarray(array)
def preprocess_inferring_data(data):
    array = []
    for d in data:
        array.append([d, 0, 0, 0, 0, 0, 0])
    print(np.asarray(array))
    print(np.asarray(array).shape)
    return np.asarray(array)

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

def retrieve_other_modalites():
    modelInput = []
    # TODO: integrate the modalities.
    return modelInput # e.g. [10074.535, 2079.027, 828.732, 1558.949, 322.472, 639.5, 36.95]


model = create_model()
history = train(X_train, Y_train, model, epochs, batchSize)
accr = model.evaluate(X_test, Y_test)
print(model.summary())
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
model.save("train.h5")

y_pred = model.predict_classes(X_test, batch_size=batchSize, verbose=0)
print(y_pred[1])
print(Y_test[1])
y_labels = np.argmax(Y_test, axis=1)
print(y_labels[1])
print('ANN model accuracy:', (accuracy_score(y_labels, y_pred)) * 100)
cm = confusion_matrix(y_labels, y_pred)
print(cm)
print(classification_report(y_labels, y_pred))
plot_loss(history)
plot_acuuracy(history)
