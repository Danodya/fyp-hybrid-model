from random import shuffle

import pandas
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Conv1D, MaxPooling1D
import numpy as np
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

np.random.seed(7)

epochs = 200
batchSize = 150

def preprocess(df):
    # df = shuffle(data)
    dataset = df.values
    X = dataset[: , 0:7]
    Y = dataset[: , 7]
    dummy_y = np_utils.to_categorical(Y)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaler = min_max_scaler.fit_transform(X)
    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scaler, dummy_y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
    print(X_train.shape)
    return  X_train, X_val_and_test, Y_train, Y_val_and_test,  X_val, X_test, Y_val, Y_test

def create_model():
    model = Sequential()
    model.add(Conv1D(32, input_shape=(465, 7, 1), activation='sigmoid', kernel_size=(2)))
    model.add(Conv1D(64, 2, activation='sigmoid'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train(X_train, Y_train, X_val, Y_val, model, epochs, batchSize):
    return model.fit(X_train, Y_train,
                 batch_size=batchSize, epochs=epochs,
                 validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

dataFrame = pandas.read_csv("../data/modelinputwithmeanNew.csv")
X_train, X_val_and_test, Y_train, Y_val_and_test, X_val, X_test, Y_val, Y_test =  preprocess(dataFrame)

model = create_model()

history =  train(X_train, Y_train, X_val, Y_val, model, epochs, batchSize)

accr = model.evaluate(X_test, Y_test)

print(model.summary())
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

model.save("CNN.h5")


