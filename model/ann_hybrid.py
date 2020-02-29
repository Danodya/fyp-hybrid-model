from random import shuffle

import pandas
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.engine.saving import load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Conv1D, MaxPooling1D
import numpy as np
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
    model = Sequential()
    model.add(Dense(32, input_dim=7, activation='sigmoid'))
    model.add(Dropout(0.2))
    # model.add(Dense(60, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train(X_train, Y_train, model, epochs, batchSize):
    return model.fit(X_train, Y_train,
                 batch_size=batchSize, epochs=epochs,
                 validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

dataFrame = pandas.read_csv("../data/modelinputwithmeanNew.csv")
X_train, X_val_and_test, Y_train, Y_val_and_test, X_val, X_test, Y_val, Y_test =  preprocess(dataFrame)

def run(scheme):
    if scheme == "test":
        model = load_model("CNN_2.h5")
        if (model ==  None):
            print("Invalid Path")
        else:
            print("Model loaded.")
            # Do whatever
    elif scheme == "train":
        path = input("Enter a path to save the model:")
        model = create_model()
        history = train(X_train, Y_train, model, epochs, batchSize)
        accr = model.evaluate(X_test, Y_test)
        print(model.summary())
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
        model.save(path)
        y_pred = model.predict_classes(X_test, batch_size=batchSize, verbose=0)
        print(y_pred[1])
        print(Y_test[1])
        y_labels = np.argmax(Y_test, axis=1)
        print(y_labels[1])
        print('ANN model accuracy:', (accuracy_score(y_labels, y_pred)) * 100)
        cm = confusion_matrix(y_labels, y_pred)
        print(cm)
        print(classification_report(y_labels, y_pred))
    else:
        print("PLEASE ENTER A FUCKING LEGIT SCHEME.")

if __name__ == '__main__':
    userInput = input("Please enter a scheme: ")
    run(userInput)