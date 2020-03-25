#import relevant libraries
import time
from pickle import dump, load

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
import win32com.client as wincl
from sklearn.utils import shuffle

np.random.seed(7)

epochs = 200
batchSize = 150

def preprocess(df):
    dataset = df.values
    # dataset = shuffle(dataset)
    X = dataset[:, [6, 7, 8, 9, 19, 22]]
    Y = dataset[:, 5]
    dummy_y = np_utils.to_categorical(Y)
    print(dummy_y[1])
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaler = min_max_scaler.fit_transform(X)
    dump(min_max_scaler, open('Xscaler.pkl', 'wb'))
    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scaler, dummy_y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
    print(X_test[1])
    print(X_train.shape)
    return  X_train, X_val_and_test, Y_train, Y_val_and_test,  X_val, X_test, Y_val, Y_test

def create_model():
    model = Sequential()
    model.add(Dense(32, input_dim=6, activation='relu'))
    model.add(Dropout(0.2))
    # model.add(Dense(20, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train(X_train, Y_train, model, epochs, batchSize):
    return model.fit(X_train, Y_train,
                 batch_size=batchSize, epochs=epochs,
                 validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001, verbose=1)])

dataFrame = pandas.read_csv("../data/preprocessedNew.csv")
X_train, X_val_and_test, Y_train, Y_val_and_test, X_val, X_test, Y_val, Y_test = preprocess(dataFrame)


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

def run(scheme):
    if scheme.lower() == "test":
        modelPath = input("Enter a model path: ")
        model = load_model(str(modelPath))
        scaler = load(open('Xscaler.pkl', 'rb'))
        if (model ==  None):
            print("Invalid Path")
        else:
            Xnew = np.array(([0.712715176, 0.130172962, 0.046944429, 0.091218199, 641, 36.95],[0.704010824, 0.126155956, 0.050516172, 0.097548184, 771.5, 41.94], [0.607857918, 0.153220842, 0.071482456, 0.139569143, 1021, 60.26]))
            for i in range(len(Xnew)):
                array = np.asarray(Xnew[i]).reshape(1, 6)
                X_scaler = scaler.transform(array)
                print('Scaled input', X_scaler)
                pred = model.predict(X_scaler)
                labels = ['Awake', 'Moderate', 'Drowsy']
                print("Predicted vector: ", pred, " Predicted Class: ", labels[np.argmax(pred)])
                speak = wincl.Dispatch("SAPI.SpVoice")
                if labels[np.argmax(pred)] == 'Awake':
                    speak.Speak("person is awake")
                elif labels[np.argmax(pred)] == 'Moderate':
                    speak.Speak("person is moderately drowsy")
                else:
                    speak.Speak("person is drowsy")
                # time.sleep(3)
                # print("X=%s, Predicted=%s" % (X_scaler[i], pred[i]))

    elif scheme.lower() == "train":
        path = input("Enter a path to save the model: ")
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
        plot_loss(history)
        plot_acuuracy(history)

    else:
        print("PLEASE ENTER A LEGIT SCHEME.")

if __name__ == '__main__':
    userInput = input("Please enter a scheme: ")
    run(userInput)