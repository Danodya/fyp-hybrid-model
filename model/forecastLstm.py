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
def preprocess(dataFrame):
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

    return X_train, Y_train, X_test, Y_test

# define model
def create_model():
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
    return model

def train(X_train, Y_train, model, epochs, batchSize):
    return model.fit(X_train, Y_train, epochs=epochs, batch_size=batchSize, verbose=0, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=15, min_delta=0.0001, verbose=1)])

def run(scheme):
    if scheme.lower() == "test":
        modelPath = input("Enter a model path: ")
        model = load_model(str(modelPath))
        scaler = load(open('Xscaler.pkl', 'rb'))
        if (model ==  None):
            print("Invalid Path")
        else:
            x_input = np.array(([0.729211356, 0.135472394, 0.04666585400000001, 0.07155067, 681.5, 40.32], [0.850180693, 0.065641971, 0.025659067999999997, 0.046881322, 681.0, 40.32], [0.6263684420000001, 0.158185173, 0.068583374, 0.11815509199999999, 710.5, 40.32]))
            X_scaler = scaler.transform(x_input)
            x = list()
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
                print("Predicted vector: ", ynew, " Predicted Class: ", labels[np.argmax(ynew)])

    elif scheme.lower() == "train":
        path = input("Enter a path to save the model: ")
        dataFrame = pandas.read_csv("../data/preprocessedNew.csv")
        X_train, Y_train, X_test, Y_test = preprocess(dataFrame)
        model = create_model()
        history = train(X_train, Y_train, model, epochs, batchSize)
        accr = model.evaluate(X_test, Y_test)
        print(model.summary())
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
        model.save(path)
        y_pred = model.predict_classes(X_test, batch_size=batchSize, verbose=0)
        y_labels = np.argmax(Y_test, axis=1)
        print('LSTM model accuracy:', (accuracy_score(y_labels, y_pred)) * 100)
        cm = confusion_matrix(y_labels, y_pred)
        print(cm)
        print(classification_report(y_labels, y_pred))
        plot_loss(history)
        plot_acuuracy(history)

    else:
        print("PLEASE ENTER A LEGIT SCHEME.")

if __name__ == '__main__':
    batchSize = 100
    epochs = 200
    n_steps = 2
    n_features = 6
    userInput = input("Please enter a scheme: ")
    run(userInput)