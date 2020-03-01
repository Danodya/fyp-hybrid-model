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
import matplotlib.pyplot as plt

np.random.seed(7)

epochs = 200
batchSize = 150

def preprocess(df):
    # df = shuffle(data)
    dataset = df.values
    X = dataset[: ,[11,12,13,14,19,22]]
    # X = dataset[:, [0, 1, 2, 3, 4, 19, 22]]
    Y = dataset[: , 5]
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
    # model.add(Dense(32, input_dim=7, activation='sigmoid'))
    model.add(Dense(32, input_dim=6, activation='sigmoid'))
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

dataFrame = pandas.read_csv("../data/preprocessedNew.csv")
X_train, X_val_and_test, Y_train, Y_val_and_test, X_val, X_test, Y_val, Y_test = preprocess(dataFrame)

# input e.g. [10074.535, 2079.027, 828.732, 1558.949, 322.472, 639.5, 36.95]
def preprocess_inferring_data(data):
    array = []
    for d in data:
        array.append([d, 0, 0, 0, 0, 0, 0])
    return np.asarray(array)

def plot_history(history):
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

def retrieve_other_modalites():
    modelInput = []
    # TODO: integrate the modalities.
    return modelInput # e.g. [10074.535, 2079.027, 828.732, 1558.949, 322.472, 639.5, 36.95]

def run(scheme):
    if scheme.lower() == "test":
        modelPath = input("Enter a model path: ")
        model = load_model(str(modelPath))
        if (model ==  None):
            print("Invalid Path")
        else:
            print("Model loaded. Please enter an input with shape: " + str(model.layers[0].input_shape))
            # input to be inferred.
            pred = model.predict(preprocess_inferring_data([10074.535, 2079.027, 828.732, 1558.949, 322.472, 639.5, 36.95]))
            labels = ['Awake', 'Moderate', 'Drowsy']
            print("Predicted vector: ", pred , " Predicted Class: ", labels[np.argmax(pred)])
            # Do whatever
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
        plot_history(history)
    else:
        print("PLEASE ENTER A LEGIT SCHEME.")

if __name__ == '__main__':
    userInput = input("Please enter a scheme: ")
    run(userInput)