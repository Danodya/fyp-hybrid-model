#import relevant libraries
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
# import win32com.client as wincl

np.random.seed(7)

epochs = 200
batchSize = 150

def preprocess(df):
    dataset = df.values
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

def run(scheme):
    if scheme.lower() == "test":
        modelPath = input("Enter a model path: ")
        model = load_model(str(modelPath))
        scaler = load(open('Xscaler.pkl', 'rb'))
        if (model ==  None):
            print("Invalid Path")
        else:
            # print("Model loaded. Please enter an input with shape: " + str(model.layers[0].input_shape))
            # input to be inferred.
            # pred = model.predict(preprocess_inferring_data([10074.535, 2079.027, 828.732, 1558.949, 322.472, 639.5, 36.95]))
            # array = [0.623895554, 0.111140939, 0.072458134, 0.157411484, 722, 60.335] #drowsy input for ann_relu_median2.h5
            # array = [0.712715176, 0.130172962, 0.046944429, 0.091218199, 641, 36.95] #awake input for ann_relu_median2.h5
            # array = [0.67233705, 0.138048457, 0.062574542, 0.104199826, 871.5, 42.525] #modertate input for ann_relu_median .h5
            # array = [0.336062623, 0.165285991, 0.115331128, 0.311817149, 981, 62.17]#drowsy input
            # array = [0.452617331, 0.154536723, 0.097000164, 0.233127982, 862, 62.88] #drowsy
            # array = [10074.535, 2079.027, 828.732, 1558.949, 322.472, 639.5, 36.95] #awake input for ANN_median.h5
            # array = [5432.867, 2672.052, 1864.47, 5040.909, 1155.936, 981, 62.17] #drowsy input for ANN_median.h5
            # array = [0.52087612, 0.41389001, 0.50340108, 0.41204823, 0.79015335, 0.97323066] #drowsy scaled
            # array = [0.71250135, 0.30932417, 0.29223186, 0.23184345, 0.0653753,  0.05903924] #awake scaled
            # Xnew = np.array(([[0.557861696, 0.151026969, 0.088255769, 0.172795282, 880.5, 56.16],[0.623895554, 0.111140939, 0.072458134, 0.157411484, 722, 60.335],[0.336062623, 0.165285991, 0.115331128, 0.311817149, 981, 62.17]]))
            Xnew = np.array([[0.712715176, 0.130172962, 0.046944429, 0.091218199, 641, 36.95]])
           # array = np.asarray(array).reshape(1,6)
            X_scaler = scaler.transform(Xnew)
            print(X_scaler)
            # make a prediction
            pred = model.predict(X_scaler)
            # pred = model.predict(preprocess_inferringd_data(X_test[1]))
            labels = ['Awake', 'Moderate', 'Drowsy']
            print("Predicted vector: ", pred , " Predicted Class: ", labels[np.argmax(pred)])
            # # speak = wincl.Dispatch("SAPI.SpVoice")
            # if labels[np.argmax(pred)] == 'Awake':
            #     speak.Speak("person is awake")
            # elif labels[np.argmax(pred)] == 'Moderate':
            #     speak.Speak("person is moderately drowsy")
            # else:
            #     speak.Speak("person is drowsy")
            # Do whatever
            # for i in range(len(Xnew)):
            #     print("X=%s, Predicted=%s" % (X_scaler[i], pred[i]))

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