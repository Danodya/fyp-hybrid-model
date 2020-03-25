# importing necessary libraries
from pickle import load

import pandas
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve

def preprocess(df):
    dataset = df.values
    X = dataset[:, [6, 7, 8, 9, 19, 22]]
    Y = dataset[: , 5]
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaler = min_max_scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaler, Y, test_size=0.2, random_state=0)
    print(X_train.shape)
    return X_train, X_test, y_train, y_test

def create_model(X_train, y_train):
    model = SVC(kernel='linear', C=1, gamma='auto')
    return model

def train(X_train, y_train, model):
    return model.fit(X_train, y_train)

#read dataframe using pandas
dataFrame = pandas.read_csv("../data/preprocessedNew.csv")
X_train, X_test, y_train, y_test = preprocess(dataFrame)

# # input e.g. [10074.535, 2079.027, 828.732, 1558.949, 322.472, 639.5, 36.95]
# def preprocess_inferring_data(data):
#     array = []
#     for d in data:
#         array.append([d, 0, 0, 0, 0, 0, 0])
#     return np.asarray(array)

def plot_history(history):
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

# def retrieve_other_modalites():
#     modelInput = []
#     # TODO: integrate the modalities.
#     return modelInput # e.g. [10074.535, 2079.027, 828.732, 1558.949, 322.472, 639.5, 36.95]

def run(scheme):
    if scheme.lower() == "test":
        modelPath = input("Enter a model path: ")
        # model = load_model(str(modelPath))
        model = pickle.load(open(str(modelPath),'rb'))
        scaler = load(open('Xscaler.pkl', 'rb'))
        if (model ==  None):
            print("Invalid Path")
        else:
            print("Model loaded")
            # input to be inferred.
            # array = [14087.732, 2573.036, 927.917, 1803.045, 374.556, 641, 36.95]
            array = [[7128.828,2629.557,1074.656,1879.437,941,62.02]]
            X_scaler = scaler.transform(array)
            print('Scaled input', X_scaler)
            pred = model.predict(X_scaler)
            labels = ['Awake', 'Moderate', 'Drowsy']
            if pred[0] == 0:
                print(" Predicted Class: ", labels[0])
            elif pred[0] == 1:
                print(" Predicted Class: ", labels[1])
            else:
                print(" Predicted Class: ", labels[2])
            # Do whatever
    elif scheme.lower() == "train":
        path = input("Enter a path to save the model: ")
        model = create_model(X_train, y_train)
        history = train(X_train, y_train, model)
        # create the Cross validation object
        loo = LeaveOneOut()

        # calculate cross validated (leave one out) accuracy score
        scores = cross_val_score(model, X_train, y_train, cv=loo, scoring='accuracy')

        print('Cross validated accuracy', scores.mean())
        # accuracy = model.score(X_test, y_test) * 100
        # print('accuracy of the SVM model: ', accuracy)
        y_pred = model.predict(X_test)
        pickle.dump(model,open(path, 'wb'))
        print('SVM model accuracy:', (accuracy_score(y_test, y_pred)) * 100)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(classification_report(y_test, y_pred))
        # plot_history(history)
    else:
        print("PLEASE ENTER A LEGIT SCHEME.")

if __name__ == '__main__':
    userInput = input("Please enter a scheme: ")
    run(userInput)