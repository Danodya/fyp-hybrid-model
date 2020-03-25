# importing necessary libraries
from pickle import load

import matplotlib
import pandas
import pandas as pd
import numpy as np
from keras import Sequential
from keras.engine.saving import load_model
from sklearn.utils import shuffle
from sklearn import datasets, preprocessing
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve, LeaveOneOut
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# dataset = pd.read_csv("../data/modelinputwithmeanNew.csv")
# dataset = pd.read_csv('../data/preprocessedNew.csv')

def preprocess(X, y):
    # df = shuffle(dataset)
    # X = df.iloc[:, [6, 7, 8, 9, 19, 22]].values
    # print(X)
    # y = df.iloc[:, 5].values
    # print(y)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaler = min_max_scaler.fit_transform(X)
    print(X_scaler)

    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test



# # loo = LeaveOneOut()
# # train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel='rbf'), X, y, train_sizes=[10, 50, 80, 110, 500], cv=10)
# train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel='rbf'), X, y, train_sizes=np.linspace(0.01, 1.0, 50), cv=10)

def draw_learning_curve(X, y):
    train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel='rbf'), X, y, train_sizes=[10, 50, 80, 110, 500], cv=10)
    print('train_scores', train_scores)
    print('valid_scores', valid_scores)

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)

    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, valid_mean, color="#111111", label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue')
    plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, color="#DDDDDD")

    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

# svm_model_linear = SVC(kernel='linear', C=1)
# hist = svm_model_linear.fit(X_train, y_train)
# svm_predictions = svm_model_linear.predict(X_test)

def create_model():
    svm_model_linear = SVC(kernel='rbf', C=100, gamma=1)
    return svm_model_linear

def train(X_train, y_train, model):
    history = model.fit(X_train, y_train)
    return history

def reports(X, y):
    kernals = ['Polynomial', 'RBF', 'Sigmoid', 'linear']

    # A function which returns the corresponding SVC model
    def getClassifier(ktype):
        if ktype == 0:
            # Polynomial kernal
            return SVC(kernel='poly', degree=8, gamma="auto")
        elif ktype == 1:
            # Radial Basis Function kernal
            return SVC(kernel='rbf', gamma="auto")
        elif ktype == 2:
            # Sigmoid kernal
            return SVC(kernel='sigmoid', gamma="auto")
        elif ktype == 3:
            # Linear kernal
            return SVC(kernel='linear', gamma="auto")


    for i in range(4):
        # Separate data into test and training sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

        # Train a SVC model using different kernal
        svclassifier = getClassifier(i)
        svclassifier.fit(X_train, y_train)

        # Make prediction
        y_pred = svclassifier.predict(X_test)

        # Evaluate our model
        print("Evaluation:", kernals[i], "kernel")
        print(classification_report(y_test,y_pred))
#
def gridSearch(X_train,y_train, X_test, y_test):
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]}
    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
    grid.fit(X_train,y_train)
    print('best estimator', grid.best_estimator_)
    grid_predictions = grid.predict(X_test)
    print(confusion_matrix(y_test,grid_predictions))
    print(classification_report(y_test,grid_predictions))

def run(scheme):
    if scheme.lower() == "test":
        modelPath = input("Enter a model path: ") #use svm_model_rbf2.pkl
        model = pickle.load(open(str(modelPath),'rb'))
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
                if pred[0] == 0:
                    print(" Predicted Class: ", labels[0])
                elif pred[0] == 1:
                    print(" Predicted Class: ", labels[1])
                else:
                    print(" Predicted Class: ", labels[2])


    elif scheme.lower() == "train":
        path = input("Enter a path to save the model: ")
        dataset = pd.read_csv('../data/preprocessedNew.csv')
        df = shuffle(dataset)
        X = df.iloc[:, [6, 7, 8, 9, 19, 22]].values
        print(X.shape)
        y = df.iloc[:, 5].values
        X_train, X_test, y_train, y_test = preprocess(X, y)

        #draw learning curve
        draw_learning_curve(X, y)

        #classification reports for different kernels
        # reports(X, y)

        #finding the best optimizer
        gridSearch(X_train, y_train, X_test, y_test)

        model = create_model()
        train(X_train, y_train, model)

        svm_predictions = model.predict(X_test)

        # model accuracy for X_test
        accuracy = model.score(X_test, y_test) * 100
        print('accuracy of the SVM model: ', accuracy)

        # creating a confusion matrix
        cm = confusion_matrix(y_test, svm_predictions)

        with open(path, "wb") as file:  # save model file "svm_model_rbf2.pkl"
            pickle.dump(model, file)

        # print confusion matrix
        print(cm)
        # print classification report
        print(classification_report(y_test, svm_predictions))

    else:
        print("PLEASE ENTER A LEGIT SCHEME.")

if __name__ == '__main__':
    userInput = input("Please enter a scheme: ")
    run(userInput)