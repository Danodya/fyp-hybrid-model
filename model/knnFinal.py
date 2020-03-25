import pickle
from pickle import load

from matplotlib import rcParams
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_correlation(dataframe):
    '''
    plot correlation's matrix to explore dependency between features
    '''
    # init figure size
    # rcParams['figure.figsize'] = 15, 20
    fig = plt.figure()
    sns.heatmap(dataframe.corr(), annot=True, fmt=".2f")
    plt.show()
    # fig.savefig('corr.png')

def plot_densities(data, names):
    '''
    Plot features densities depending on the outcome values
    '''
    # change fig size to fit all subplots beautifully
    rcParams['figure.figsize'] = 10,5

    # separate data based on outcome values
    outcome_0 = data[data['Class'] == 0]
    outcome_1 = data[data['Class'] == 1]
    outcome_2 = data[data['Class'] == 2]

    # init figure
    fig, axs = plt.subplots(6, 1)
    # fig.suptitle('Features densities for different outcomes 0/1/2')
    plt.subplots_adjust(left = 0.2, right = 0.9, bottom = 0.1, top = 0.9,
                        wspace = 0.2, hspace = 0.9)

    # plot densities for outcomes
    # name = ['Ralpha', 'Rbeta', 'MedianNN', 'MedianX', 'Class']
    for column_name in names[:-1]:
        ax = axs[names.index(column_name)]
        #plt.subplot(4, 2, names.index(column_name) + 1)
        outcome_0[column_name].plot(kind='density', ax=ax, subplots=True,
                                    sharex=False, color="red", legend=True,
                                    label=column_name + ' for Class = 0')
        outcome_1[column_name].plot(kind='density', ax=ax, subplots=True,
                                     sharex=False, color="green", legend=True,
                                     label=column_name + ' for Class = 1')
        outcome_2[column_name].plot(kind='density', ax=ax, subplots=True,
                                    sharex=False, color="blue", legend=True,
                                    label=column_name + ' for Class = 2')
        ax.set_xlabel(column_name + ' values')
        ax.set_title(column_name + ' density')
        ax.grid('on')
        ax.legend(prop={'size': 5.5})
    plt.show()
    # fig.savefig('densities.png')


def preprocess(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) #randomly select 30% as testing data set

    minmax = MinMaxScaler() # scale the data set
    X_train = minmax.fit_transform(X_train)
    X_test = minmax.transform(X_test)
    return X_train, X_test, y_train, y_test

def optimalk(X_train, y_train):
    # creating odd list of K for KNN
    neighbors = list(range(1, 50, 2))

    # empty list that will hold cv scores
    cv_scores = []

    # perform 10-fold cross validation
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

    # changing to misclassification error
    mse = [1 - x for x in cv_scores]

    # determining best k
    optimal_k = neighbors[cv_scores.index(max(cv_scores))]
    print("The optimal number of neighbors from accuracy is {}".format(optimal_k))

    optimal_k_mse = neighbors[mse.index(min(mse))]
    print("The optimal number of neighbors from mse is {}".format(optimal_k_mse))
    return  neighbors,cv_scores,mse, optimal_k

def plot_cvaccuracy(neighbors, cv_scores):
    # plot Cross Validation accuracy vs k
    plt.plot(neighbors, cv_scores)
    plt.xlabel("Number of Neighbors K")
    plt.ylabel("Cross Validation accuracy")
    plt.grid('on')
    plt.show()

def plot_mse(neighbors, mse):
    # plot misclassification error vs k
    plt.plot(neighbors, mse)
    plt.xlabel("Number of Neighbors K")
    plt.ylabel("Misclassification Error")
    plt.grid('on')
    plt.show()

def creat_model(optimal_k):
    model = KNeighborsClassifier(n_neighbors=optimal_k)
    return model

# Train the model using the training sets
def train(model, X_train, y_train):
    model.fit(X_train,y_train)

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
        dataframe = pd.read_csv("../data/model_data_final.csv") #read training data set
        names = list(dataframe.columns)
        print(names)
        df = shuffle(dataframe)
        X = df.iloc[: , 0:6].values
        print(X)
        y = df.iloc[:, 6].values
        print(y)
        X_train, X_test, y_train, y_test = preprocess(X, y)

        ## plot correlation & densities
        plot_densities(dataframe, names)

        # plot correlation & densities
        plot_correlation(dataframe)


        #classification reports for different kernels
        neighbors,cv_scores,mse, optimal_k =optimalk(X_train, y_train)

        # plot Cross Validation accuracy vs k
        plot_cvaccuracy(neighbors, cv_scores)

        # plot misclassification error vs k
        plot_mse(neighbors, mse)

        model = creat_model(optimal_k)
        train(model, X_train, y_train)

        with open(path, "wb") as file:  # save model file "knnfinalnew.pkl"
            pickle.dump(model, file)

        # Predict Output
        y_pred = model.predict(X_test)
        # print(y_pred)

        print('KNN model accuracy: ', (accuracy_score(y_test, y_pred)) * 100)

        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(classification_report(y_test, y_pred))

    else:
        print("PLEASE ENTER A LEGIT SCHEME.")

if __name__ == '__main__':
    userInput = input("Please enter a scheme: ")
    run(userInput)