# # importing necessary libraries
# import matplotlib
# import pandas as pd
# import numpy as np
# from sklearn.utils import shuffle
# from sklearn import datasets, preprocessing
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# import pickle
# import matplotlib.pyplot as plt
# # dataset = pd.read_csv("../data/modelinputwithmeanNew.csv")
# dataset = pd.read_csv('../data/preprocessedNew.csv')
# # print(dataset.describe())
# # dataset.hist(figsize=(6,6))
# # plt.figure()
# # dataset.plot.hist(alpha=0.3, stacked=True, bins=10)
# # dataset.plot.hist(alpha=0.3, stacked=True)
# # plt.show()
# # gca stands for 'get current axis'
# # ax = plt.gca()
# #
# # dataset.plot(kind='scatter',x='Delta Power',y='Class',ax=ax)
# # dataset.plot(kind='scatter',x='Theta Power',y='Class', color='red', ax=ax)
#
# # plt.show()
# # dataset.plot(kind='histogram')
# df = shuffle(dataset)
# # X = df.iloc[:, 0:7].values
# # X = df.iloc[:, [6,7,8,20,21,23,24]].values
# X = df.iloc[:, [6,7,8,19,22]].values
# print(X)
# Y = df.iloc[:, 5]
# y = df.iloc[:, 5].values
# print(y)
# #
# # # Y.value_counts().plot(kind='bar')
# # # plt.ylabel('Frequency')
# # # plt.xlabel('Drowsiness level')
# # # plt.title('Distribution')
# # #
# # # plt.show()
#
# min_max_scaler = preprocessing.MinMaxScaler()
# X_scaler = min_max_scaler.fit_transform(X)
# print(X_scaler)
#
# # dividing X, y into train and test data
# X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.2, random_state=0)
# print(y_test)
# # training a linear SVM classifier
# from sklearn.svm import SVC
#
# svm_model_linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
# svm_predictions = svm_model_linear.predict(X_test)
#
# # model accuracy for X_test
# accuracy = svm_model_linear.score(X_test, y_test)*100
# print('accuracy of the SVM model: ', accuracy)
#
# kfold = KFold(n_splits=10, shuffle=True)
# results = cross_val_score(SVC(kernel='linear', C=1), X_scaler, y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
#
# # creating a confusion matrix
# cm = confusion_matrix(y_test, svm_predictions)
#
# # with open("svm_model_linear3.pkl", "wb") as file:  # save model file
# #     pickle.dump(svm_model_linear, file)
#
# #print confusion matrix
# print(cm)
# #print classification report
# print(classification_report(y_test, svm_predictions))


# importing necessary libraries
import matplotlib
import pandas
import pandas as pd
import numpy as np
from keras import Sequential
from keras.engine.saving import load_model
from sklearn.utils import shuffle
from sklearn import datasets, preprocessing
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import pickle
import matplotlib.pyplot as plt
# dataset = pd.read_csv("../data/modelinputwithmeanNew.csv")
dataset = pd.read_csv('../data/preprocessedNew.csv')
# print(dataset.describe())
# dataset.hist(figsize=(6,6))
# plt.figure()
# dataset.plot.hist(alpha=0.3, stacked=True, bins=10)
# dataset.plot.hist(alpha=0.3, stacked=True)
# plt.show()
# gca stands for 'get current axis'
# ax = plt.gca()
#
# dataset.plot(kind='scatter',x='Delta Power',y='Class',ax=ax)
# dataset.plot(kind='scatter',x='Theta Power',y='Class', color='red', ax=ax)

# plt.show()
# dataset.plot(kind='histogram')
# df = shuffle(dataset)
# X = df.iloc[:, 0:7].values
# X = df.iloc[:, [6,7,8,20,21,23,24]].values
# X = df.iloc[:, [6,7,8,19,22]].values
# print(X)
# Y = df.iloc[:, 5]
# y = df.iloc[:, 5].values
# print(y)
#
# # Y.value_counts().plot(kind='bar')
# # plt.ylabel('Frequency')
# # plt.xlabel('Drowsiness level')
# # plt.title('Distribution')
# #
# # plt.show()

# min_max_scaler = preprocessing.MinMaxScaler()
# X_scaler = min_max_scaler.fit_transform(X)
# print(X_scaler)
#
# # dividing X, y into train and test data
# X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.2, random_state=0)
# print(y_test)
# # training a linear SVM classifier
from sklearn.svm import SVC

# svm_model_linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
# svm_predictions = svm_model_linear.predict(X_test)
#
# accuracy = svm_model_linear.score(X_test, y_test)*100
# print('accuracy of the SVM model: ', accuracy)
# # model accuracy for X_test
#
# kfold = KFold(n_splits=10, shuffle=True)
# results = cross_val_score(SVC(kernel='linear', C=1), X_scaler, y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
#
# # creating a confusion matrix
# cm = confusion_matrix(y_test, svm_predictions)

# with open("svm_model_linear3.pkl", "wb") as file:  # save model file
#     pickle.dump(svm_model_linear, file)

#print confusion matrix
# print(cm)
# #print classification report
# print(classification_report(y_test, svm_predictions))

####################
# epochs = 200
# batchSize = 150

def preprocess(df):
    # df = shuffle(data)
    dataset = df.values
    # X = dataset[: ,[11,12,13,14,19,22]]
    # X = dataset[:, [0, 1, 2, 3, 4, 19, 22]]
    X = dataset[:, [0, 1, 2, 3, 4, 19,22]]
    Y = dataset[: , 5]
    # dummy_y = np_utils.to_categorical(Y)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaler = min_max_scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaler, Y, test_size=0.2, random_state=0)
    print(X_train.shape)
    return X_train, X_test, y_train, y_test

def create_model(X_train, y_train):
    model = SVC(kernel='linear', C=1)
    return model

def train(X_train, y_train, model):
    return model.fit(X_train, y_train)
# def train(X_train, Y_train, model, epochs, batchSize):
#     return model.fit(X_train, Y_train,
#                  batch_size=batchSize, epochs=epochs,
#                  validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

dataFrame = pandas.read_csv("../data/preprocessedNew.csv")
X_train, X_test, y_train, y_test = preprocess(dataFrame)

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
        # model = load_model(str(modelPath))
        model = pickle.load(open(str(modelPath),'rb'))
        if (model ==  None):
            print("Invalid Path")
        else:
            print("Model loaded")
            # input to be inferred.
            # array = [14087.732, 2573.036, 927.917, 1803.045, 374.556, 641, 36.95]
            array = [7128.828,2629.557,1074.656,1879.437,410.078,941,62.02]
            pred = model.predict(preprocess_inferring_data(array))
            labels = ['Awake', 'Moderate', 'Drowsy']
            print("Predicted vector: ", pred , " Predicted Class: ", labels[np.argmax(pred)])
            # Do whatever
    elif scheme.lower() == "train":
        path = input("Enter a path to save the model: ")
        model = create_model(X_train, y_train)
        history = train(X_train, y_train, model)
        # history = train(X_train, Y_train, model, epochs, batchSize)
        accuracy = model.score(X_test, y_test) * 100
        print('accuracy of the SVM model: ', accuracy)
        y_pred = model.predict(X_test)
        # accr = model.evaluate(X_test, y_test)
        # print(model.summary())
        # print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
        # Save the trained model as a pickle string.
        pickle.dump(model,open(path, 'wb'))

        # Load the pickled model
        # knn_from_pickle = pickle.loads(saved_model)

        # Use the loaded pickled model to make predictions
        # knn_from_pickle.predict(X_test)
        # model.save(path)
        # y_pred = model.predict_classes(X_test, batch_size=batchSize, verbose=0)
        # print(y_pred[1])
        # print(Y_test[1])
        # y_labels = np.argmax(Y_test, axis=1)
        # print(y_labels[1])
        print('SVM model accuracy:', (accuracy_score(y_test, y_pred)) * 100)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(classification_report(y_test, y_pred))
        # plot_history(history)
        # accuracy = svm_model_linear.score(X_test, y_test) * 100
        # print('accuracy of the SVM model: ', accuracy)
    else:
        print("PLEASE ENTER A LEGIT SCHEME.")

if __name__ == '__main__':
    userInput = input("Please enter a scheme: ")
    run(userInput)