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
df = shuffle(dataset)
# X = df.iloc[:, 0:7].values
# X = df.iloc[:, [6,7,8,20,21,23,24]].values
X = df.iloc[:, [0,1,2,3,4,19,22]].values
print(X)
# Y = df.iloc[:, 5]
y = df.iloc[:, 5].values
print(y)
#
# # Y.value_counts().plot(kind='bar')
# # plt.ylabel('Frequency')
# # plt.xlabel('Drowsiness level')
# # plt.title('Distribution')
# #
# # plt.show()

min_max_scaler = preprocessing.MinMaxScaler()
X_scaler = min_max_scaler.fit_transform(X)
print(X_scaler)

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.2, random_state=0)
print(y_test)
# training a linear SVM classifier
from sklearn.svm import SVC

svm_model_linear = SVC(kernel='linear', C=1)
hist = svm_model_linear.fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)

# model accuracy for X_test
accuracy = svm_model_linear.score(X_test, y_test)*100
print('accuracy of the SVM model: ', accuracy)

# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)

with open("svm_model_linear3.pkl", "wb") as file:  # save model file
    pickle.dump(svm_model_linear, file)

# print confusion matrix
print(cm)
#print classification report
print(classification_report(y_test, svm_predictions))