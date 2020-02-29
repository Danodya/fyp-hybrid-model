#Accuracy 95, datapre2
# import numpy as np
# import pylab as pl
# import pandas as pd
# import matplotlib.pyplot as plt
# #%matplotlib inline
# import seaborn as sns
# from sklearn.utils import shuffle
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import confusion_matrix,classification_report
# from sklearn.model_selection import cross_val_score, GridSearchCV
# dataset = shuffle(pd.read_csv("data.csv"))
# train_outcome = pd.crosstab(index=dataset["Class"],  # Make a crosstab
#                               columns="count")      # Name the count column
#
# print(train_outcome)
# X = dataset.iloc[:, 0:5].values
# print(X)
# y = dataset.iloc[:, -1].values
# print(y)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #randomly select 20% as testing data set
# print("Dimension of Train set",X_train.shape)
# print("Dimension of Test set",X_test.shape,"\n")


# importing necessary libraries
import matplotlib
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn import datasets, preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import pickle
import matplotlib.pyplot as plt
dataset = pd.read_csv("../data/preprocessed - Copy.csv")
df = shuffle(dataset)
X = df.iloc[:, 0:2].values
print(X)
# Y = df.iloc[:, -1]
y = df.iloc[:, -1].values
print(y)

# Y.value_counts().plot(kind='bar')
# plt.ylabel('Frequency')
# plt.xlabel('Drowsiness level')
# plt.title('Distribution')
#
# plt.show()

min_max_scaler = preprocessing.MinMaxScaler()
X_scaler = min_max_scaler.fit_transform(X)
print(X_scaler)

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.2, random_state=0)
print(y_test)
# training a linear SVM classifier
from sklearn.svm import SVC

svm_model_linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)

# model accuracy for X_test
accuracy = svm_model_linear.score(X_test, y_test)*100
print('accuracy of the SVM model: ', accuracy)

kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(SVC(kernel='linear', C=1), X_scaler, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)

# with open("svm_model_linear3.pkl", "wb") as file:  # save model file
#     pickle.dump(svm_model_linear, file)

#print confusion matrix
print(cm)
#print classification report
print(classification_report(y_test, svm_predictions))