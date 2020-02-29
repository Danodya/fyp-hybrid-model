import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# def train():
from sklearn.utils import shuffle

dataframe = pd.read_csv("../data/modelinputANN.csv") #read training data set
df = shuffle(dataframe)
#d = dataset.head()
#X = dataset.iloc[:, 0:2].values  # first two cloumns are inputs for train model
# X = df.iloc[: , [1,2,3,5,6]].values
X = df.iloc[:, 0:7].values
print(X)
y = df.iloc[:, -1].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) #randomly select 20% as testing data set

sc = StandardScaler() # scale the data set
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# minmax = MinMaxScaler() # scale the data set
# X_train = minmax.fit_transform(X_train)
# X_test = minmax.transform(X_test)

classif = RandomForestClassifier(n_estimators=10, random_state=1) #train using random forest classifier
classif.fit(X_train, y_train)
y_pred = classif.predict(X_test) # predict the test data

# print(y_pred)
# print(y_test)
print('model accuracy: ',(accuracy_score(y_test, y_pred))*100)

# estimator = KerasClassifier(build_fn=classif, epochs=100, batch_size=30, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(classif, X, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))