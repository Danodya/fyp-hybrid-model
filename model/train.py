#acc 61 dat-cop(2)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,cross_val_score,KFold
# def train():
df = pd.read_csv("preprocessed (1).csv") #read training data set

# dataset = shuffle(df)
dataset = df
#d = dataset.head()
#X = dataset.iloc[:, 0:2].values  # first two cloumns are inputs for train model
X = dataset.iloc[:, 0:4].values
print(X)
y = dataset.iloc[:, -1].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0) #randomly select 20% as testing data set

sc = StandardScaler() # scale the data set
min_max_sc=MinMaxScaler()
# X_train = min_max_sc.fit_transform(X_train)
# X_test = min_max_sc.fit_transform(X_test)
X_st =sc.fit_transform(X)

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

classif = RandomForestClassifier(n_estimators=25, random_state=0) #train using random forest classifier
classif.fit(X_train, y_train)
y_pred = classif.predict(X_test) # predict the test data

print(y_pred)
print(y_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print((accuracy_score(y_test, y_pred))*100)

kfold = KFold(n_splits=10,shuffle=True)
results = cross_val_score(classif,X_train,y_train,cv=kfold)
print("Baseline:%.2f%%(%.2f%%)"%(results.mean()*100,results.std()*100))

# with open("randomfmodel.pkl", "wb") as file: #save model file
#     pickle.dump(classif, file)