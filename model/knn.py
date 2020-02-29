from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle

dataframe = pd.read_csv("../data/modelinputANN.csv") #read training data set
df = shuffle(dataframe)
#d = dataset.head()
#X = dataset.iloc[:, 0:2].values  # first two cloumns are inputs for train model
X = df.iloc[: , [1,2,3,5,6]].values
# X = df.iloc[:, 0:7].values
print(X)
y = df.iloc[:, -1].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) #randomly select 20% as testing data set

# sc = StandardScaler() # scale the data set
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

minmax = MinMaxScaler() # scale the data set
X_train = minmax.fit_transform(X_train)
X_test = minmax.transform(X_test)

model = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training sets
model.fit(X_train,y_train)

#Predict Output
y_pred= model.predict(X_test) # 0:Overcast, 2:Mild
print(y_pred)

print('model accuracy: ',(accuracy_score(y_test, y_pred))*100)

kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

