import pandas as pd
import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.optimizers import adam, sgd
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from scipy import stats
import seaborn as sns
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold

# eeg_df = pd.read_csv('../../data/preprocessedNew.csv')
from sklearn.pipeline import Pipeline

dataframe = pd.read_csv('../../data/2class.csv')

print(dataframe)
dataSet = dataframe.values
# X = eegSet[:, [6,7,8,20,21,23,24]]
# X = eegSet[:, [6,7,8,19,22]]
X = dataSet[:, [0,1,2,3,4,5,8]]
print(X)
print(len(X))
Y = dataSet[:,11].astype(int)
print(Y)

# dummy_y = np_utils.to_categorical(Y)

min_max_scaler = preprocessing.MinMaxScaler()
X_scaler = min_max_scaler.fit_transform(X)
print(X_scaler)

# std_scaler = preprocessing.StandardScaler()
# X_scaler = std_scaler.fit_transform(X)
# print(X_scaler)

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scaler, Y, test_size=0.3)

X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

# optimizer = adam(lr=0.0001)
# define baseline model
# def baseline_model():
#     # create model
#     model = Sequential()
#     model.add(Dense(32, input_dim=7, activation='relu'))
#     model.add(Dropout(0.2))
#     # model.add(Dense(60, activation='relu'))
#     # model.add(Dropout(0.2))
#     model.add(Dense(3, activation='softmax'))
#     # Compile model
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=7, activation='relu'))
    model.add(Dropout(0.2))
    # model.add(Dense(60, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

# hist = baseline_model().fit(X_train, Y_train,
#                  batch_size=100, epochs=150,
#                  validation_data=(X_val, Y_val), callbacks=[EarlyStopping(monitor='val_loss', mode='min',  verbose=1,  patience=3, min_delta=0.001)])

hist = baseline_model().fit(X_train, Y_train,
                 batch_size=100, epochs=150,
                 validation_data=(X_val, Y_val))

estimator = KerasClassifier(build_fn=baseline_model, epochs=150, batch_size=100, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(estimator, X_scaler, Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

# evaluate = baseline_model().evaluate(X_test, Y_test) #correct one
# print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(evaluate[0],evaluate[1])) #correct one

# evaluate baseline model with standardized dataset
estimators = []
estimators.append(('standardize', min_max_scaler))
estimators.append(('mlp', KerasClassifier(build_fn=baseline_model, epochs=150, batch_size=100, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X_scaler, Y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

y_pred = baseline_model().predict_classes(X_test, batch_size=100, verbose=0)
# y_pred = baseline_model().fit(X_train, Y_train).predict(X_test)
print(y_pred[1])
print(Y_test[1])
# y_labels = np.argmax(Y_test, axis=1)
# print(y_labels[1])
print('ANN model accuracy:', (accuracy_score(Y_test, y_pred))*100)
cm = confusion_matrix(Y_test, y_pred)
print(cm)
print(classification_report(Y_test, y_pred))

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val'], loc='upper right')
# plt.show()
#
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val'], loc='lower right')
# plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()