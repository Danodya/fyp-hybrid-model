
# multi-class classification with Keras
import numpy as np
import pandas
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import adam
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras as keras
from sklearn.utils import shuffle
np.random.seed(7)
# load dataset
dataframe = pandas.read_csv("../data/preprocessedNew.csv")
# df = shuffle(dataframe)
dataset = dataframe.values
# X = dataset[:, 0:7].astype(float)
# X = dataset[: , [1,2,3,5,6]]
X = dataset[: ,[6, 7, 8, 9, 19, 22]].astype(float)
Y = dataset[:, 5]
print(X)
# print(Y)
# # encode class values as integers
# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(Y)
print(dummy_y[0])
# min_max_scaler = preprocessing.MinMaxScaler()
# X_scaler = min_max_scaler.fit_transform(X)
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(X)
X_scaler = min_max_scaler.transform(X)
print(X_scaler)



# std_scaler = preprocessing.StandardScaler()
# X_scaler = std_scaler.fit_transform(X)
# print(X_scaler)

epochs = 200
batchSize = 150

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scaler, dummy_y, test_size=0.3)

X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

# optimizer = adam(lr=0.0001)
# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=6, activation='relu'))
    model.add(Dropout(0.2))
    # model.add(Dense(60, activation='sigmoid'))
    # model.add(Dropout(0.2))
    # model.add(Dense(60, activation='tahn'))
    # model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])
    return model

model = baseline_model()

# hist = model.fit(X_train, Y_train,
#                  batch_size=batchSize, epochs=epochs,
#                  validation_data=(X_val, Y_val), callbacks=[EarlyStopping(monitor='val_loss', mode='min',  verbose=1,  patience=3, min_delta=0.001)])

# model.save("MLPrel2.h5")

hist = model.fit(X_train, Y_train,
                 batch_size=batchSize, epochs=epochs,
                 validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


# estimator = KerasClassifier(build_fn=baseline_model, epochs=epochs, batch_size=batchSize, verbose=0)
# kfold = KFold(n_splits=10, shuffle=True, random_state=7)
# results = cross_val_score(estimator, X_scaler, dummy_y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

# # evaluate baseline model with standardized dataset
# estimators = []
# estimators.append(('standardize', min_max_scaler))
# estimators.append(('mlp', KerasClassifier(build_fn=baseline_model, epochs=600, batch_size=100, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = StratifiedKFold(n_splits=10, shuffle=True)
# results = cross_val_score(pipeline, X_scaler, dummy_y, cv=kfold)
# print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

y_pred = model.predict_classes(X_test, batch_size=batchSize, verbose=0)
# def train(X_train, Y_train, model, epochs, batchSize):
#     return model.fit(X_train, Y_train,
#                  batch_size=batchSize, epochs=epochs,
#                  validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
print(y_pred)
print(Y_test)
y_labels = np.argmax(Y_test, axis=1)
print(y_labels)
print('ANN model accuracy:', (accuracy_score(y_labels, y_pred))*100)
cm = confusion_matrix(y_labels, y_pred)
print(cm)
print(classification_report(y_labels, y_pred))

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

# array = [0.623895554, 0.111140939, 0.072458134, 0.157411484, 722, 60.335] #drowsy input for ann_relu_median2.h5
# array = [0.712715176, 0.130172962, 0.046944429, 0.091218199, 641, 36.95] #awake input for ann_relu_median.h5
# array = [0.557861696, 0.151026969, 0.088255769, 0.172795282, 880.5, 56.16] #modertate input for ann_relu_median .h5
# array = [0.336062623, 0.165285991, 0.115331128, 0.311817149, 981, 62.17]#drowsy input
# array = [0.452617331, 0.154536723, 0.097000164, 0.233127982, 862, 62.88] #drowsy
# array = [10074.535, 2079.027, 828.732, 1558.949, 322.472, 639.5, 36.95] #awake input for ANN_median.h5
# array = [5432.867, 2672.052, 1864.47, 5040.909, 1155.936, 981, 62.17] #drowsy input for ANN_median.h5
# Xnew = np.array(([[0.557861696, 0.151026969, 0.088255769, 0.172795282, 880.5, 56.16],[0.623895554, 0.111140939, 0.072458134, 0.157411484, 722, 60.335],[0.336062623, 0.165285991, 0.115331128, 0.311817149, 981, 62.17]]))
Xnew = np.array([[0.712715176, 0.130172962, 0.046944429, 0.091218199, 641, 36.95]])
X_scaler = min_max_scaler.transform(Xnew)
ynew = model.predict_classes(Xnew)
for i in range(len(Xnew)):
    print("X=%s, Predicted=%s" % (X_scaler[i], ynew[i]))
# print(X_scaler)
# array = np.asarray(X_scaler).reshape(1,6)
# pred = model.predict(array)
# pred = model.predict_classes(preprocess_inferring_data([7128.828,2629.557,1074.656,1879.437,410.078,941,62.02]), verbose=0)
# pred = model.predict(preprocess_inferringd_data(X_test[1]))
labels = ['Awake', 'Moderate', 'Drowsy']
# y_label = np.argmax(pred)
# print("Predicted vector: ", pred , " Predicted Class: ", labels[np.argmax(pred)])