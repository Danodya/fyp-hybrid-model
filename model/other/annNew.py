# multi-class classification with Keras
import numpy as np
import pandas
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(7)

# load dataset
dataframe = pandas.read_csv("../data/preprocessedNew.csv")
# df = shuffle(dataframe)
dataset = dataframe.values
X = dataset[: ,[6, 7, 8, 9, 19, 22]]
Y = dataset[:, 5]
# print(X)
# print(Y)
# # encode class values as integers
# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(Y)
print(dummy_y[0])
min_max_scaler = preprocessing.MinMaxScaler()
X_scaler = min_max_scaler.fit_transform(X)
print(X_scaler)

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
    # model.add(Dense(60, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = baseline_model()

# hist = model.fit(X_train, Y_train,
#                  batch_size=batchSize, epochs=epochs,
#                  validation_data=(X_val, Y_val), callbacks=[EarlyStopping(monitor='val_loss', mode='min',  verbose=1,  patience=3, min_delta=0.001)])

model.save("MLPrel1.h5")

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

# evaluate = baseline_model().evaluate(X_test, Y_test)[1]
# evaluate = baseline_model().evaluate(X_test, Y_test) #correct one
# print('Loss:   Accuracy: ' (evaluate[0], evaluate[1]))
# print("Loss: %.2f%%  Accuracy: (%.2f%%)" % (evaluate[0], evaluate[1]))
# print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(evaluate[0],evaluate[1])) #correct one

y_pred = baseline_model().predict_classes(X_test, batch_size=batchSize, verbose=0)

print(y_pred[1])
print(Y_test[1])
y_labels = np.argmax(Y_test, axis=1)
print(y_labels[1])
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
