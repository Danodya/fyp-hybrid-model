import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as ts

#Independent variables
df = pd.read_csv('../data/finalSet3.csv')
dataset = df.values
X = dataset[:,0:4]
#X2 = dataset[:,2:4]
#X3 = np.concatenate([X1, X2])
print(X)

#Dependent variable
print(dataset[:,7])
Y = dataset[:,7]

min_max_scaler = preprocessing.MinMaxScaler()
X_scaler = min_max_scaler.fit_transform(X)
print(X_scaler)

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scaler, Y, test_size=0.3)

X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

#Note that ‘Dense’ refers to a fully-connected layer, which is what we will be using
model = Sequential([
    Dense(30, activation='relu', input_shape=(4,)),
    Dense(30, activation='relu'),
    Dense(1, activation='sigmoid'),])

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X_train, Y_train,
                 batch_size=30, epochs=100,
                 validation_data=(X_val, Y_val))

model.evaluate(X_test, Y_test)[1]

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