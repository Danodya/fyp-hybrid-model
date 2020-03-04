#import relevant libraries
from pickle import load

import pandas
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.engine.saving import load_model
from keras.layers import Dense, Dropout
import numpy as np
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# from annTrain import min_max_scaler


model = load_model("train.h5")
scaler = load(open('scaler.pkl', 'rb'))
Xnew = np.array([[0.712715176, 0.130172962, 0.046944429, 0.091218199, 641, 36.95]])
print(Xnew)
# min_max_scaler = preprocessing.MinMaxScaler()
xaa = scaler.transform(Xnew)
# xaa = xaa.reshape(xaa.shape +(1,))
print("print FeaturesTest scalled: ")
print(xaa)
ynew = model.predict_classes(xaa)
for i in range(len(Xnew)):
    print("X=%s, Predicted=%s" % (xaa[i], ynew[i]))