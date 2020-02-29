import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout
import itertools
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score
from sklearn.preprocessing import StandardScaler
from collections import Counter
from imblearn.combine import SMOTETomek


METRICS = [
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.MeanSquaredError(name='mse'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),

]

es = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    verbose=1,
    patience=20,
    mode='min',
)
mc = ModelCheckpoint('best_model.h5', monitor='val_recall', mode='max', save_best_only=True, verbose=1)

#df = pd.read_csv('../data/part2.csv' )
#df = pd.read_csv('../data/part3.csv' )
df = pd.read_csv('../data/part4.csv' )


def variableSelection(df):
    dataset = df.values
    #Independent variables
    X = dataset[:,0:7]
    print(df['Flood_Status'])

    #Dependent variable
    print(dataset[:,7])
    Y = dataset[:,7]
    return X,Y

def stardardization():
    # Create the Scaler object
    scaler = preprocessing.StandardScaler()
    # Fit your data on the scaler object
    scaled_df = scaler.fit_transform(X)
    X_scaler = pd.DataFrame(scaled_df)
    return X_scaler

X, Y = variableSelection(df)

X_scaler = stardardization()

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scaler, Y, test_size=0.3)

X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

print("Before ReSampling, counts of label '1': {}".format(sum(Y_train==1)))
print("Before ReSampling, counts of label '0': {} \n".format(sum(Y_train==0)))

smote_tomek = SMOTETomek(random_state=0)

X_train_res, Y_train_res = smote_tomek.fit_resample(X_train, Y_train)
print(sorted(Counter(Y_train_res).items()))

print('After ReSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After ReSampling, the shape of train_Y: {} \n'.format(Y_train_res.shape))

print("After ReSampling, counts of label '1': {}".format(sum(Y_train_res==1)))
print("After ReSampling, counts of label '0': {}".format(sum(Y_train_res==0)))

def retrain(oldModel, newModel):
    # Recreate the exact same model, including its weights and the optimizer
    model = tf.keras.models.load_model(oldModel)

    # Show the model architecture
    model.summary()

    acc = model.evaluate(X_test, Y_test)[1]
    print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

    hist = model.fit(X_train_res, Y_train_res,
                     epochs=100,
                     callbacks=[es],
                     validation_data=(X_val, Y_val))

    model.save(newModel)

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

    model.evaluate(X_test, Y_test)[1]
    return model

#for part1
#model = retrain('model4.h5', "model.h5")

#for part2
#model = retrain('model.h5', "model2.h5")

#for part3
#model = retrain('model2.h5', "model3.h5")

#for part4
model = retrain('model3.h5', "model4.h5")


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting normalize=True.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

Y_train_pre = model.predict_classes(X_train)
print(Y_train_pre)
cnf_matrix_tra =confusion_matrix(Y_train, Y_train_pre)
print(cnf_matrix_tra)

print("Recall metric in the train dataset: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))
print("Precision metric in the train dataset: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[0,1]+cnf_matrix_tra[1,1])))

class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix')
plt.show()

Y_pre = model.predict_classes(X_test)
cnf_matrix = confusion_matrix(Y_test, Y_pre)

print("Recall metric in the testing dataset: {}%".format(100*cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])))
print("Precisiom metric in the testing dataset: {}%".format(100*cnf_matrix[1,1]/(cnf_matrix[0,1]+cnf_matrix[1,1])))

class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix , classes=class_names, title='Confusion matrix')
plt.show()

y_pred_sample_score = model.predict(X_test).ravel()
print(">>>>" ,y_pred_sample_score)
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_sample_score)

roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

Y_train_pre = model.predict_classes(X_train)
print(Y_train_pre)
cnf_matrix_tra =confusion_matrix(Y_train, Y_train_pre)
print(cnf_matrix_tra)

