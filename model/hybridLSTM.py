# lstm model
import pandas
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical, np_utils
from matplotlib import pyplot


# load a single file as a numpy array
from seaborn import load_dataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# def load_file(filepath):
#     dataframe = read_csv(filepath, header=None, delim_whitespace=True)
#     return dataframe.values
dataframe = pandas.read_csv('../data/preprocessedNew.csv')
print(dataframe)
dataSet = dataframe.values
X = dataSet[:, [0,1,2,3,4,19,22]]
print(X)
print(len(X))
Y = dataSet[:,5]
print(Y)
dummy_y = np_utils.to_categorical(Y)

min_max_scaler = preprocessing.MinMaxScaler()
X_scaler = min_max_scaler.fit_transform(X)
print(X_scaler)

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scaler, dummy_y, test_size=0.3)

X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)


# # load a list of files and return as a 3d numpy array
# def load_group(filenames, prefix=''):
#     loaded = list()
#     for name in filenames:
#         data = load_file(prefix + name)
#         loaded.append(data)
#     # stack group so that features are the 3rd dimension
#     loaded = dstack(loaded)
#     return loaded
#
#
# # load a dataset group, such as train or test
# def load_dataset_group(group, prefix=''):
#     filepath = prefix + group + '/Inertial Signals/'
#     # load all 9 files as a single array
#     filenames = list()
#     # total acceleration
#     filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
#     # body acceleration
#     filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
#     # body gyroscope
#     filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
#     # load input data
#     X = load_group(filenames, filepath)
#     # load class output
#     y = load_file(prefix + group + '/y_' + group + '.txt')
#     return X, y
#
#
# # load the dataset, returns train and test X and y elements
# def load_dataset(prefix=''):
#     # load all train
#     trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
#     print(trainX.shape, trainy.shape)
#     # load all test
#     testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
#     print(testX.shape, testy.shape)
#     # zero-offset class values
#     trainy = trainy - 1
#     testy = testy - 1
#     # one hot encode y
#     trainy = to_categorical(trainy)
#     testy = to_categorical(testy)
#     print(trainX.shape, trainy.shape, testX.shape, testy.shape)
#     return trainX, trainy, testX, testy


# fit and evaluate a model
def evaluate_model(X_train, Y_train, X_test, Y_test):
    verbose, epochs, batch_size = 0, 100, 60
    n_timesteps, n_features, n_outputs = 1, 7, 3
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)
    return accuracy


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# run an experiment
def run_experiment(repeats=10):
    # load data
    # X_train, Y_train, X_test, Y_test =
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(X_train, Y_train, X_test, Y_test)
        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)


# run the experiment
run_experiment()