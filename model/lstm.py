# LSTM for sequence classification in the IMDB dataset
import numpy
import pandas
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

numpy.random.seed(7)

# load dataset
dataframe = pandas.read_csv("../data/modelinputANN.csv")
df = shuffle(dataframe)
dataset = df.values
X = dataset[:, 0:7].astype(float)
Y = dataset[:, 7]

min_max_scaler = preprocessing.MinMaxScaler()
X_scaler = min_max_scaler.fit_transform(X)
print(X_scaler)

X_train, X_test, y_train, y_test = train_test_split(X_scaler, Y, test_size=0.2, random_state=0)
# # load the dataset but only keep the top n words, zero the rest
# top_words = len(X_scaler)
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
# max_review_length =6
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 6
model = Sequential()
# model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dense(40, input_dim=7, activation='relu'))
model.add(LSTM(100))
model.add(Dense(1, activation='relu'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))