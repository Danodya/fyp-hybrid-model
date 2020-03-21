import threading
import time
import pandas
import pythoncom
from pickle import load
from threading import Thread
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
import win32com.client as wincl



def listen():
    """
    Spawns daemon threads
    :return: returns the spawned thread
    """
    lock = threading.Lock()

    thread1 = Thread(target=getEeg, args=(lock,))
    thread2 = Thread(target=getEcg, args=(lock,) )
    thread3 = Thread(target=getEmg, args=(lock,))
    thread4 = Thread(target=trigger, args=(lock,))
    # thread.daemon = True
    print("+++++++ Threads Started +++++++")
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    return thread1,thread2,thread3,thread4

def getEeg(lock):
    # global i
    for i in range(len(eegdataset)):
        lock.acquire()
        eeg = eegdataset[i, 0:4]
        queue['eeg'] = eeg
        # i = i+1
        print('EEG RECEIVED ', eeg)
        lock.release()
        time.sleep(10)

def getEcg(lock):
    # global j
    j = 0
    while j<len(ecgdataset):
        lock.acquire()
        # print('j value ', j)
        ecg = ecgdataset[j:j + 10, 0]
        queue['ecg'] = ecg
        j = j + 10
        print('ECG RECEIVED ', ecg)
        lock.release()
        time.sleep(10)

def getEmg(lock):
    # global k
    # print('global k is: ', k)
    k = 0
    while k < len(emgdataset):
        # print('k value ', k)
        lock.acquire()
        emg = emgdataset[k:k + 12, 0]
        queue['emg'] = emg
        k = k + 12
        print('EMG RECEIVED ', emg)
        lock.release()
        time.sleep(10)

def trigger(lock):
    """
    Triggers to consume from queue when the features are retrieved.
    :return:
    """

    global queue
    while True:
        if len(queue) == 3:
            lock.acquire()
            # Consume from the queue one by one.
            print("*****QUEUE IS FULL. Pass to the models*****")
            xtest, XtestLSTM = consume()
            queue.clear()
            if len(XtestLSTM) == 2:
                XnewLSTM = np.array(XtestLSTM)
                print('LSTM model input: ', XtestLSTM)
                X_scaler = scalerLSTM.transform(XnewLSTM)
                # print('LSTM scaled input: ', X_scaler)
                x_input = X_scaler.reshape((1, 2, 6))
                XtestLSTM.remove(XtestLSTM[0])
                # print(XtestLSTM)
                global sess2
                global graph2
                with graph2.as_default():
                    set_session(sess2)
                    pred = modelLSTM.predict(x_input)
                    labels = ['Awake', 'Moderate', 'Drowsy']
                    print("LSTM Predicted vector: ", pred, " LSTM Predicted Class: ", labels[np.argmax(pred)])
            xANN = np.array([xtest])
            X_scaler = scaler.transform(xANN)
            # xtest.clear()
            xtest = []
            # print('ANN scaled input: ', X_scaler)
            global sess
            global graph
            with graph.as_default():
                set_session(sess)
                pred = model.predict(X_scaler)
                labels = ['Awake', 'Moderate', 'Drowsy']
                print("ANN Predicted vector: ", pred, " ANN Predicted Class: ", labels[np.argmax(pred)])
                print('-------------------------------------------------------------------------------------------------------')
                pythoncom.CoInitialize()
                speak = wincl.Dispatch("SAPI.SpVoice")
                if labels[np.argmax(pred)] == 'Awake':
                    speak.Speak("person is awake")
                elif labels[np.argmax(pred)] == 'Moderate':
                    speak.Speak("person is moderately drowsy")
                else:
                    speak.Speak("person is drowsy")
            lock.release()


# processes the arrived set of data
def consume():
    """
    Consumption method
    :return:
    """

    global queue
    eeg_val = queue['eeg']
    emg_val = queue['emg']
    emg_val = np.median(emg_val)
    # print(emg_val)
    ecg_val = queue['ecg']
    ecg_val = np.median(ecg_val)
    # print(ecg_val)
    # Xnew = eeg_val + ecg_val + emg_val
    global Xnew
    Xnew = []
    global XnewLSTM
    # print('XnewLSTM', XnewLSTM)
    for i in range(len(eeg_val)):
        Xnew.append(eeg_val[i])
    Xnew.append(ecg_val)
    Xnew.append(emg_val)
    XnewLSTM.append(Xnew)
    print('ANN model input: ', Xnew)
    # print('after appending', XnewLSTM)
    return Xnew, XnewLSTM

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))

    eegFrame = pandas.read_csv("../data/data-preprocess/eeg1.csv")
    ecgFrame = pandas.read_csv("../data/data-preprocess/ecg2.csv")
    emgFrame = pandas.read_csv("../data/data-preprocess/emg1.csv")
    # X_train, X_val_and_test, Y_train, Y_val_and_test, X_val, X_test, Y_val, Y_test = preprocess(dataFrame)

    eegdataset = eegFrame.values
    ecgdataset = ecgFrame.values
    emgdataset = emgFrame.values
    sess = tf.Session()
    graph = tf.get_default_graph()
    set_session(sess)
    model = load_model("../model/ann_relu_median2.h5")
    scaler = load(open('../model/Xscaler.pkl', 'rb'))
    sess2 = tf.Session()
    graph2 = tf.get_default_graph()
    set_session(sess2)
    modelLSTM = load_model("../model/forecastusefinalmodel7.h5")
    scalerLSTM = load(open('../model/Xscaler.pkl', 'rb'))
    queue = dict()
    Xnew = []
    XnewLSTM = []

    # Spawns the worker thread
    thread1,thread2,thread3,thread4 = listen()
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
