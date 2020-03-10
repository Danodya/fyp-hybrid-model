from __future__ import print_function

import flask
import yaml
from pickle import load
from flask import request
import json
from threading import Thread
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

app = flask.Flask(__name__)


def listen():
    """
    Spawns daemon threads
    :return: returns the spawned thread
    """
    thread = Thread(target=trigger)
    thread.daemon = True
    thread.start()
    print("+++++++ Thread Started +++++++")
    return thread


def trigger():
    """
    Triggers to consume from queue when the features are retrieved.
    :return:
    """
    global queue
    while True:
        if len(queue) == 3:
            # Consume from the queue one by one.
            print("QUEUE IS FULL. Pass to the model.")
            xtest = consume()
            queue.clear()
            # model = load_model("ann_relu_median2.h5")
            # scaler = load(open('Xscaler.pkl', 'rb'))
            Xnew = np.array([xtest])
            X_scaler = scaler.transform(Xnew)
            xtest.clear()
            print(X_scaler)
            global sess
            global graph
            with graph.as_default():
                set_session(sess)
                # session = tf.Session()
                # with session.as_default():
                # with tf.Graph().as_default():
                pred = model.predict(X_scaler)
                labels = ['Awake', 'Moderate', 'Drowsy']
                print("Predicted vector: ", pred, " Predicted Class: ", labels[np.argmax(pred)])


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
    for i in range(len(eeg_val)):
        Xnew.append(eeg_val[i])
    Xnew.append(ecg_val)
    Xnew.append(emg_val)
    print(Xnew)
    return Xnew


# Receives Data
@app.route("/eeg/data", methods=["POST"])
def predictEeg():
    print('EEG data: RECEIVED')
    req = request.data.decode("utf-8")
    data = json.loads(req)
    array = data.get('eeg')
    queue['eeg'] = array
    print(type(array))
    print(array)
    return {"SUCCESS": 200}


# Receives Data
@app.route("/emg/data", methods=["POST"])
def predictEmg():
    print('EMG data: RECEIVED')
    req = request.data.decode("utf-8")
    data = json.loads(req)
    array = data.get('emg')
    queue['emg'] = array
    print(type(array))
    print(array)
    return {"SUCCESS": 200}


# Receives Data
@app.route("/ecg/data", methods=["POST"])
def predictEcg():
    print('ECG data: RECEIVED')
    req = request.data.decode("utf-8")
    data = json.loads(req)
    array = data.get('ecg')
    queue['ecg'] = array
    print(type(array))
    print(array)
    return {"SUCCESS": 200}


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))

    sess = tf.Session()
    graph = tf.get_default_graph()
    set_session(sess)
    model = load_model("ann_relu_median2.h5")
    scaler = load(open('Xscaler.pkl', 'rb'))

    queue = dict()
    Xnew = []

    # Add threaded=False if you want to use keras instead of tensorflow.keras
    with open("config.yaml", 'r') as stream:
        try:
            host = yaml.safe_load(stream)
        # TODO: Handle exceptions
        except yaml.YAMLError as exc:
            print(exc)

    # Spawns the worker thread
    thread = listen()

    app.run(host['host'], port='5000')