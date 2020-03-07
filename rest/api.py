from __future__ import print_function

import flask
import yaml
from keras.engine.saving import load_model
from pickle import load
from flask import request
import json
from threading import Thread

app = flask.Flask(__name__)

model = load_model("ann_relu_median2.h5")
scaler = load(open('Xscaler.pkl', 'rb'))

queue = dict()


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
    while True:
        if len(queue) == 6:
            # Consume from the queue one by one.
            print("QUEUE IS FULL. Train the model.")
            queue.clear()


# processes the arrived set of data
def consume():
    """
    Consumption method
    :return:
    """
    print("Consume")


with open("config.yaml", 'r') as stream:
    try:
        host = yaml.safe_load(stream)
    # TODO: Handle exceptions
    except yaml.YAMLError as exc:
        print(exc)


# Receives Data
@app.route("/eeg/data", methods=["POST"])
def predictEeg():
    req = request.data.decode("utf-8")
    data = json.loads(req)
    array = data.get('data')
    queue['eeg'] = array
    return array


# Receives Data
@app.route("/emg/data", methods=["POST"])
def predictEmg():
    req = request.data.decode("utf-8")
    data = json.loads(req)
    array = data.get('data')
    queue['emg'] = array
    return array


# Receives Data
@app.route("/ecg/data", methods=["POST"])
def predictEcg():
    req = request.data.decode("utf-8")
    data = json.loads(req)
    array = data.get('data')
    queue['ecg'] = array
    return array


# Spawns the worker thread
thread = listen()


app.run(host['host'])