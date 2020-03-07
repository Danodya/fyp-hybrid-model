import flask
import yaml
from keras.engine.saving import load_model
from pickle import load
from flask import request, jsonify
import json

app = flask.Flask(__name__)

model = load_model("ann_relu_median2.h5")
scaler = load(open('Xscaler.pkl', 'rb'))

with open("config.yaml", 'r') as stream:
    try:
        host = yaml.safe_load(stream)
    # TODO: Handle exceptions
    except yaml.YAMLError as exc:
        print(exc)


def load_models(model):
    return load_model(model)


# Receives Data
@app.route("/predict", methods=["POST"])
def predict():
    req = request.data.decode("utf-8")
    data = json.loads(req)
    array = data.get('data')
    return array


app.run(host['host'])
