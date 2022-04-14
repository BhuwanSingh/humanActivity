import flask
import io
import string
import time
import os
import numpy as np
# import tensorflow as tf
import pandas as pd
import keras
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)


regressor = keras.models.load_model('static/models/v1')

# def preprocess_raw_data()

def predict_activity(l):

    if(l[0]>l[1] and l[0]>l[2] and l[0]>l[3] and l[0]>l[4] and l[0]>l[5]):
        return 'Downstairs'

    if(l[1]>l[0] and l[1]>l[2] and l[1]>l[3] and l[1]>l[4] and l[1]>l[5]):
        return 'Jogging'

    if(l[2]>l[0] and l[2]>l[1] and l[2]>l[3] and l[2]>l[4] and l[2]>l[5]):
        return 'Sitting'

    if(l[3]>l[0] and l[3]>l[1] and l[3]>l[2] and l[3]>l[4] and l[3]>l[5]):
        return 'Standing'

    if(l[4]>l[0] and l[4]>l[1] and l[4]>l[2] and l[4]>l[3] and l[4]>l[5]):
        return 'Upstairs'

    if(l[5]>l[0] and l[5]>l[1] and l[5]>l[2] and l[5]>l[3] and l[5]>l[4]):
        return 'Walking'

    return 'Error'

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route("/predict", methods=["GET"])
def predict():
    posted_data = request.data.decode()

    with open('static/test3.txt', 'w') as f:
        f.write(str(posted_data))


    columns2 = ['x-axis', 'y-axis', 'z-axis']
    df2 = pd.read_csv('static/test3.txt', header=None, names=columns2, on_bad_lines='skip', sep=",")
    df2 = df2.dropna()
    df2.head()

    N_TIME_STEPS = 200
    N_FEATURES = 3
    step = 20
    segments2 = []

    for i in range(0, len(df2) - N_TIME_STEPS, step):
        xs = df2['x-axis'].values[i: i + N_TIME_STEPS]
        ys = df2['y-axis'].values[i: i + N_TIME_STEPS]
        zs = df2['z-axis'].values[i: i + N_TIME_STEPS]
        segments2.append([xs, ys, zs])

    reshaped_segments2 = np.asarray(segments2, dtype=np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)

    predicted_val = regressor.predict(reshaped_segments2)
    res = []

    ###############    IDEA ---- SLIDING WINDOW MAX IN RESULT ALSO #################
    for val in predicted_val:
        res.append(predict_activity(val))

    return jsonify(res)



import os
if __name__ == '__main__':
    app.run(debug=True, port = os.environ['PORT'])
