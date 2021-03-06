from flask import Flask, render_template, request
import cv2
from imageio import imread
import numpy as np
import re
import sys
import os
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import json


model_weights_path = './models_files/mnist_fashion_model.h5'
model_json_path = './models_files/mnist_fashion_model.json'

fashion_dict = {0: 'T_shirt',
                1: 'Trouser',
                2: 'Pullover',
                3: 'Dress',
                4: 'Coat',
                5: 'Sandal',
                6: 'Shirt',
                7: 'Sneaker',
                8: 'Bag',
                9: 'Ankle boot'}

# initalize our flask app
app = Flask(__name__)

def init_model():
    '''
    Function for initializing keras model from json and h5 files.
    '''
    # https://stackoverflow.com/questions/53212672/read-only-mode-in-keras
    json_file = open(model_json_path,'r')
    model_json = json_file.read()
    json_file.close()

    model = model_from_json(model_json)
    model.load_weights(model_weights_path)
    print('Model loaded')
    return model


# decoding an image from base64 into raw representation
def convertImage(imgData1):
    '''
    Function decode image from base64 format into raw png representation.
    input - base64 image from site widget
    output - saved output.png
    '''
    imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    # get the raw data format of the image
    imgData = request.get_data()
    # encode it into a suitable format and save result
    convertImage(imgData)
    # read the image into memory
    x = imread('output.png', pilmode = 'L')
    # make it the right size and shape
    x = cv2.resize(x, (28, 28))
    x = x.reshape(1, 28, 28, 1)

    model = init_model()
    output = model.predict(x)
    print(output)
    response = np.argmax(output, axis=1)
    print(fashion_dict[response[0]])
    return str(fashion_dict[response[0]])

if __name__ == "__main__":
    # run the app locally on the given port
    app.run(host='0.0.0.0', port=9091)
# optional if we want to run in debugging mode
# app.run(debug=True)
