from flask import Flask, render_template, request
import cv2
from imageio import imread
import numpy as np

import re
import sys
import os

import base64

from keras.models import load_model
from keras.models import model_from_json

import json

model_path = 'model.h5'
model_json_path = 'model.json'

# initalize our flask app
app = Flask(__name__)

def init_model():
    # https://stackoverflow.com/questions/53212672/read-only-mode-in-keras
    json_file = open('model.json','r')
    model_json = json_file.read()
    json_file.close()

    model = model_from_json(model_json)
    model.load_weights('model.h5')
    print('Model loaded')
    return model


# decoding an image from base64 into raw representation
def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))


@app.route('/', methods=['GET'])  #add metods here
def index():
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    # get the raw data format of the image
    imgData = request.get_data()
    # encode it into a suitable format
    convertImage(imgData)
    # read the image into memory
    x = imread('output.png', pilmode = 'L')
    # make it the right size
    x = cv2.resize(x, (28, 28))
    # imsave('final_image.jpg', x)
    # convert to a 4D tensor to feed into our model
    x = x.reshape(1, 28, 28, 1)

    model = init_model()
    output = model.predict(x)
    print(output)
    print(np.argmax(output, axis=1))
    response = np.argmax(output, axis=1)
    return str(response[0])

if __name__ == "__main__":
    # run the app locally on the given port
    app.run(host='0.0.0.0', port=9091)
# optional if we want to run in debugging mode
# app.run(debug=True)
