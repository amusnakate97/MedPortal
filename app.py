# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:34:20 2020
@author: Krish Naik
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model_resnet50.h5'

# Load your trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "The patient is normal"
    elif preds == 1:
        preds = "The patient is affected by covid"


    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index_dl.html')

@app.route('/corona', methods=['GET'])
def image_corona():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None

@app.route('/chronic', methods=['GET', 'POST'])
def predict_ckd():
    model2 = load_model('./classifier.h5')
    if request.method == 'POST':
        rbc = request.form['rbc']
        if rbc=='normal':
            rbc=1
        else:
            rbc=0
        al=int(request.form['al'])
        rc = int(request.form['rc'])
        sg = int(request.form['sg'])
        hemo = int(request.form['hemo'])

        htn_yes=request.form['htn']
        if htn_yes=='yes':
            htn_yes=1
        else:
            htn_yes=0

        pc_normal=request.form['pc']
        if pc_normal=='normal':
            pc_normal=1
        else:
            pc_normal=0

        dm_yes=request.form['dm']
        if dm_yes=='yes':
            dm_yes=1
        else:
            dm_yes=0

        sod=request.form['sod']
        bgr = request.form['bgr']

        htn_no = request.form['htn']
        if htn_no == 'no':
            htn_no = 1
        else:
            htn_no = 0

        pot = request.form['pot']

        pe_yes = request.form['pe']
        if pe_yes == 'yes':
            pe_yes = 1
        else:
            pe_yes = 0

        dm_no=request.form['dm']
        if dm_no=='no':
            dm_no=1
        else:
            dm_no=0
        wc = request.form['wc']
        bu = request.form['bu']
        age = request.form['age']
        bp = request.form['bp']


        featueres = [rbc, al, rc, hemo, sg, htn_yes, pc_normal,
       dm_yes, sod, bgr, htn_no, pot, pe_yes, dm_no, wc, bu,
       age, bp]
        import pickle
        scalerfile = 'scaler.sav'
        scaler = pickle.load(open(scalerfile, 'rb'))
        test_scaled_set = scaler.transform([featueres])

        prediction = model2.predict([test_scaled_set])
        output = prediction[0][0]
        #op_dict = {0: 'Fail', 1: 'Succeed'}
        #op = op_dict[output]
        if output>0.5:
            output='You are diagnosed with chronic kidney disease'
        else:
            output = 'You are not diagnosed with chronic kidney disease'

        return render_template('chronic_kidney.html',prediction_text=output)
    else:
        return render_template('chronic_kidney.html')


if __name__ == '__main__':
    app.run(debug=True)