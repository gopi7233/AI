# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:26:28 2020

@author: thummago
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle 

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['post'])

def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text='Employee salary should be $ {}'.format(output))


if __name__ =="__main__":
    app.run(debug=True)