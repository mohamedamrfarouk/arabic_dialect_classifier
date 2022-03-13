from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import json
import re
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import string
from cleaning_preprocessing import *

def predict_the_dialect(text):
    loaded_model = tf.keras.models.load_model('F:\dialect classifier project for AIM\deployment\models\deep_learning_model.h5')
    clean_text = clean_the_text(text)
    tokenizerr = load_tokenizer(r'F:\dialect classifier project for AIM\deployment\models\tokenizer.pickle')
    preprocessed_text = preprocess_text_for_predicting(tokenizerr , clean_text)
    prediction = predict_dialect(loaded_model, preprocessed_text)
    
    return  prediction


app = Flask(__name__)


@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')


@app.route('/',methods=['POST'])
def predict():
    text = request.form.get('arabic_text')
    pred = predict_the_dialect(text)   
    return render_template('index.html' , prediction = pred) 


if __name__ == '__main__':
    app.run(port=3000 , debug=True)