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
import pickle

vocab_size = 100000
embedding_dim = 64
max_length = 50
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def remove_any_punctuation(text):
    punctuations = string.punctuation
    for ele in text:
        if ele in punctuations:
            text = text.replace(ele, "")
    return text


def remove_hashtags_mentions(text):
    text = re.sub("@([a-zA-Z0-9_]{1,50})","",text)
    text = re.sub("#([a-zA-Z0-9_]{1,50})","",text)
    return text


def remove_emojies(text):
    RE_EMOJI = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    text = re.sub(RE_EMOJI,"",text)
    return text


def clean_the_text(text):
    text = str(text)
    text = normalize_arabic(text)
    text = remove_hashtags_mentions(text)
    text = remove_any_punctuation(text)
    text = remove_emojies(text)
    text = text.strip()
    return  text

def tokenize_data(X):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(X)
    return tokenizer

def preprocess_text_for_predicting(tokenizer,text):
    t_text = tokenizer.texts_to_sequences([text])
    p_text = pad_sequences(t_text, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    return np.array(p_text)

def predict_dialect(loaded_model , p_text ):
    df = get_data_from_csv('F:\dialect classifier project for AIM/clean_data.csv')
    X = df.text
    y = df.dialect
    le = LabelEncoder()
    le.fit(y)

    prediction = le.inverse_transform([ np.argmax( loaded_model.predict(np.array(p_text)) )])[0]
    return prediction

def save_tokenizer_file(tokenizer):
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer(handel):
    with open(handel, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def get_data_from_csv(handel):
    df = pd.read_csv(handel, lineterminator='\n')    
    return df


def sava_clean_data_tocsv(df , handel):
    df.to_csv(handel)
