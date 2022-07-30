import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip3", "install", package])
install('fastapi')
install('pickle')

import pickle
import tensorflow as tf
from fastapi import FastAPI

maxlen=190

with open('./test_nlp/data/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Restore the weights
model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(10000, 16, input_length=maxlen),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
        tf.keras.layers.Dense(190, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
])
model.load_weights('./test_nlp/data/test_model')

def make_predict(text: str):
        sequence = tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, truncating='post', padding='post', maxlen=maxlen)
        p = model.predict(padded)[0]
        return p[0]

app = FastAPI()

@app.get("/test-nlp")
def read_classification(text: str):
        return str(make_predict(text))
