import pickle
import tensorflow as tf
from fastapi import FastAPI

maxlen=190

with open('./test_nlp/data/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = tf.keras.models.load_model("./test_nlp/data/test_model.h5")

def make_predict(text: str):
        sequence = tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, truncating='post', padding='post', maxlen=maxlen)
        p = model.predict(padded)[0]
        return p[0]

app = FastAPI()

@app.get("/")
def read_classification(text: str):
        return str(make_predict(text))
