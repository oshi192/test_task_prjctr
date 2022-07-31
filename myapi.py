from fastapi import FastAPI
import pickle
import tensorflow as tf

maxlen = 190

try:
    with open('./data/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    model = tf.keras.models.load_model("./data/test_model.h5")
except Exception as inst:
    print("An exception occurred")
    print(inst)


def make_predict(text: str):
    sequence = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, truncating='post', padding='post', maxlen=maxlen)
    p = model.predict(padded)[0]
    return p[0]


app = FastAPI()


@app.get("/test", description='Make predict for "CommonLit Readability Prize"')
def predict(text: str):
    return str(make_predict(text))
