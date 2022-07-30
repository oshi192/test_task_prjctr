import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import pickle

maxlen = 190

def show_history(h):
    epochs_trained = len(h.history['loss'])
    plt.plot(range(0, epochs_trained), h.history.get('loss'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_loss'), label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

test = pd.read_csv('./data/test.csv', index_col=0)
train = pd.read_csv('./data/train.csv', index_col=0)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train['excerpt'])

lengths = [len(t.split(' ')) for t in train['excerpt']]
plt.hist(lengths, bins = len(set(lengths)))

def get_sequences(tokenizer, text):
  sequences = tokenizer.texts_to_sequences(text)
  padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, truncating='post', padding='post', maxlen=maxlen)
  return padded

padded_train_seq = get_sequences(tokenizer, train['excerpt'])

# saving
with open('./data/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# loading
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(10000, 16, input_length=maxlen),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
        tf.keras.layers.Dense(190, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
])
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()

x_train = padded_train_seq
y_train = train['target']

history = model.fit(
    x_train, y_train,
    shuffle=True,
    validation_split=0.2,
    epochs=20,
    verbose=1
)

show_history(history)
model.save_weights('./data/test_model')

