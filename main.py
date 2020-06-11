import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


df = pd.read_csv('data.csv',  delimiter=',')


posts = df['posts']
types = df['type']
training_size = 3000
training_sentences = posts[0:training_size]
testing_sentences = posts[training_size:]
training_labels = types[0:training_size]
testing_labels = types[training_size:]
oov_token = "<OOV>"
vocab_size = 1000
max_length = 200
pad = 'post'


tokenizer1 = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer1.fit_on_texts(training_sentences)

word_index = tokenizer1.word_index

tokenizer2 = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer2.fit_on_texts(training_labels)

label_index = tokenizer2.word_index

training_seq = tokenizer1.texts_to_sequences(training_sentences)
training_pad = pad_sequences(training_seq, padding=pad, maxlen=max_length, truncating=pad)
training_lab_seq = tokenizer2.texts_to_sequences(training_labels)

testing_seq = tokenizer1.texts_to_sequences(testing_sentences)
testing_pad = pad_sequences(testing_seq, padding=pad, maxlen=max_length, truncating=pad)
testing_label_seq = tokenizer2.texts_to_sequences(testing_labels)
training_lab = []
for x in training_lab_seq:
    training_lab.append(x[0])

testing_lab = []
for x  in testing_label_seq:
    testing_lab.append(x[0])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(3000, 64, input_length=max_length))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
for units in [64, 64]:
  model.add(tf.keras.layers.Dense(units, activation='relu'))
model.add(tf.keras.layers.Dense(18, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',  metrics=['accuracy'])
num_epochs = 30

history = model.fit(np.array(training_pad), np.array(training_lab), epochs=num_epochs, validation_data=(np.array(testing_pad), np.array(testing_lab)), verbose=2)