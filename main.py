import pandas as pd
import tensorflow as tf
import keras as k
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras import losses


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

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(3000, 16, input_length=max_length))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(17, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(metrics=['accuracy'], loss='mean_absolute_error')
num_epochs = 30

history = model.fit(np.array(training_pad), np.array(training_lab_seq), epochs=num_epochs, validation_data=(np.array(testing_pad), np.array(testing_label_seq)), verbose=2)