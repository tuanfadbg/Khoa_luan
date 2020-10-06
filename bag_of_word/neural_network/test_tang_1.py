import csv
import random
import re
import string
import time

import numpy as np
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.python.keras.layers import TextVectorization

start_time = time.time()

is_covid_position = 13
title_position = 14
des_position = title_position + 1
content_position = des_position + 1
category_position = content_position + 1
keyword_position = category_position + 1

r = csv.reader(open('../data/combine/output.csv'))
# r = csv.reader(open('../dictionary/data_without_segmentation.csv'))
data_csv = list(r)

# loop_count = len(data_csv)
loop_count = 1000
data = []
is_covid = np.empty([loop_count, ])

sample_items = []
if loop_count == len(data_csv):
    sample_items = data_csv
else:
    sample_items = random.sample(data_csv, loop_count)



def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

max_features = 10000
sequence_length = 250

batch_size = 32
seed = 42

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

with open('../data/stopwords.txt', 'r') as f:
    stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])

    for i in range(loop_count):
        # data.append(sample_items[i][1])
        # is_covid = 0
        if sample_items[i][is_covid_position]:
            is_covid[i] = 0
        else:
            is_covid[i] = 1
        # print("Vectorized review", vectorize_text(sample_items[i][content_position], is_covid))

        # tftxt.load
        # data.append(SplitWord(sample_items[i][title_position], stopwords).get_words_feature())
        data.append(sample_items[i][title_position])



    print(data)

    VOCAB_SIZE = 1000

    tf.compat.v1.disable_eager_execution()

    tokens = tf.compat.v1.string_split(data)
    indices = tft.compute_and_apply_vocabulary(tokens, top_k=VOCAB_SIZE)
    bow_indices, weight = tft.tfidf(indices, VOCAB_SIZE + 1)

    # print(bow_indices.shape)
    # print(weight.graph.)
    # tft.tfidf(data, vocab_size=1000)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    history = model.fit(weight, epochs=10,
                        validation_data=is_covid,
                        validation_steps=30)




