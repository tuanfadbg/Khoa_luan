import csv
import os
import random

import keras
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from bag_of_word.split_word import SplitWord

is_covid_position = 13
title_position = 14
des_position = title_position + 1
content_position = des_position + 1
category_position = content_position + 1
keyword_position = category_position + 1

# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# print(x_train)

r = csv.reader(open('../data/combine/output.csv'))
data_csv = list(r)

# loop_count = len(data_csv)
loop_count = 3000
data = []
is_covid = np.empty([loop_count, ])

sample_items = []
if loop_count == len(data_csv):
    sample_items = data_csv
else:
    sample_items = random.sample(data_csv, loop_count)

for i in range(loop_count):
    data.append(SplitWord(sample_items[i][content_position]).segmentation())
    if sample_items[i][is_covid_position]:
        is_covid[i] = 0
    else:
        is_covid[i] = 1
# instantiate CountVectorizer()
cv = CountVectorizer(max_df=0.5)

# this steps generates word counts for the words in your docs
word_count_vector = cv.fit_transform(data)

print(word_count_vector.shape)

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)

# print idf values
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf_weights"])

# sort ascending
df_idf.sort_values(by=['idf_weights'])

df_idf = df_idf[10000:]


# print(df_idf)

# count matrix
count_vector = cv.transform(data)

tf_idf_vector = tfidf_transformer.transform(count_vector)

feature_names = cv.get_feature_names()

print(np.array(tf_idf_vector))

X_train, X_test, y_train, y_test = train_test_split(tf_idf_vector, is_covid, test_size=0.1, random_state=42)

VOCAB_SIZE = len(df_idf)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

print(model.summary())

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True), optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

checkpoint_path = "training_3/test_tang_1_save_model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

epochs = 4
history = model.fit(
    X_train.toarray(),
    y_train,
    batch_size=64,
    epochs=epochs,
    validation_data=(X_test.toarray(),y_test),
    callbacks=[cp_callback])

model.load_weights(checkpoint_path)

# Re-evaluate the model
loss,acc = model.evaluate(X_test.toarray(),y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
