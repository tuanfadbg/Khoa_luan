import csv
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
loop_count = 100
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
cv = CountVectorizer()

# this steps generates word counts for the words in your docs
word_count_vector = cv.fit_transform(data)

print(word_count_vector.shape)

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)

# print idf values
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf_weights"])

# sort ascending
df_idf.sort_values(by=['idf_weights'])

# print(df_idf)

# count matrix
count_vector = cv.transform(data)

# print("print(count_vector)")
# print(count_vector)
# tf-idf scores
tf_idf_vector = tfidf_transformer.transform(count_vector)
# print("print(tf_idf_vector)")
# print(type(tf_idf_vector))
# print(tf_idf_vector)

feature_names = cv.get_feature_names()

# print(feature_names)

# print(type(np.array(tf_idf_vector)))
print(np.array(tf_idf_vector))

X_train, X_test, y_train, y_test = train_test_split(tf_idf_vector, is_covid, test_size=0.1, random_state=42)

VOCAB_SIZE = len(feature_names)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

print(model.summary())
# model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               optimizer=tf.keras.optimizers.Adam(1e-4),
#               metrics=['accuracy'])
print(X_train)
print(y_train)


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True), optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
# print("print(convert_sparse_matrix_to_sparse_tensor(X_train))")
# print()
# print(convert_sparse_matrix_to_sparse_tensor(X_train))
history = model.fit(
    X_train,
    np.array(y_train),
    batch_size=64,
    epochs=2)

y_pred = model.predict(X_test, y_test)

print(classification_report(y_test, y_pred))

# X_train, X_test, y_train, y_test = train_test_split(tf_idf_vector, is_covid, test_size=0.1, random_state=42)
# clf = LogisticRegression(random_state=1).fit(X_train, y_train)
# print(clf.score(X_test, y_test))
# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))

