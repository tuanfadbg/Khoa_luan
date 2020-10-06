import csv
import time

from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
from bag_of_word.split_word import SplitWord

import numpy as np

import tensorflow as tf

# !pip install -q tensorflow-hub
# !pip install -q tensorflow-datasets
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

is_covid_position = 13
title_position = 14
des_position = title_position + 1
content_position = des_position + 1
category_position = content_position + 1
keyword_position = category_position + 1

start_time = time.time()

r = csv.reader(open('../data/combine/output.csv'))
data_csv = list(r)

# loop_count = len(data_csv)
loop_count = 1000
data = []
categories = []

sample_items = []
# if loop_count == len(data_csv):
sample_items = data_csv
# else:
#     sample_items = random.sample(data_csv, loop_count)
# categories = np.empty([4, ])
for i in range(loop_count):
    if sample_items[i][is_covid_position] == '1':
        str = sample_items[i][title_position] + " " \
              + sample_items[i][des_position] + " " \
              + sample_items[i][content_position] + " " \
              + sample_items[i][category_position] + " " \
              + sample_items[i][keyword_position]
        data.append(SplitWord(str).segmentation())
        category = []
        if sample_items[i][0] == '1' or sample_items[i][1] == '1' or sample_items[i][2] == '1':
            category.append(1)
        else:
            category.append(0)

        if sample_items[i][3] == '1':
            category.append(1)
        else:
            category.append(0)

        if sample_items[i][4] == '1' or sample_items[i][5] == '1' or sample_items[i][6] == '1' \
                or sample_items[i][7] == '1' or sample_items[i][8] == '1' or sample_items[i][9] == '1':
            category.append(1)
        else:
            category.append(0)

        if sample_items[i][10] == '1' or sample_items[i][11] == '1' or sample_items[i][12] == '1':
            category.append(1)
        else:
            category.append(0)
        # print(category)

        categories.append(category)

print(categories)
X_train, X_test, y_train, y_test = train_test_split(data, categories, test_size=0.2, random_state=42)

X_train_np = np.array(y_train)
print(X_train_np.shape)

input_dim = X_train_np.shape[1]

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(X_train, y_train,
                    epochs=100,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=-1)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))