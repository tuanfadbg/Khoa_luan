import csv
import os
import pickle
import re
import time

import numpy as np
from keras import layers
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import tensorflow as tf

from bag_of_word.settings import STOP_WORDS
from bag_of_word.split_word import SplitWord

is_covid_position = 13
title_position = 14
des_position = title_position + 1
content_position = des_position + 1
category_position = content_position + 1
keyword_position = category_position + 1

start_time = time.time()
folder = "model_neural_network_2"
r = csv.reader(open('../data/combine/output_temp_2.csv'))
data_csv = list(r)

loop_count = len(data_csv)
# loop_count = 2000
data = []
categories = []

sample_items = []
# if loop_count == len(data_csv):
sample_items = data_csv
# else:
#     sample_items = random.sample(data_csv, loop_count)
# categories = np.empty([4, ])


offset = 13

data_0 = []
data_1 = []
data_2 = []
data_3 = []

data.append([])
data.append([])
data.append([])
data.append([])

categories.append([])
categories.append([])
categories.append([])
categories.append([])

with open(STOP_WORDS, 'r') as f:
    stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
    for i in range(loop_count):
        if sample_items[i][is_covid_position] == '1':
            str = sample_items[i][offset + 1] + ' ' + sample_items[i][offset + 2] + ' ' + sample_items[i][
                offset + 3] + ' ' + re.sub('\|', ' | ', sample_items[i][offset + 4]) + ' ' + re.sub('\|', ' | ', sample_items[i][offset + 5])

            category = sample_items[i][:13]
            for j in range(len(category)):
                if category[j] == '1':
                    category[j] = 1
                else:
                    category[j] = 0

            if sample_items[i][0] == '1' or sample_items[i][1] == '1' or sample_items[i][2] == '1':
                data[0].append(SplitWord(str, stopwords).segmentation_remove_stop_word())
                categories[0].append(category[:3])

            if sample_items[i][3] == '1':
                data[1].append(SplitWord(str, stopwords).segmentation_remove_stop_word())

            if sample_items[i][4] == '1' or sample_items[i][5] == '1' or sample_items[i][6] == '1' \
                    or sample_items[i][7] == '1' or sample_items[i][8] == '1' or sample_items[i][9] == '1':
                data[2].append(SplitWord(str, stopwords).segmentation_remove_stop_word())
                categories[2].append(category[4:10])

            if sample_items[i][10] == '1' or sample_items[i][11] == '1' or sample_items[i][12] == '1':
                data[3].append(SplitWord(str, stopwords).segmentation_remove_stop_word())
                categories[3].append(category[10:13])

    print(data[0][0])
    # print(categories)

    # for i in range(5):
    #     print("----")
    #     print(data[0][i])
    #     print(categories[0][i])

    def convert_sparse_matrix_to_sparse_tensor(X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    for i in range(4):
        if i != 1:
            embedding_dim = 16
            max_features = 30000

            cv = CountVectorizer(max_features=max_features)
            word_count_vector = cv.fit_transform(data[i])

            tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
            tfidf_transformer.fit(word_count_vector)
            pickle.dump(tfidf_transformer, open(folder + "/logistic_regression_tfidf_transformer_{}.pkl".format(i), 'wb'))
            pickle.dump(cv, open(folder + "/logistic_regression_count_vectorizer_{}.pkl".format(i), 'wb'))

            count_vector = cv.transform(data[i])

            tf_idf_vector = tfidf_transformer.transform(count_vector)

            X_train, X_test, y_train, y_test = train_test_split(tf_idf_vector, np.array(categories[i]), test_size=0.2,
                                                                random_state=42)

            # text_clf = OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)

            y_train_np = np.array(y_train)
            y_test_np = np.array(y_test)

            print('LogisticRegression')
            for k in range(len(categories[i][0])):
                print('Category {} {}'.format(i, k))

                VOCAB_SIZE = len(cv.get_feature_names())
                print("VOCAB_SIZE = {}".format(VOCAB_SIZE))

                # model = tf.keras.Sequential([
                #     tf.keras.layers.Embedding(max_features + 1, embedding_dim),
                #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                #     tf.keras.layers.Dense(64, activation='relu'),
                #     tf.keras.layers.Dense(1)
                # ])


                model = tf.keras.Sequential([
                    layers.Embedding(max_features + 1, embedding_dim),
                    layers.Dropout(0.2),
                    layers.GlobalAveragePooling1D(),
                    layers.Dropout(0.2),
                    layers.Dense(1)])

                print(model.summary())

                model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True), optimizer='adam',
                              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

                checkpoint_path = folder + "/logistic_regression_{}_{}.ckpt".format(i, k)
                checkpoint_dir = os.path.dirname(checkpoint_path)

                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                                 save_weights_only=True,
                                                                 verbose=1)

                epochs = 0
                history = model.fit(
                    X_train.toarray(),
                    y_train_np[:, k],
                    batch_size=64,
                    epochs=epochs,
                    validation_data=(X_test.toarray(), y_test_np[:, k]),
                    callbacks=[cp_callback])

                model.load_weights(checkpoint_path)

                # Re-evaluate the model
                loss, acc = model.evaluate(X_test.toarray(), y_test_np[:, k], verbose=2)

                print("Restored model, accuracy: {:5.2f}%, loss: {:5.2f}%".format(100 * acc, 100 * loss))

                # prediction = model.predict(X_test.toarray())
                # print(classification_report(y_test_np[:, k], prediction))