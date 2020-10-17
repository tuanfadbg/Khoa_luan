import csv
import pickle
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from bag_of_word.settings import STOP_WORDS
from bag_of_word.split_word import SplitWord

is_covid_position = 13
title_position = 14
des_position = title_position + 1
content_position = des_position + 1
category_position = content_position + 1
keyword_position = category_position + 1

start_time = time.time()
folder = "model"
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
stopwords = []
with open(STOP_WORDS, 'r') as f:
    stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])

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

for i in range(loop_count):
    if sample_items[i][is_covid_position] == '1':
        str = sample_items[i][offset + 1] + ' ' + sample_items[i][offset + 2] + ' ' + sample_items[i][
            offset + 3] + ' ' + sample_items[i][offset + 4] + ' ' + sample_items[i][offset + 5]

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

# print(categories)

# for i in range(5):
#     print("----")
#     print(data[0][i])
#     print(categories[0][i])


for i in range(4):
    if i != 1:
        cv = CountVectorizer()
        word_count_vector = cv.fit_transform(data[i])

        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(word_count_vector)
        pickle.dump(tfidf_transformer, open(folder + "/logistic_regression_tfidf_transformer_{}.pkl".format(i), 'wb'))
        pickle.dump(cv, open(folder + "/logistic_regression_count_vectorizer_{}.pkl".format(i), 'wb'))

        count_vector = cv.transform(data[i])

        tf_idf_vector = tfidf_transformer.transform(count_vector)

        X_train, X_test, y_train, y_test = train_test_split(tf_idf_vector, categories[i], test_size=0.2,
                                                            random_state=42)

        text_clf = OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)

        y_train_np = np.array(y_train)
        y_test_np = np.array(y_test)

        print('LogisticRegression')
        for k in range(len(categories[i][0])):
            print('Category {} {}'.format(i, k))

            text_clf.fit(X_train, y_train_np[:, k])

            prediction = text_clf.predict(X_test)
            print(classification_report(y_test_np[:, k], prediction))
            print('Test accuracy is {}'.format(accuracy_score(y_test_np[:, k], prediction)))
            pickle.dump(text_clf, open(folder + '/logistic_regression_{}_{}.pkl'.format(i, k), 'wb'))
