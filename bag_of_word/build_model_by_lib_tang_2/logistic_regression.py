import csv
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from bag_of_word.split_word import SplitWord

is_covid_position = 13
title_position = 14
des_position = title_position + 1
content_position = des_position + 1
category_position = content_position + 1
keyword_position = category_position + 1

start_time = time.time()

r = csv.reader(open('../data/combine/output.csv'))
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

NB_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=0.0)),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
])

y_train_np = np.array(y_train)
y_test_np = np.array(y_test)

print('LogisticRegression')
for i in range(4):
    print('Category {}'.format(i))
    # train the model using X_dtm & y
    NB_pipeline.fit(X_train, y_train_np[:, i])
    # compute the testing accuracy
    prediction = NB_pipeline.predict(X_test)
    print(classification_report(y_test_np[:, i], prediction))
    print('Test accuracy is {}'.format(accuracy_score(y_test_np[:, i], prediction)))
