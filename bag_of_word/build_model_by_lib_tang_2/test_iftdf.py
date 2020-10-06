import csv
import random

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from bag_of_word.split_word import SplitWord

r = csv.reader(open('../dictionary/data_without_segmentation.csv'))
data_csv = list(r)

loop_count = len(data_csv)
# loop_count = 10000
data = []
is_covid = np.empty([loop_count, ])

sample_items = []
if loop_count == len(data_csv):
    sample_items = data_csv
else:
    sample_items = random.sample(data_csv, loop_count)

for i in range(loop_count):
    data.append(SplitWord(sample_items[i][1]).segmentation())
    if sample_items[i][0]:
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

# tf-idf scores
tf_idf_vector = tfidf_transformer.transform(count_vector)
print(tf_idf_vector)

feature_names = cv.get_feature_names()

print(feature_names)

X_train, X_test, y_train, y_test = train_test_split(tf_idf_vector, is_covid, test_size=0.1, random_state=42)
clf = LogisticRegression(random_state=1).fit(X_train, y_train)
print(clf.score(X_test, y_test))
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
# # get tfidf vector for first document
# first_document_vector = tf_idf_vector[0]
#
# # print the scores
# df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
# df.sort_values(by=["tfidf"], ascending=False)
#
# print(df)

