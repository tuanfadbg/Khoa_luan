import csv
import pickle
import random
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from bag_of_word.split_word import SplitWord

start_time = time.time()
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1),
                                              max_df=0.8,
                                              max_features=None)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())
                     ])

r = csv.reader(open('../dictionary/data_without_segmentation.csv'))
data_csv = list(r)

loop_count = len(data_csv)
# loop_count = 16000
data = []
is_covid = np.empty([loop_count, ])

sample_items = []
if loop_count == len(data_csv):
    sample_items = data_csv
else:
    sample_items = random.sample(data_csv, loop_count)

for i in range(loop_count):
    data.append(SplitWord(sample_items[i][1]).segmentation())
    # data.append(sample_items[i][1])
    if sample_items[i][0]:
        is_covid[i] = 0
    else:
        is_covid[i] = 1

X_train, X_test, y_train, y_test = train_test_split(data, is_covid, test_size=0.1, random_state=42)

text_clf = text_clf.fit(X_train, y_train)

train_time = time.time() - start_time
print('Done training Navie Bayes in', train_time, 'seconds.')

print(text_clf.score(X_test, y_test))
# Save model
pickle.dump(text_clf, open("model/navie_bayes.pkl", 'wb'))

y_pred = text_clf.predict(X_test)
print(classification_report(y_test, y_pred))