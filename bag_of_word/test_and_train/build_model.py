import csv
import random

import pandas as pd
import numpy as np
from pyvi import ViTokenizer
from joblib import dump, load
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from numpy import asarray
from numpy import savetxt

from bag_of_word.settings import STOP_WORDS
from bag_of_word.split_word import SplitWord

NUMCOUNT = '9_1000_items_last_in_dict'


class Model:
    def __init__(self):
        self.loop_count = 1000
        self.level = 0

        with open(STOP_WORDS, 'r') as f:
            self.stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])

        r = csv.reader(open('../dictionary/dictionary_segmentation_6.csv'))
        self.idfs = list(r)

        # if self.level != 0:
        #     for i in range(len(self.idfs)):
        #         print(i)
        #         if float(self.idfs[i][1]) < self.level:
        #             self.idfs = self.idfs[:i]
        #             break

        self.idfs = self.idfs[(len(self.idfs) - 15000):]
        print(self.idfs[0])
        print(len(self.idfs))

        self.word_dictionary = []
        word_value = []
        for i in range(len(self.idfs)):
            self.word_dictionary.append(self.idfs[i][0])
            word_value.append(self.idfs[i][1])
            # word_value.append(1)

        self.idfs_dict = dict(zip(self.word_dictionary, word_value))

        # print(self.idfs_dict)
        r = csv.reader(open('../dictionary/data_without_segmentation.csv'))
        data_csv = list(r)

        # self.loop_count = len(data_csv)
        self.data = []
        self.is_covid = np.empty([self.loop_count, ])

        sample_items = random.sample(data_csv, self.loop_count)

        for i in range(self.loop_count):
            self.data.append(sample_items[i][1])
            if sample_items[i][0]:
                self.is_covid[i] = 0
            else:
                self.is_covid[i] = 1

        # print(self.is_covid)

    def get_bag_of_word(self):
        bag_word_dict = []
        tf_bow = []
        for i in range(self.loop_count):
            print("get_bag_of_word loop 1 " + str(i))
            bag_word_dict.append(dict.fromkeys(self.word_dictionary, 0))
            # bow = self.data[i].split(" ")
            bow = SplitWord(self.data[i], self.stopwords).get_words_feature()

            for word in bow:
                if word in self.word_dictionary:
                    bag_word_dict[i][word] += 1
            tf_bow.append(self.compute_TF(bag_word_dict[i], bow))

        tfidf_bow = []
        for i in range(self.loop_count):
            print("get_bag_of_word loop 2 " + str(i))
            tfidf_bow.append(self.compute_TFIDF(tf_bow[i]))

        return tfidf_bow

    def get_value_bow(self, tfidf_bow):
        value_bow = np.empty([len(tfidf_bow), len(tfidf_bow[0])])
        for i in range(len(tfidf_bow)):
            print("get_value_bow " + str(i))
            array = np.array(list(tfidf_bow[i].values()))
            # print(type(array))
            value_bow[i] = array
            # print(value)

        return value_bow

    def save_value_bow_to_file(self, value_bow, file_name):
        savetxt(file_name, value_bow, delimiter=',')

    def compute_TF(self, word_dict, bow):
        tf_dict = {}
        bow_count = len(bow)
        for word, count in word_dict.items():
            tf_dict[word] = count / float(bow_count)
        return tf_dict

    def compute_IDF(self, doc_list):
        import math
        idf_dict = {}
        N = len(doc_list)

        # count number of documents that contain this word
        idf_dict = dict.fromkeys(doc_list[0].keys(), 0)
        for doc in doc_list:
            for word, count in doc.items():
                if count > 0:
                    idf_dict[word] += 1

        for word, count in idf_dict.items():
            idf_dict[word] = math.log(N / float(count))

        return idf_dict

    def compute_TFIDF(self, tf_bow):
        tfidf = {}
        for word, val in tf_bow.items():
            tfidf[word] = val * float(self.idfs_dict[word])
        return tfidf


MODEL_FILENAME = 'model/is_covid_' + NUMCOUNT + '.joblib'

dic = Model()
bow = dic.get_bag_of_word()

data = dic.get_value_bow(bow)

dic.save_value_bow_to_file(data, "data/data_" + NUMCOUNT + ".csv")
dic.save_value_bow_to_file(dic.is_covid, "data/data_" + NUMCOUNT + "_value.csv")

# SVM
X_train, X_test, y_train, y_test = train_test_split(data, dic.is_covid, test_size=0.05, random_state=1)
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
dump(clf, MODEL_FILENAME)
print(clf.score(X_test, y_test))

# clf = load(MODEL_FILENAME)
# print(clf.score(data, dic.is_covid))


# logistic regestion
# X_train, X_test, y_train, y_test = train_test_split(data, dic.is_covid, test_size=0.05, random_state=1)
# clf = LogisticRegression(random_state=1).fit(X_train, y_train)
# dump(clf, MODEL_FILENAME)
# print(clf.score(X_test, y_test))
# clf = load(MODEL_FILENAME)
# print(clf.score(data, dic.is_covid))
