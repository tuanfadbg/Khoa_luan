import csv
import operator
import pickle
import random

from bag_of_word.split_word import SplitWord

STOP_WORDS = "stopwords.txt"

PREFIX = 'segmentation_7_10000_items'

class Dictionary:
    def __init__(self):
        self.loop_count = 10000
        self.level = 0

        with open(STOP_WORDS, 'r') as f:
            self.stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])

    def split(self):
        r = csv.reader(open('data_without_segmentation.csv'))
        data_csv = list(r)
        self.data = []
        self.is_covid = []

        self.loop_count = len(data_csv)
        sample_items = random.sample(data_csv, self.loop_count)

        for i in range(self.loop_count):
            self.data.append(sample_items[i][1])
            self.is_covid.append(sample_items[i][0])

    def create_dictionary(self):
        self.word_dict = None
        # sample_items = random.sample(self.data, self.loop_count)

        for i in range(self.loop_count):
            print('create_dictionary ' + str(i))
            content = self.data[i]
            # for i in range(len(sample_items)):
            #     content = sample_items[i]
            # bow = content.split(" ")
            bow = SplitWord(content, self.stopwords).get_words_feature()
            if self.word_dict is None:
                self.word_dict = set(bow)
            else:
                self.word_dict = self.word_dict.union(set(bow))

        with open('dictionary_' + PREFIX + '.pickle', 'wb') as output:
            pickle.dump(self.word_dict, output, pickle.HIGHEST_PROTOCOL)

    def print_tf_idf(self):

        with open('dictionary_' + PREFIX + '.pickle', 'rb') as input:
            self.word_dict = pickle.load(input)

            bag_word_dict = []
            tf_bow = []
            for i in range(self.loop_count):
                print("print_tf_idf " + str(i))
                bag_word_dict.append(dict.fromkeys(self.word_dict, 0))
                # bow = self.data[i].split(" ")
                bow = SplitWord(self.data[i], self.stopwords).get_words_feature()

                for word in bow:
                    bag_word_dict[i][word] += 1

                tf_bow.append(self.compute_TF(bag_word_dict[i], bow))

            idfs = self.compute_IDF(bag_word_dict)

            idfs_sorted = sorted(idfs.items(), key=operator.itemgetter(1), reverse=True)

            writer_test = csv.writer(open('dictionary_' + PREFIX + '.csv', 'w'))
            writer_test.writerows(idfs_sorted)

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

    def compute_TFIDF(self, tf_bow, idfs):
        tfidf = {}
        for word, val in tf_bow.items():
            tfidf[word] = val * idfs[word]
        return tfidf

    def store_segment(self):
        r = csv.reader(open('../data/combine/output.csv'))
        data_csv = list(r)
        self.data = []
        self.is_covid = []

        # self.loop_count = len(data_csv)

        for i in range(self.loop_count):
            self.data.append(data_csv[i][1])
            self.is_covid.append(data_csv[i][0])

        save_data = []

        for i in range(self.loop_count):
            save_data.append([self.is_covid[i], self.data[i]])
        # writer_test = csv.writer(open('data_segmentation.csv', 'w'), delimiter='[', quotechar='', quoting=csv.QUOTE_MINIMAL)
        writer_test = csv.writer(open('data_segmentation.csv', 'w'))
        writer_test.writerows(save_data)

    def store_without_segment(self):
        is_covid_position = 13
        title_position = 14
        des_position = title_position + 1
        content_position = des_position + 1
        category_position = content_position + 1
        keyword_position = category_position + 1

        r = csv.reader(open('../data/combine/output.csv'))
        self.data_csv = list(r)
        self.data = []
        self.is_covid = []

        # self.loop_count = len(self.data_csv)

        for i in range(self.loop_count):
            category = " ".join(self.data_csv[i][category_position].split("|"))
            keyword = " ".join(self.data_csv[i][keyword_position].split("|"))
            str = self.data_csv[i][title_position] + " " \
                  + self.data_csv[i][des_position] + " " \
                  + self.data_csv[i][content_position] + " " \
                  + category + " " \
                  + keyword
            self.data.append(str)

            self.is_covid.append(self.data_csv[i][is_covid_position])

        save_data = []
        for i in range(self.loop_count):
            save_data.append([self.is_covid[i], self.data[i]])

        writer_test = csv.writer(open('data_without_segmentation.csv', 'w'))
        writer_test.writerows(save_data)


dic = Dictionary()
dic.split()
dic.create_dictionary()
dic.print_tf_idf()
