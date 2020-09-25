import os

from gensim import corpora, matutils
from pyvi import ViTokenizer

from bag_of_word import settings
import csv

class FileReader(object):
    def __init__(self, filePath, encoder = None):
        self.filePath = filePath
        self.encoder = encoder if encoder != None else 'utf-16le'

    def read(self):
        with open(self.filePath) as f:
            s = f.read()
        return s

    def content(self):
        s = self.read()
        return s.decode(self.encoder)

    # def read_json(self):
    #     with open(self.filePath) as f:
    #         s = json.load(f)
    #     return s

    def read_csv(self):
        r = csv.reader(open(self.filePath))
        lines = list(r)
        return lines

    def read_stopwords(self):
        with open(self.filePath, 'r') as f:
            stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
        return stopwords

    def load_dictionary(self):
        return corpora.Dictionary.load_from_text(self.filePath)

class NLP(object):
    def __init__(self, text = None):
        self.text = text
        # self.__set_stopwords()

    def segmentation(self):
        return ViTokenizer.tokenize(self.text)

    def __set_stopwords(self):
        self.stopwords = FileReader(settings.STOP_WORDS).read_stopwords()

    def split_words(self):
        text = self.segmentation()
        try:
            return [x.strip(settings.SPECIAL_CHARACTER).lower() for x in text.split()]
        except TypeError:
            return []

    def get_words_feature(self):
        split_words = self.split_words()
        return [word for word in split_words if word.encode('utf-8') not in self.stopwords]


class FileStore(object):
    def __init__(self, filePath, data = None):
        self.filePath = filePath
        self.data = data

    def store_dictionary(self, dict_words):
        dictionary = corpora.Dictionary(dict_words)
        dictionary.filter_extremes(no_below=20, no_above=0.3)
        dictionary.save_as_text(self.filePath)

class FeatureExtraction(object):
    def __init__(self, data):
        self.data = data

    def __build_dictionary(self):
        print('Building dictionary')
        dict_words = []
        i = 0
        for text in self.data:
            i += 1
            print("Step {} / {}".format(i, len(self.data)))
            words = NLP(text = text['content']).get_words_feature()
            dict_words.append(words)
        FileStore(filePath=settings.DICTIONARY_PATH).store_dictionary(dict_words)

    def __load_dictionary(self):
        if os.path.exists(settings.DICTIONARY_PATH) == False:
            self.__build_dictionary()
        self.dictionary = FileReader(settings.DICTIONARY_PATH).load_dictionary()

    def __build_dataset(self):
        self.features = []
        self.labels = []
        i = 0
        for d in self.data:
            i += 1
            print("Step {} / {}".format(i, len(self.data)))
            self.features.append(self.get_dense(d['content']))
            self.labels.append(d['category'])

    def get_dense(self, text):
        self.__load_dictionary()
        words = NLP(text).get_words_feature()
        # Bag of words
        vec = self.dictionary.doc2bow(words)
        dense = list(matutils.corpus2dense([vec], num_terms=len(self.dictionary)).T[0])
        return dense

    def get_data_and_label(self):
        self.__build_dataset()
        return self.features, self.labels

temp = "Có thể tách từ theo nhiều cách khác nhau gây ra sự nhập nhằng về mặt ngữ nghĩa. Đây là một bài toán hết sức thú vị. Tuy nhiên chúng ta có một số công cụ để thực hiện việc này mà phổ biến nhất đó là VnTokenizer bạn đọc có thể cài đặt gói thư viện hỗ trợ Python bằng cách sử dụng pip với câu lệnh sau:"
print(NLP(text=temp).segmentation())
