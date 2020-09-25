from pyvi import ViTokenizer

SPECIAL_CHARACTER = '0123456789%@$.,=+!;/()*"&^:#…,”“?…._|\n\t\''


class SplitWord(object):
    def __init__(self, text=None, stopwords=None):
        self.text = text
        self.stopwords = stopwords

    def segmentation(self):
        return ViTokenizer.tokenize(self.text)

    def split_words(self):
        for i in SPECIAL_CHARACTER:
            self.text = self.text.replace(i, '')

        # text = self.text.replace(SPECIAL_CHARACTER, " ")
        text = self.segmentation()
        # text = self.text
        try:
            return [x.lower() for x in text.split()]
        except TypeError:
            return []

    def get_words_feature(self):
        split_words = self.split_words()
        return [word for word in split_words if word not in self.stopwords]
