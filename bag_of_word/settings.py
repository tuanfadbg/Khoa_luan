import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

DICTIONARY_PATH = 'dictionary.txt'
DATA_TRAIN_JSON = 'data/output_covid.csv'
DATA_TEST_JSON = 'data/output_covid.csv'

STOP_WORDS = os.path.join(DIR_PATH, "dictionary/stopwords.txt")

SPECIAL_CHARACTER = '0123456789%@$.,=+-!;/()*"&^:#…,”“?…...|\n\t\''


# from pathlib import Path
# path = Path("/here/your/path/file.txt")
# print(path.parent)