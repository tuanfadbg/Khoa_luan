#
# import tensorflow_datasets as tfds
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
#
# sample_string = 'Hello TensorFlow.'
# cv = CountVectorizer()
# word_count_vector = cv.fit_transform([sample_string])
# print(cv.get_feature_names())
# encoder = tfds.features.text.SubwordTextEncoder(cv.get_feature_names())
#
# encoded_string = encoder.encode(sample_string)
#
# for index in encoded_string:
#   print('{} ----> {}'.format(index, encoder.decode([index])))
#
# print(encoded_string)
from underthesea import word_tokenize

import regex as re

from bag_of_word.settings import STOP_WORDS
from bag_of_word.split_word import SplitWord

uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"


def loaddicchar():
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic


dicchar = loaddicchar()


# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

text = covert_unicode('chữa Covid-19 Nếu bạn không xử lý vấn đề này, khi đưa vào mô hình học máy tính sẽ hiểu đó là các từ khác nhau mặc dù ta đang nhìn thấy chúng chẳng khác nhau gì.')

print(word_tokenize(text, format="text"))
with open(STOP_WORDS, 'r') as f:
    stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
    print(SplitWord(text, stopwords).segmentation_remove_stop_word())
# def chuan_hoa_dau_tu_tieng_viet(word):
#     if not is_valid_vietnam_word(word):
#         return word
#
#     chars = list(word)
#     dau_cau = 0
#     nguyen_am_index = []
#     qu_or_gi = False
#     for index, char in enumerate(chars):
#         x, y = nguyen_am_to_ides.get(char, (-1, -1))
#         if x == -1:
#             continue
#         elif x == 9:  # check qu
#             if index != 0 and chars[index - 1] == 'q':
#                 chars[index] = 'u'
#                 qu_or_gi = True
#         elif x == 5:  # check gi
#             if index != 0 and chars[index - 1] == 'g':
#                 chars[index] = 'i'
#                 qu_or_gi = True
#         if y != 0:
#             dau_cau = y
#             chars[index] = bang_nguyen_am[x][0]
#         if not qu_or_gi or index != 1:
#             nguyen_am_index.append(index)
#     if len(nguyen_am_index) < 2:
#         if qu_or_gi:
#             if len(chars) == 2:
#                 x, y = nguyen_am_to_ids.get(chars[1])
#                 chars[1] = bang_nguyen_am[x][dau_cau]
#             else:
#                 x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
#                 if x != -1:
#                     chars[2] = bang_nguyen_am[x][dau_cau]
#                 else:
#                     chars[1] = bang_nguyen_am[5][dau_cau] if chars[1] == 'i' else bang_nguyen_am[9][dau_cau]
#             return ''.join(chars)
#         return word
#
#     for index in nguyen_am_index:
#         x, y = nguyen_am_to_ids[chars[index]]
#         if x == 4 or x == 8:  # ê, ơ
#             chars[index] = bang_nguyen_am[x][dau_cau]
#             # for index2 in nguyen_am_index:
#             #     if index2 != index:
#             #         x, y = nguyen_am_to_ids[chars[index]]
#             #         chars[index2] = bang_nguyen_am[x][0]
#             return ''.join(chars)
#
#     if len(nguyen_am_index) == 2:
#         if nguyen_am_index[-1] == len(chars) - 1:
#             x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
#             chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
#             # x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
#             # chars[nguyen_am_index[1]] = bang_nguyen_am[x][0]
#         else:
#             # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
#             # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
#             x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
#             chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
#     else:
#         # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
#         # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
#         x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
#         chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
#         # x, y = nguyen_am_to_ids[chars[nguyen_am_index[2]]]
#         # chars[nguyen_am_index[2]] = bang_nguyen_am[x][0]
#     return ''.join(chars)
#
#
# def is_valid_vietnam_word(word):
#     chars = list(word)
#     nguyen_am_index = -1
#     for index, char in enumerate(chars):
#         x, y = nguyen_am_to_ids.get(char, (-1, -1))
#         if x != -1:
#             if nguyen_am_index == -1:
#                 nguyen_am_index = index
#             else:
#                 if index - nguyen_am_index != 1:
#                     return False
#                 nguyen_am_index = index
#     return True
#
#
# def chuan_hoa_dau_cau_tieng_viet(sentence):
#     """
#         Chuyển câu tiếng việt về chuẩn gõ dấu kiểu cũ.
#         :param sentence:
#         :return:
#         """
#     sentence = sentence.lower()
#     words = sentence.split()
#     for index, word in enumerate(words):
#         cw = re.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')
#         # print(cw)
#         if len(cw) == 3:
#             cw[1] = chuan_hoa_dau_tu_tieng_viet(cw[1])
#         words[index] = ''.join(cw)
#     return ' '.join(words)
#
#
# """
#     End section: Chuyển câu văn về cách gõ dấu kiểu cũ: dùng òa úy thay oà uý
#     Xem tại đây: https://vi.wikipedia.org/wiki/Quy_tắc_đặt_dấu_thanh_trong_chữ_quốc_ngữ
# """
#
# print(chuan_hoa_dau_cau_tieng_viet('anh hoà, đang làm.. gì'))