docA = "the quick brown fox jumps over the lazy dog and"
docB = "never jump over the lazy dog quickly"

bowA = docA.split(" ")
bowB = docB.split(" ")

# Create dictionary
word_dict = set(bowA).union(set(bowB))

wordDictA = dict.fromkeys(word_dict, 0)
wordDictB = dict.fromkeys(word_dict, 0)

# count the word in bads
for word in bowA:
    wordDictA[word] += 1

for word in bowB:
    wordDictB[word] += 1

# print(bowA)
print(wordDictA)
print(wordDictB)

def compute_TF(word_dict, bow):
    tf_dict = {}
    bow_count = len(bow)
    for word, count in word_dict.items():
        tf_dict[word] = count / float(bow_count)

    return tf_dict

tf_bowA = compute_TF(wordDictA, bowA)
tf_bowB = compute_TF(wordDictB, bowB)

print(tf_bowA)
print(tf_bowB)


def compute_IDF(doc_list):
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

idfs = compute_IDF([wordDictA, wordDictB])

print(idfs)

def compute_TFIDF(tf_bow, idfs):
    tfidf = {}
    for word, val in tf_bow.items():
        tfidf[word] = val*idfs[word]
    return tfidf

tfidf_bowA = compute_TFIDF(tf_bowA, idfs)
tfidf_bowB = compute_TFIDF(tf_bowB, idfs)

print("=========")
print(tfidf_bowA)
print(tfidf_bowB)
