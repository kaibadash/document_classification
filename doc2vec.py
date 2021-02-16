import glob
import os
import random
import pickle
import MeCab
import pandas as pd
import const
from scdv import SparseCompositeDocumentVectors, build_word2vec
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from document import Document

# ドキュメントのvectorを計算する

tagger = MeCab.Tagger('-Owakati')
model = pickle.load(open("model/model.pkl", "rb"))
scdv = SparseCompositeDocumentVectors(
    model,
    const.CLUSTER_NUM,
    const.VECTOR_NUM,
    "model/gmm_cluster.pkl",
    "model/gmm_prob_cluster.pkl"
)

files = glob.glob("tmp/text/**/*.txt")
tagger = MeCab.Tagger('-Owakati')
categories = []
sentences = [] # 文章ごとの単語の配列
df = pd.DataFrame()
for file_path in files:
    # 少ないデータで試す
    if random.randrange(100) != 0:
        continue
    print("load: " + file_path)

    with open(file_path, 'r') as file:
        text = file.read()
    lines = text.split("\n")
    # Remove gabages
    lines.pop(0)
    lines.pop(0)
    categories += [file_path.split("/")[-2]]
    words = []
    for line in lines:
        if line.rstrip() == "":
            continue
        words.extend(tagger.parse(line).strip().split())
    sentences.append(words)

scdv.get_probability_word_vectors(sentences)
doc_vectors = scdv.make_gwbowv(sentences) # 文章ごとのvectorが得られる
print("doc_vectorss %d, sentences %d, shape:%s" % (len(doc_vectors), len(sentences), doc_vectors.shape))
pickle.dump(categories, open("model/categories.pkl", "wb"))
pickle.dump(doc_vectors, open("model/doc_vectors.pkl", "wb"))
