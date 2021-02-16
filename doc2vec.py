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
samples = 0 # n分の1をサンプルとする。0だと全部が対象
model = pickle.load(open("model/model.pkl", "rb"))
scdv = SparseCompositeDocumentVectors(
    model,
    const.CLUSTER_NUM,
    const.VECTOR_NUM,
    "model/gmm_cluster.pkl",
    "model/gmm_prob_cluster.pkl"
)

files = glob.glob("tmp/text/**/*.txt")
tagger = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
tagger.parse("")
categories = []
sentences = [] # 文章ごとの単語の配列
df = pd.DataFrame()
for file_path in files:
    # 少ないデータで試す
    if samples != 0 and random.randrange(samples) != 0:
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
        node = tagger.parseToNode(line).next
        while node:
            cat = node.feature.split(",")[0]
            if cat == "記号":
                node = node.next
                continue
            # 基本形を使う
            words.append(node.feature.split(",")[6])
            node = node.next
    sentences.append(words)

scdv.get_probability_word_vectors(sentences)
doc_vectors = scdv.make_gwbowv(sentences) # カテゴリ, vectors の配列
print("doc_vectorss %d, sentences %d, shape:%s" % (len(doc_vectors), len(sentences), doc_vectors.shape))
pickle.dump(categories, open("model/categories.pkl", "wb"))
pickle.dump(doc_vectors, open("model/doc_vectors.pkl", "wb"))
