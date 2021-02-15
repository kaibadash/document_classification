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
vectors = [] # cluster数 * 次元の配列の配列
df = pd.DataFrame()
for file_path in files:
    # 少ないデータで試す
    if random.randrange(100) != 0:
        continue
    print("load: " + file_path)

    with open(file_path, 'r') as file:
        text = file.read()
    lines = text.split("\n")
    lines.pop(0)
    lines.pop(0)
    categories += [file_path.split("/")[-2]]
    sentences = []
    for line in lines:
        if line.rstrip() == "":
            continue
        sentences.append(tagger.parse(line).strip().split())
    scdv.get_probability_word_vectors(sentences)
    doc_vector = scdv.make_gwbowv(sentences)
    print("doc_vectors %d" % (len(doc_vector)))
    print(doc_vector)
    vectors += [doc_vector]

print("categories %d vectors %d" % (len(categories), len(vectors)))
df = pd.DataFrame(
    data={'category': categories, 'vectors': vectors},
    columns=['category', 'vectors']
)
print(df)

pickle.dump(df, open("model/vectors.pkl", "wb"))
