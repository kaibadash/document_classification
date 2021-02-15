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

# ドキュメントのvectorから分類器を生成する
df = pickle.load(open("model/vectors.pkl", "rb"))

# split label and data
y = df["category"]
x = df["vectors"]

for label in y:
    print("label %d %s" % (len(label), label))

for data in x:
    print("data %d" % len(data))
    print(data)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True)


# learning
clf = SVC()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print("正解率 = " , accuracy_score(y_test, y_pred))
