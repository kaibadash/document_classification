import glob
import os
import random
import pickle
import MeCab
import pandas as pd
import const
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from document import Document

# ドキュメントのvectorから分類器を生成する
doc_vectors = pickle.load(open("model/doc_vectors.pkl", "rb"))
categories = pickle.load(open("model/categories.pkl", "rb"))

print("doc_vectors:%s categories:%s" % (len(doc_vectors), len(categories)))
print(doc_vectors)
x_train, x_test, y_train, y_test = train_test_split(doc_vectors, categories, test_size = 0.2, shuffle = True)

# learning
clf = SVC()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print("accuracy_score = " , accuracy_score(y_test, y_pred))
