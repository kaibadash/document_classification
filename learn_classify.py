import glob
import os
import random
import pickle
import MeCab
import pandas as pd
import const
import warnings
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from document import Document
from sklearn.utils.testing import all_estimators
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

# ドキュメントのvectorから分類器を生成する
doc_vectors = pickle.load(open("model/doc_vectors.pkl", "rb"))
categories = pickle.load(open("model/categories.pkl", "rb"))

print("doc_vectors:%s categories:%s" % (len(doc_vectors), len(categories)))
print(doc_vectors)
x_train, x_test, y_train, y_test = train_test_split(doc_vectors, categories, test_size = 0.2, shuffle = True)

# classifierのアルゴリズム全てを取得する --- (※1)
allAlgorithms = all_estimators(type_filter="classifier")
warnings.simplefilter("error")

for(name, algorithm) in allAlgorithms :
  try :
    # 各アリゴリズムのオブジェクトを作成 --- (※2)
    if(name == "LinearSVC") :
      clf = algorithm(max_iter = 10000)
    else:
      clf = algorithm()

    # 学習して、評価する --- (※3)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("==========", name," accuracy_score = ", accuracy_score(y_test, y_pred))
  
  # Warningのの内容を表示し、Exceptionは無視する --- (※4)
  except Warning as w :
    print("\033[33m"+"Warning："+"\033[0m", name, ":", w.args)
  except Exception as e :
    print(e)
    pass
