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
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

doc_vectors = pickle.load(open("model/doc_vectors.pkl", "rb"))
categories = pickle.load(open("model/categories.pkl", "rb"))

# graph. 2次元に圧縮する
tsne = TSNE(
    n_components=2, random_state=0, verbose=2
)
tsne.fit(doc_vectors)

df = pd.DataFrame(tsne.embedding_[:, 0], columns=['x'])
df['y'] = pd.DataFrame(tsne.embedding_[:, 1])
df['class'] = categories
graph = sns.lmplot(
    data=df, x='x', y='y', hue='class', fit_reg=False, size=8
)
plt.show()