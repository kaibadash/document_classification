import glob
import os
import pickle
import MeCab
import const
from scdv import SparseCompositeDocumentVectors, build_word2vec

# word2vecの学習を行う

files = glob.glob("tmp/text/**/*.txt")
sentences = []
tagger = MeCab.Tagger('-Owakati')
for file_path in files:
    print("load: " + file_path)
    with open(file_path, 'r') as file:
        text = file.read()
    lines = text.split("\n")
    lines.pop(0)
    lines.pop(0)

    for line in lines:
        if line.rstrip() == "":
            continue
        sentences.append(tagger.parse(line).strip().split())

model = build_word2vec(sentences, const.VECTOR_NUM, 5, 5, 1)
pickle.dump(model, open("model/model.pkl", "wb"))
