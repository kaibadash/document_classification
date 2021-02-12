import glob
import os
import pickle
import MeCab
from scdv import SparseCompositeDocumentVectors, build_word2vec

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

model = build_word2vec(sentences, 100, 5, 5, 1)
pickle.dump(model, open("model/model.pkl", "wb"))

vec = SparseCompositeDocumentVectors(
    model,
    2,
    100,
    "model/gmm_cluster.pkl",
    "model/gmm_prob_cluster.pkl"
)

# test
test = "■デジ通の記事をもっと見る ・iPhone 4Sの音声認識機能は使える 新しいiPadでも使える音声認識 ・英語の音声認識で英会話練習 自分の発音を客観的に知る方法 ・サービス終了後はどうなる！電子書籍は永遠に読めるのか ・キンドルが日本に参入する？Amazonの電子書籍をおさらい ・iPhoneが据え置きゲーム機も殺す？多機能化するスマートフォン"
print(vec.make_gwbowv(test))
