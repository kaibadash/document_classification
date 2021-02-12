import glob
import os
import pickle
import MeCab
from scdv import SparseCompositeDocumentVectors, build_word2vec

tagger = MeCab.Tagger('-Owakati')
model = pickle.load(open("model/model.pkl", "rb"))
scdv = SparseCompositeDocumentVectors(
    model,
    2,
    100,
    "model/gmm_cluster.pkl",
    "model/gmm_prob_cluster.pkl"
)

sentences = []
document = "ThinkPad X1 Hybridは使用するCPUがx86(インテルCore iなど)からARMに切り替わるハイブリッドなPCだが、これと同時にOSも切り替わる。\niPhoneは、2000年代に世界的な大ヒットを記録したデジタルオーディオプレーヤーであるiPodの派生製品であり、2007年当時の最新型iPodの機能と携帯電話が統合した端末として誕生した"
for line in document.split("\n"):
    sentences.append(tagger.parse(line).strip().split())
scdv.get_probability_word_vectors(sentences)
print(scdv.make_gwbowv(sentences))
