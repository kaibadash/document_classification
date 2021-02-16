import glob
import os
import pickle
import MeCab
from scdv import SparseCompositeDocumentVectors, build_word2vec
import const

# 学習済みモデルからdocument vectorを生成するテスト
s = "吾輩は猫である。■昨日は暖かかった。"

tagger = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
tagger.parse("")

node = tagger.parseToNode(s).next
while node.next:
    print(node.surface, node.feature)
    node = node.next