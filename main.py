import MeCab

tagger = MeCab.Tagger('-Owakati')
sentence = "竹やぶ焼けた"
print(tagger.parse(sentence))