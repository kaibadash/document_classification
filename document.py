class Document:
    category = ""
    vector = []

    def __init__(self, category, vector):
        self.category = category
        self.vector = vector

    # カテゴリ, 文書ベクトルを返す
    def toArray(self):
        print([len(v) for v in self.vector])
        print([self.category].extend(self.vector))
        return [self.category].extend(self.vector)