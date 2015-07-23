from sklearn.preprocessing import Normalizer

__author__ = 'Chris'


class ScikitNormalizer(object):
    def __init__(self):
        self.data_normalizer = Normalizer()

    def fit(self, data):
        self.data_normalizer.fit(data)

    def transform(self, data):
        return (self.data_normalizer.transform(data) + 1) / 2