import numpy as np

__author__ = 'Chris'


class NumpyNormalizer(object):
    def fit(self, data):
        pass

    def transform(self, data):
        result = np.zeros(data.shape)
        for i in range(0, data.shape[0]):
            x = data[i]
            result[i] = (x - min(x)) / (max(x) - min(x))
        return result