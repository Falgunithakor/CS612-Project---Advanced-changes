__author__ = 'Chris'


class FakeFeatureEliminator(object):

    def fit(self, input, target):
        pass

    def transform(self, input):
        return input[:, 0:3]