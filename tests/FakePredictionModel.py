__author__ = 'Chris'


class FakePredictionModel(object):
    def __init__(self):
        self.predict_input = None

    def fit(self, input, target):
        pass

    def predict(self, input):
        self.predict_input = input
        return input[:, 0].ravel()

    def score(self, input, target):
        if self.predict_input.shape == (7, 3):
            return 1
        else:
            return 0