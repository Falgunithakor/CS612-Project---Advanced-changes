import unittest
import numpy as np
from NumpyNormalizer import NumpyNormalizer


class TestNumpyNormalizer(unittest.TestCase):
    def test_normalize_numpy_array(self):
        #Arrange Create Array [[-70,-0.5,0,0.3,100]] , Create Normalizer, Fit Normalizer to Array
        data = np.array([[-70, -0.5, 0, 0.3, 100]])
        normalizer = NumpyNormalizer()
        normalizer.fit(data)
        #Act Normalize the data
        result = normalizer.transform(data)
        #Assert the values of the result
        expected = np.array([[0.0, 0.40882353, 0.41176471, 0.41352941, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_numpy_array2(self):
        #Arrange Create Array [[-1050,-0.35,0,0.13,4]] , Create Normalizer, Fit Normalizer to Array
        data = np.array([[-1050, -0.35, 0, 0.13, 4]])
        normalizer = NumpyNormalizer()
        normalizer.fit(data)
        #Act Normalize the data
        result = normalizer.transform(data)
        #Assert the values of the result
        expected = np.array([[0., 0.99587287, 0.99620493, 0.99632827, 1.]])
        np.testing.assert_array_almost_equal(result, expected)