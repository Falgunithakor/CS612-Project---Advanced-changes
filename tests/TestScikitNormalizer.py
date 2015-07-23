import unittest
import numpy as np
from ScikitNormalizer import ScikitNormalizer


class TestScikitNormalizer(unittest.TestCase):
    def test_normalize_numpy_array(self):
        #Arrange Create Array [[-70,-0.5,0,0.3,100]] , Create Normalizer, Fit Normalizer to Array
        data = np.array([[-70, -0.5, 0, 0.3, 100]])
        normalizer = ScikitNormalizer()
        normalizer.fit(data)
        #Act Normalize the data
        result = normalizer.transform(data)
        #Assert the values of the result
        expected = np.array([[ 0.2132721 ,  0.49795194,  0.5 ,  0.50122883,  0.90961129]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_numpy_array2(self):
        #Arrange Create Array [[-1050,-0.35,0,0.13,4]] , Create Normalizer, Fit Normalizer to Array
        data = np.array([[-1050, -0.35, 0, 0.13, 4]])
        normalizer = ScikitNormalizer()
        normalizer.fit(data)
        #Act Normalize the data
        result = normalizer.transform(data)
        #Assert the values of the result
        expected = np.array([[  3.65968771e-06,   4.99833335e-01,   5.00000000e-01, 5.00061904e-01,   5.01904748e-01]])
        np.testing.assert_array_almost_equal(result, expected)