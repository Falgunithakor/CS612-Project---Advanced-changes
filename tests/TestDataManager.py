import unittest
import numpy as np
from sklearn.svm import LinearSVC
from DataManager import DataManager
from FileLoader import FileLoader
from NumpyNormalizer import NumpyNormalizer
from SplitTypes import SplitTypes


class TestDataManager(unittest.TestCase):
    def test_feature_elimination_array_ones(self):
        data_manager = DataManager()
        ones_data = np.ones((37, 397))
        data_manager.set_data(ones_data)
        data_manager.split_data(0.19, 0.62)

        data_manager.run_feature_elimination()
        self.assertEqual(data_manager.inputs[SplitTypes.Test].shape, (7, 395))

    def test_feature_elimination_using_linear_svc_train(self):
        file_path = "../Datasets/HIV_37_Samples/MergedDataset.csv"
        loaded_data = FileLoader.load_file(file_path)
        np.random.seed(5)
        feature_eliminator = LinearSVC(C=0.01, penalty='l1', dual=False)
        the_data_manager = DataManager(feature_eliminator)
        the_data_manager.set_data(loaded_data)
        the_data_manager.split_data(test_split=0.19, train_split=0.62)

        the_data_manager.run_feature_elimination()

        self.assertEqual(the_data_manager.transformed_input[SplitTypes.Train].shape, (23, 27))

    def test_feature_elimination_using_linear_svc_valid(self):
        file_path = "../Datasets/HIV_37_Samples/MergedDataset.csv"
        loaded_data = FileLoader.load_file(file_path)
        np.random.seed(5)
        feature_eliminator = LinearSVC(C=0.01, penalty='l1', dual=False)
        the_data_manager = DataManager(feature_eliminator)
        the_data_manager.set_data(loaded_data)
        the_data_manager.split_data(test_split=0.19, train_split=0.62)

        the_data_manager.run_feature_elimination()

        self.assertEqual(the_data_manager.transformed_input[SplitTypes.Valid].shape, (7, 27))

    def test_feature_elimination_using_linear_svc_test(self):
        file_path = "../Datasets/HIV_37_Samples/MergedDataset.csv"
        loaded_data = FileLoader.load_file(file_path)
        np.random.seed(5)
        feature_eliminator = LinearSVC(C=0.01, penalty='l1', dual=False)
        the_data_manager = DataManager(feature_eliminator)
        the_data_manager.set_data(loaded_data)
        the_data_manager.split_data(test_split=0.19, train_split=0.62)

        the_data_manager.run_feature_elimination()

        self.assertEqual(the_data_manager.transformed_input[SplitTypes.Test].shape, (7, 27))

    def test_normalizer_input(self):
        #Arrange Create the normalizer, Create the data manager, Create the data
        normalizer = NumpyNormalizer()
        data_manager = DataManager(feature_selection_algorithm=None, normalizer=normalizer)
        data = np.array([[-70, -0.5, 0, 0.3, 100, 0]])
        np.random.seed(5)

        #Act set the data
        data_manager.set_data(data)

        #Assert Verify the data was normalized
        result = data_manager.data[:, 0:5]
        expected = np.array([[0.0, 0.40882353, 0.41176471, 0.41352941, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_no_normalizer_input(self):
        #Arrange Create the normalizer, Create the data manager, Create the data
        data_manager = DataManager(feature_selection_algorithm=None, normalizer=None)
        data = np.array([[-70, -0.5, 0, 0.3, 100]])
        np.random.seed(5)

        #Act set the data
        data_manager.set_data(data)

        #Assert Verify the data was normalized
        result = data_manager.data
        expected = data
        np.testing.assert_array_almost_equal(result, expected)

    def test_experiment_all_zeros_no_valid_data_splitting(self):
        the_data_manager = DataManager()
        array_all_zeroes = np.zeros((37, 397))
        the_data_manager.set_data(array_all_zeroes)
        the_data_manager.split_data(test_split=0.19, train_split=0.81)
        print('Data splitting with zeros data:')
        print('Train data shape:', the_data_manager.inputs[SplitTypes.Train].shape)
        print('Test data shape:', the_data_manager.inputs[SplitTypes.Test].shape)
        print('Valid data shape:', the_data_manager.inputs[SplitTypes.Valid].shape)
        self.assertEqual(the_data_manager.inputs[SplitTypes.Train].shape, (30,396))

    def test_experiment_merged_dataset_no_valid_data_splitting(self):
        file_path = "../Datasets/HIV_37_Samples/MergedDataset.csv"
        loaded_data = FileLoader.load_file(file_path)
        the_data_manager = DataManager()
        the_data_manager.set_data(loaded_data)
        the_data_manager.split_data(test_split=0.19, train_split=0.81)
        print('Data splitting with Merged dataset:')
        print('Train data shape:', the_data_manager.inputs[SplitTypes.Train].shape)
        print('Test data shape:', the_data_manager.inputs[SplitTypes.Test].shape)
        print('Valid data shape:', the_data_manager.inputs[SplitTypes.Valid].shape)
        self.assertEqual(the_data_manager.inputs[SplitTypes.Train].shape, (30,396))
