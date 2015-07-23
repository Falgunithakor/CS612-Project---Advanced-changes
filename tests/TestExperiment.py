import unittest
import numpy as np
from sklearn import svm
from DataManager import DataManager
from Experiment import Experiment
from FakeFeatureEliminator import FakeFeatureEliminator
from FakePredictionModel import FakePredictionModel
from FileLoader import FileLoader
from SplitTypes import SplitTypes


class TestExperiment(unittest.TestCase):
    def test_experiment_all_zeros_r2_1(self):
        the_data_manager = DataManager()
        array_all_zeroes = np.zeros((37, 397))
        the_data_manager.set_data(array_all_zeroes)
        the_data_manager.split_data(test_split=0.19, train_split=0.62)

        the_model = svm.SVR()
        exp = Experiment(the_data_manager, the_model)
        exp.run_experiment()

        r2_train = exp.get_r2(SplitTypes.Train)
        expected = 1.0
        self.assertEqual(r2_train, expected)

    def test_experiment_svm_svr_37dataset_r2_train(self):
        file_path = "../Datasets/HIV_37_Samples/MergedDataset.csv"
        loaded_data = FileLoader.load_file(file_path)
        the_data_manager = DataManager()
        the_data_manager.set_data(loaded_data)
        the_data_manager.split_data(test_split=0.19, train_split=0.62)
        the_model = svm.SVR()
        exp = Experiment(the_data_manager, the_model)
        exp.run_experiment()

        r2_train = exp.get_r2(SplitTypes.Train)
        expected_svm_r2_value = 0.93994377385638073
        self.assertEqual(r2_train, expected_svm_r2_value)

    def test_experiment_svr_37dataset_r2_valid(self):
        file_path = "../Datasets/HIV_37_Samples/MergedDataset.csv"
        loaded_data = FileLoader.load_file(file_path)
        the_data_manager = DataManager()
        the_data_manager.set_data(loaded_data)
        the_data_manager.split_data(test_split=0.19, train_split=0.62)
        the_model = svm.SVR()
        exp = Experiment(the_data_manager, the_model)

        exp.run_experiment()

        r2_valid = exp.get_r2(SplitTypes.Valid)
        expected_svm_r2_value = -0.12569465965376159
        self.assertEqual(r2_valid, expected_svm_r2_value)

    def test_experiment_svr_37dataset_r2_test(self):
        file_path = "../Datasets/HIV_37_Samples/MergedDataset.csv"
        loaded_data = FileLoader.load_file(file_path)
        the_data_manager = DataManager()
        the_data_manager.set_data(loaded_data)
        the_data_manager.split_data(test_split=0.19, train_split=0.62)
        the_model = svm.SVR()
        exp = Experiment(the_data_manager, the_model)

        exp.run_experiment()

        r2_test = exp.get_r2(SplitTypes.Test)
        expected_svm_r2_value = -0.33005242525900247
        self.assertEqual(r2_test, expected_svm_r2_value)

    def test_experiment_sum_of_squares_zeros_test(self):
        the_data_manager = DataManager()
        an_array_of_all_ones = np.ones((37, 397))
        the_model = svm.SVR()
        the_data_manager.set_data(an_array_of_all_ones)
        the_data_manager.split_data(test_split=0.19, train_split=0.62)
        exp = Experiment(the_data_manager, the_model)

        exp.run_experiment()
        sum_of_squares_test = exp.get_sum_of_squares(SplitTypes.Test)

        expected = 0
        self.assertEquals(expected, sum_of_squares_test)

    def test_experiment_sum_of_squares_real37_test(self):
        file_path = "../Datasets/HIV_37_Samples/MergedDataset.csv"
        loaded_data = FileLoader.load_file(file_path)
        the_data_manager = DataManager()
        the_data_manager.set_data(loaded_data)
        the_model = svm.SVR()
        the_data_manager.split_data(test_split=0.19, train_split=0.62)
        exp = Experiment(the_data_manager, the_model)

        exp.run_experiment()
        sum_of_squares_test = exp.get_sum_of_squares(SplitTypes.Test)

        expected = 6.708898437500002

        self.assertAlmostEqual(expected, sum_of_squares_test)

    def test_experiment_sum_of_squares_real37_valid(self):
        file_path = "../Datasets/HIV_37_Samples/MergedDataset.csv"
        loaded_data = FileLoader.load_file(file_path)
        the_data_manager = DataManager()
        the_data_manager.set_data(loaded_data)
        the_model = svm.SVR()
        the_data_manager.split_data(test_split=0.19, train_split=0.62)
        exp = Experiment(the_data_manager, the_model)

        exp.run_experiment()
        sum_of_squares_test = exp.get_sum_of_squares(SplitTypes.Valid)

        expected = 6.0453984375000012

        self.assertAlmostEqual(expected, sum_of_squares_test)

    def test_experiment_sum_of_squares_real37_train(self):
        file_path = "../Datasets/HIV_37_Samples/MergedDataset.csv"
        loaded_data = FileLoader.load_file(file_path)
        the_data_manager = DataManager()
        the_data_manager.set_data(loaded_data)
        the_model = svm.SVR()
        the_data_manager.split_data(test_split=0.19, train_split=0.62)
        exp = Experiment(the_data_manager, the_model)

        exp.run_experiment()
        sum_of_squares_test = exp.get_sum_of_squares(SplitTypes.Train)

        expected = 1.00969921875

        self.assertAlmostEqual(expected, sum_of_squares_test)

    def test_experiment_on_transformed_test(self):
        file_path = "../Datasets/HIV_37_Samples/MergedDataset.csv"
        loaded_data = FileLoader.load_file(file_path)
        feature_eliminator = FakeFeatureEliminator()
        data_manager = DataManager(feature_eliminator)
        data_manager.set_data(loaded_data)
        data_manager.split_data(test_split=0.19, train_split=0.62)
        learning_model = FakePredictionModel()
        exp = Experiment(data_manager, learning_model)

        exp.run_experiment()

        self.assertEquals(1, exp.get_r2(SplitTypes.Test))

    def test_experiment_not_transformed_test(self):
        file_path = "../Datasets/HIV_37_Samples/MergedDataset.csv"
        loaded_data = FileLoader.load_file(file_path)
        data_manager = DataManager()
        data_manager.set_data(loaded_data)
        data_manager.split_data(test_split=0.19, train_split=0.62)
        learning_model = FakePredictionModel()
        exp = Experiment(data_manager, learning_model)

        exp.run_experiment()

        self.assertEquals(0, exp.get_r2(SplitTypes.Test))

