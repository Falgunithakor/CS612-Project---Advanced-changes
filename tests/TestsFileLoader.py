import unittest
import numpy as np
from DataManager import DataManager
from FileLoader import FileLoader
from SplitTypes import SplitTypes


class TestSequenceFunctions(unittest.TestCase):
    def test_load_CSV_into_numpy(self):

        file_loader = FileLoader()
        file_path = "test_load_csv_into_numpy.csv"
        result = file_loader.load_file(file_path)
        expected = np.array([[1,2],[3,4]])
        self.assertTrue(np.array_equal(result, expected))

    def test_load_MergedCSV_into_numpy(self):

        file_loader = FileLoader()

        file_path = "../Datasets/HIV_37_Samples/MergedDataset.csv"
        result= file_loader.load_file(file_path)
        expected = np.zeros((37,397))
        self.assertTrue(result.shape == expected.shape)

    def test_split_merge_csv_7_7_23(self):

         file_loader = FileLoader()
         data_manager = DataManager()
         file_path = "../Datasets/HIV_37_Samples/MergedDataset.csv"
         result = file_loader.load_file(file_path)
         data_manager.set_data(result)
         data_manager.split_data(test_split=0.19,train_split=0.62)


         valid_and_test_shapes = (7, 397)
         train_shapes = (23, 397)
         expected = np.array([valid_and_test_shapes, valid_and_test_shapes, train_shapes])
         result = np.array([data_manager.datum[SplitTypes.Test].shape, data_manager.datum[SplitTypes.Valid].shape, data_manager.datum[SplitTypes.Train].shape])
         self.assertTrue(np.array_equal(result, expected))


    def test_split_merge_csv_4_25_8(self):
        file_loader = FileLoader()
        data_manager = DataManager()
        file_path = "../Datasets/HIV_37_Samples/MergedDataset.csv"
        result = file_loader.load_file(file_path)
        data_manager.set_data(result)
        data_manager.split_data(test_split=0.11,train_split=0.22)

        test_shapes = np.zeros((4, 397)).shape
        valid_shapes = np.zeros((25,397)).shape
        train_shapes = np.zeros((8, 397)).shape
        expected = np.array([test_shapes, valid_shapes, train_shapes])
        result = np.array([data_manager.datum[SplitTypes.Test].shape, data_manager.datum[SplitTypes.Valid].shape, data_manager.datum[SplitTypes.Train].shape])
        self.assertTrue(np.array_equal(result, expected))

    def test_split_into_target_and_input(self):
        file_loader = FileLoader()
        data_manager = DataManager()
        file_path = "../Datasets/HIV_37_Samples/MergedDataset.csv"
        result = file_loader.load_file(file_path)
        data_manager.set_data(result)
        data_manager.split_data(test_split=0.11,train_split=0.22)
        test_shapes_input = np.zeros((4, 396)).shape
        valid_shapes_input = np.zeros((25,396)).shape
        train_shapes_input = np.zeros((8, 396)).shape
        test_shapes_target = np.zeros((4, )).shape
        valid_shapes_target = np.zeros((25,)).shape
        train_shapes_target = np.zeros((8, )).shape
        expected = np.array([test_shapes_input, valid_shapes_input, train_shapes_input, test_shapes_target, valid_shapes_target, train_shapes_target])
        result = np.array([data_manager.inputs[SplitTypes.Test].shape, data_manager.inputs[SplitTypes.Valid].shape, data_manager.inputs[SplitTypes.Train].shape, data_manager.targets[SplitTypes.Test].shape, data_manager.targets[SplitTypes.Valid].shape, data_manager.targets[SplitTypes.Train].shape])
        self.assertTrue(np.array_equal(result, expected))

if __name__=='__main__':
    unittest.main()

