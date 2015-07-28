import unittest
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection.univariate_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, Binarizer, Imputer, KernelCenterer, LabelBinarizer, LabelEncoder, \
    MultiLabelBinarizer, Normalizer, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.svm.classes import LinearSVC, SVR
from sklearn import svm, linear_model
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, RFE, RFECV
import sys
import time
from DataManager import DataManager
from Experiment import Experiment
from FakeFeatureEliminator import FakeFeatureEliminator
from FakePredictionModel import FakePredictionModel
from FileLoader import FileLoader
from NumpyNormalizer import NumpyNormalizer
from ScikitNormalizer import ScikitNormalizer
from SplitTypes import SplitTypes



class TestAlgoCombination(unittest.TestCase):
    def getnormalizer(self, param):
        normalizer = None
        if param == 'None':
            normalizer = None
        elif param == 'StandardScaler':
            normalizer = StandardScaler()
        elif param == 'NumpyNormalizer':
            normalizer = NumpyNormalizer()
        elif param == 'ScikitNormalizer':
            normalizer = ScikitNormalizer()
        elif param == 'MinMaxScaler':
            normalizer = MinMaxScaler()
        elif param == 'Binarizer':
            normalizer = Binarizer()
        elif param == 'Imputer':
            normalizer = Imputer()
        elif param == 'KernelCenterer':
            normalizer = KernelCenterer()
        elif param == 'Normalizer':
            normalizer = Normalizer()
        return normalizer

    def getfeature_eliminator(self, param):
        feature_eliminator = None
        if param == 'VarianceThreshold':
            feature_eliminator = VarianceThreshold()
        elif param == 'RFE':
            feature_eliminator = RFE(estimator=SVR(kernel="linear"),n_features_to_select=5)
        elif param == 'RFECV':
            feature_eliminator = RFECV(estimator=SVR(kernel="linear"))
        return feature_eliminator

    def get_model(self, param):
        model = None
        if param == "BayesianRidge":
            model = linear_model.BayesianRidge()
        elif param == "SVR":
            model = svm.SVR()
        elif param == 'LinearRegression':
            model = LinearRegression()
        print(model)
        return model

    def test_experiment(self):
        output_filename_header = FileLoader.create_output_file()
        time.sleep(1)
        loaded_algorithm_combinations = FileLoader.read_csv_file("../Datasets/test.csv")
        file_path = "../Datasets/HIV_37_Samples/MergedDataset.csv"
        loaded_data = FileLoader.load_file(file_path)
        #feature_eliminator = SelectKBest(f_regression,k=k_value)

        print(loaded_algorithm_combinations[0])
        output_filename = FileLoader.create_output_file()

        for i in range(0,80):
            normalizer = self.getnormalizer(loaded_algorithm_combinations[i][0])

            feature_eliminator = self.getfeature_eliminator(loaded_algorithm_combinations[i][1])
            the_model = self.get_model(loaded_algorithm_combinations[i][2])

            print 'taking ', type(normalizer).__name__,'and feature selector ', type(feature_eliminator).__name__ , 'model', type(the_model).__name__
            FileLoader.write_model_in_file(output_filename_header
                                                        ,type(normalizer).__name__
                                                        ,type(feature_eliminator).__name__
                                                        , type(the_model).__name__
                                                        , ''
                                                        ,''
                                                        , ''
                                                        ,''
                                                        , ''
                                                    )



            the_data_manager = DataManager(feature_eliminator,normalizer=normalizer)
            the_data_manager.set_data(loaded_data)
            the_data_manager.split_data(test_split=0.15, train_split=0.70)
            exp = Experiment(the_data_manager, the_model)

            exp.run_experiment()
            #arr_selected = feature_eliminator.get_support(indices=True)

            #if(exp.get_r2(SplitTypes.Train) > 0 and exp.get_r2(SplitTypes.Valid) > 0 and exp.get_r2(SplitTypes.Test) >  0):
            FileLoader.write_model_in_file(output_filename
                                                    ,type(normalizer).__name__
                                                    ,type(feature_eliminator).__name__
                                                    , type(the_model).__name__
                                                    , ''
                                                    , exp.fitness_matrix[0]
                                                    , exp.get_r2(SplitTypes.Train)
                                                    , exp.get_r2(SplitTypes.Valid)
                                                    , exp.get_r2(SplitTypes.Test)
                                                )
