from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection.univariate_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, Binarizer, Imputer, KernelCenterer, LabelBinarizer, LabelEncoder, \
    MultiLabelBinarizer, Normalizer, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.svm.classes import LinearSVC, SVR
from sklearn import svm, linear_model
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, RFE, RFECV

from DataManager import DataManager
from Experiment import Experiment
from FileLoader import FileLoader
from NumpyNormalizer import NumpyNormalizer
from ScikitNormalizer import ScikitNormalizer
from SplitTypes import SplitTypes
import numpy as np
from sklearn.grid_search import GridSearchCV
#####################################################################################
#NOTE THIS IS A MAIN FILE AND SHOULD ONLY BE USED TO COMPOSE THE FINAL EXPERIMENT   #
#ONLY UNIT TESTS ARE INDICATIVE OF THE SYSTEM FUNCTIONALITY                         #
#####################################################################################

normalizers = [None,NumpyNormalizer(), ScikitNormalizer(),MinMaxScaler()]
#normalizers = [None,MinMaxScaler()]

'''feature_eliminators = [None,VarianceThreshold(), SelectKBest(f_regression,k=5),LinearSVC(),
                       ExtraTreesClassifier(compute_importances=True, random_state=0),
                       RFE(estimator=SVR(kernel="linear"),n_features_to_select=5),
                       RFECV(estimator=SVR(kernel="linear")), TruncatedSVD(),
                       SelectKBest(f_regression,k=40)]
'''

feature_eliminators = [VarianceThreshold(),
                       ExtraTreesClassifier(compute_importances=True, random_state=0),
                       RFE(estimator=SVR(kernel="linear"),n_features_to_select=5),
                       RFECV(estimator=SVR(kernel="linear")), TruncatedSVD(),
                        ]

#the_models = [linear_model.BayesianRidge(), svm.SVR(), svm.SVR(kernel='rbf')]
the_models = [linear_model.BayesianRidge(), svm.SVR(), RandomForestRegressor(), LinearRegression()]

normalizers = [StandardScaler()
                #,NumpyNormalizer()#, ScikitNormalizer()
                #, MinMaxScaler() ,Binarizer()
                # ,Imputer(), KernelCenterer()
                # ,Normalizer() ,
                ]
for normalizer in normalizers:
    for feature_eliminator in feature_eliminators:
    #for k_value in range(5, 20):
    #for k_value in range(13, 14):
        for the_model in the_models:
            file_path = "../Datasets/HIV_37_Samples/MergedDataset.csv"
            loaded_data = FileLoader.load_file(file_path)
            #feature_eliminator = SelectKBest(f_regression,k=k_value)
            the_data_manager = DataManager(feature_eliminator,normalizer=normalizer)
            the_data_manager.set_data(loaded_data)
            the_data_manager.split_data(test_split=0.15, train_split=0.70)

            exp = Experiment(the_data_manager, the_model)

            exp.run_experiment()
            if(exp.get_r2(SplitTypes.Train) > 0 and exp.get_r2(SplitTypes.Valid) > 0 and exp.get_r2(SplitTypes.Test) >  0):
                print(
                feature_eliminator.get_support(indices=True),
                type(normalizer).__name__, type(feature_eliminator).__name__ ,
                type(the_model).__name__,
                "Fitness", exp.fitness_matrix[0],
                "Train", exp.get_r2(SplitTypes.Train),"Valid", exp.get_r2(SplitTypes.Valid),"Test",
                exp.get_r2(SplitTypes.Test),
                exp.get_sum_of_squares(SplitTypes.Test))

                '''
                FileLoader.write_model_in_file(output_filename
                                                        , feature_eliminator.get_support(indices=True)
                                                        #, self.feature_selector.fitness_matrix[population_idx]
                                                        , exp.get_r2(SplitTypes.Train)
                                                        , exp.get_r2(SplitTypes.Valid)
                                                        , exp.get_r2(SplitTypes.Test)
                                                    )
                '''
            #exp.plot_true_vs_predicted(SplitTypes.Train)