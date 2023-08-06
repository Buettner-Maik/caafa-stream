import os
import sys
import argparse

sys.path.append(os.path.join(os.getcwd(), '.'))#'../osm'))

import additions.constants as consts
import pandas as pd
import multiprocessing as mp
from datetime import datetime

from additions.transformers.MissTransformer import MissTransformer
from additions.transformers.ColumnSelector import ColumnSelector
from additions.transformers.CatImputer import CategoricalRememberingNaNImputer
from osm.data_streams.imputer.PairImputer import CatToCatClassifier, CatToNumClassifier, NumToCatClassifier, PairRegImputer, FeaturePairImputer
from osm.transformers.ColTransform import ColumnTransformerNoCloning

from osm.data_streams.algorithm.framework import FrameWork

from osm.data_streams.active_learner.strategy.pool_based.no_active_learner import NoActiveLearner

from osm.data_streams.oracle.simple_oracle import SimpleOracle

from osm.data_streams.windows.sliding_window import SlidingWindow
from osm.data_streams.windows.fixed_length_window import FixedLengthWindow

from osm.data_streams.budget_manager.incremental_percentile_filter import IncrementalPercentileFilter, TrendCorrectedIncrementalPercentileFilter
from osm.data_streams.budget_manager.simple_budget_manager import SimpleBudgetManager
from osm.data_streams.budget_manager.no_budget_manager import NoBudgetManager

from osm.data_streams.active_feature_acquisition.no_feature_acquisition import NoActiveFeatureAcquisition
from osm.data_streams.active_feature_acquisition.random_acquisition import RandomFeatureAcquisition
from osm.data_streams.active_feature_acquisition.supervised_merit_ranking.average_euclidean_distance import SingleWindowAED, MultiWindowAED, SWAEDFeatureCorrelationCorrected
from osm.data_streams.active_feature_acquisition.supervised_merit_ranking.entropy_based import SingleWindowIG, SingleWindowSU
from osm.data_streams.active_feature_acquisition.supervised_merit_ranking.fpi_aided import SWAED_FPITS, SWAED_FPITS_LOG
from osm.data_streams.active_feature_acquisition.supervised_merit_ranking.smr_test_methods import SWAED_T, SWAED_Q, SWAED_I, SWAED_D, SWAED_C, SWAED_P, SWAED_E, SWAED_H, SWAED_H2, SWIG_H2, SWSU_H2, SWAED_IMAX, SWAED_IEXP, SWAED_AQ, SWAED_II, SWAEDFCC_II, SWAED_RPRI, SWAED_SPRI, SWAEDFCC_RPRI, SWAEDFCC_SPRI, SWAED_TRPRI, SWAED_TRPRI_0, SWAED_TRPRI_25, SWAED_TRPRI_50, SWAED_TRPRI_75, SWAED_TRPRI_100, SWAED_TRPRI_200, SWAED_BPRI, SWAED_QT, SWAED_IQ, SWAED_FPI, RA_FPI, NAFA_FPI, SWAED_NDB, SWAED_SSBQ, SWAED_SSBQ_FPI, SWAED_IMPQ, SWAED_IMPTS, SWAED_IMPTS_2, SWAED_IMPTS_6, SWAED_IMPTS_10, SWAED_IMPQ2, PCFI_DTC
from osm.data_streams.active_feature_acquisition.supervised_merit_ranking.classifier_feature_importance import PoolClassifierFeatureImportanceAFA

from osm.data_streams.active_feature_acquisition.supervised_merit_ranking.smr_feature_set_selection import KBestSMRFSS, KPickyBestSMRFSS, KRandomSMRFSS, KQualitySMRFSS, KQGainSMRFSS, KBudgetAwareQGainSMRFSS, KBestImputerAlteredMeritSMRFSS, KBestImputerThresholdSMRFSS

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier, LinearRegression
from additions.sgd_predict_proba_fix import SGDPredictFix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder#, KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import clone as sk_clone

class Dataset():
    """
    container of all information needed for a run of a categorical data set
    can run an analysis with the the provided osm framework
    """
    def __init__(self, name:str, directory:str, filename:str, 
                 cat_cols:list, num_cols:list, targetcol:str,
                 pid_mins:list = [], pid_maxs:list = [], categories = None):
        """
        saves all meta data for a dataset
        :param name: the name of the dataset
        :param directory: the directory of the raw data
        :param filename: the name of the raw data file
        :param cat_cols: all columns with categorical data except the target column to be considered
        :param num_cols: all columns with numerical data except the target column to be considered
        :param targetcol: the target column containing the labels
        :param pid_mins: the min values the PID is initialized
        :param pid_maxs: the max values the PID is initialized
        :param categories: provide all categorical values for each categorical column
        alternatively leave None to automatically get values
        """
        self.name = name
        self.directory = directory
        self.filename = filename
        
        #remove targetcol from cols to be considered
        try:
            cat_cols.remove(targetcol)
        except ValueError:
            pass
        try:
            num_cols.remove(targetcol)
        except ValueError:
            pass

        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.targetcol = targetcol
        self.pid_mins = pid_mins
        self.pid_maxs = pid_maxs
        if categories is None:
            self.categories = self._get_categories_list(df=pd.read_pickle(directory + 'raw_data.pkl.gzip'))
        else:
            self.categories = categories

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def summary_str(self, pre_folder="", i_str="", m_str=""):
        """
        get a summary string
        """
        filepath = self.directory
        
        if pre_folder: filepath = filepath + pre_folder + '/'
        if not i_str: filepath = filepath + 'default/'
        else: filepath = filepath + i_str + '/'
        if not m_str: filepath = filepath + 'default/summary.pkl.gzip'
        else: filepath = filepath + m_str + '/summary.pkl.gzip'
        return filepath     

    def get_default_pipeline(self):
        """
        returns the default pipeline
        """
        cat_trans = Pipeline(steps=[
            ('imputer', CategoricalRememberingNaNImputer(
                categories=self._get_categories_dict())),
            ('encoder', OneHotEncoder(#handle_unknown='ignore', 
                                      categories=self.categories,
                                      sparse=False))
        ])
        num_trans = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_trans, self.num_cols),
            ('cat', cat_trans, self.cat_cols)
        ])

        feature_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])
        return feature_pipeline
        
    def get_iterative_pipeline(self):
        """
        returns the default pipeline
        """
        cat_trans = Pipeline(steps=[
            ('imputer', CategoricalRememberingNaNImputer(
                categories=self._get_categories_dict())),
            ('encoder', OneHotEncoder(#handle_unknown='ignore', 
                                      categories=self.categories,
                                      sparse=False))
        ])
        num_trans = Pipeline(steps=[
            ('imputer', IterativeImputer()),
            ('scaler', StandardScaler())
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_trans, self.num_cols),
            ('cat', cat_trans, self.cat_cols)
        ])

        feature_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])
        return feature_pipeline

    def get_featurepair_pipeline(self, pr_importance='best', pr_nrmse='maxmin', pr_pairs=25,
                                 num_num_model=LinearRegression(),
                                 num_cat_model=NumToCatClassifier(),
                                 cat_num_model=CatToNumClassifier(),
                                 cat_cat_model=CatToCatClassifier(),
                                 manual_fit=True):
        """
        returns a pair feature imputer
        """
        imputer = FeaturePairImputer(num_cols=self.num_cols, cat_cols=self.cat_cols, 
                                     pair_window_size=pr_pairs,
                                     importance=pr_importance, rmse_norm=pr_nrmse,
                                     num_num_model=num_num_model,
                                     num_cat_model=num_cat_model,
                                     cat_num_model=cat_num_model,
                                     cat_cat_model=cat_cat_model,
                                     manual_fit=manual_fit)
        cat_trans = OneHotEncoder(categories=self.categories, sparse=False)
        num_trans = StandardScaler()
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_trans, self.num_cols),
            ('cat', cat_trans, self.cat_cols)
        ])
        # selector = ColumnTransformer(transformers=[
        #     ('imputer', imputer, self.num_cols + self.cat_cols)
        # ])

        feature_pipeline = Pipeline(steps=[
            #('selector', selector),
            ('selector', ColumnSelector(self.num_cols + self.cat_cols)),
            ('imputer', imputer),
            ('preprocessor', preprocessor)
        ])

        return feature_pipeline, imputer

    def get_pairreg_pipeline(self, pr_importance='best', pr_nrmse='maxmin', pr_pairs=25, manual_fit=True):
        """
        returns the default pipeline with an PairRegImputer and the PairRegImputer
        """
        imputer = PairRegImputer(pair_window_size=pr_pairs, manual_fit=manual_fit,
                                 importance=pr_importance, rmse_norm=pr_nrmse)
        cat_trans = Pipeline(steps=[
            ('imputer', CategoricalRememberingNaNImputer(
                categories=self._get_categories_dict())),
            ('encoder', OneHotEncoder(#handle_unknown='ignore', 
                                      categories=self.categories,
                                      sparse=False))
        ])
        num_trans = Pipeline(steps=[
            ('imputer', imputer),
            ('scaler', StandardScaler())
        ])
        preprocessor = ColumnTransformerNoCloning(transformers=[
            ('num', num_trans, self.num_cols),
            ('cat', cat_trans, self.cat_cols)
        ])

        feature_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])
        return feature_pipeline, imputer

    def load_df(self):
        df = pd.read_csv(self.directory + self.filename, index_col=False)
        if any(self.cat_cols):
            cats = df[self.cat_cols]#.astype('object')
        else:
            cats = pd.DataFrame()
        if any(self.num_cols):
            nums = df[self.num_cols].astype('float64')
        else:
            nums = pd.DataFrame()
        targ = df[self.targetcol]
        return pd.concat([cats, nums, targ], axis=1, sort=False)

    def generate_feature_costs(self, strategy='equal'):
        """
        generates a dict containing the cost of the feature columns according to some specified strategy
        :param strategy: the method by which to generate the costs
        'equal' sets all feature costs to 1
        'increasing' assigns feature costs according to their order beginning with categorical and ending with numerical features
        'decreasing' assigns feature costs according to their reverse order beginning with numerical features and ending with categorical features
        'cheaper_cats' sets cost for all categorical features to 1 and sets cost for all numerical features to 2
        'cheaper_nums' sets cost for all numerical features to 1 and sets cost for all categorical features to 2
        """
        costs = {}
        if strategy == 'equal':
            costs = {col: 1 for col in self.cat_cols + self.num_cols}
        elif strategy == 'increasing':
            costs = {col: i+1 for i, col in enumerate(self.cat_cols + self.num_cols)}
        elif strategy == 'decreasing':
            cols = self.cat_cols + self.num_cols
            cols.reverse()
            costs = {col: i+1 for i, col in enumerate(cols)}
        elif strategy == 'cheaper_cats':
            for col in self.cat_cols:
                costs[col] = 1
            for col in self.num_cols:
                costs[col] = 2
        elif strategy == 'cheaper_nums':
            for col in self.cat_cols:
                costs[col] = 2
            for col in self.num_cols:
                costs[col] = 1
        else:
            raise ValueError(f"strategy {strategy} is not defined")
        return costs

    def _get_categories_list(self, df:pd.DataFrame):
        """
        get all possible values for categorical columns in the dataframe
        """
        categories = []
        for col in self.cat_cols:
            categories.append(df[col].unique())
        return categories

    def _get_categories_dict(self):
        """
        converts self.categories into a dict object
        """
        cat_vals = {}
        i = 0
        for cat in self.categories:
            cat_vals[self.cat_cols[i]] = cat
            i += 1
        return cat_vals

    def _add_miss_cols(self, df:pd.DataFrame, miss_chances:dict):
        """
        appends "_org" to all columns to be replaced with columns with nan misses
        """
        misser = MissTransformer(miss_chances=miss_chances)
        return misser.transform(df)

    def _encode_target_col(self, df:pd.DataFrame):
        """
        many a sklearn learners require target features to be numerical categories to work
        this converts the target columns labels into numerical values
        """
        df[self.targetcol] = LabelEncoder().fit_transform(df[self.targetcol])
        return df

    def _save_df(self, df:pd.DataFrame, summary_file:pd.DataFrame, 
                name:str, sub_folder:str):
        """
        saves a batch of data and updates the summary_file with a link of that data
        :param df: the data to be saved
        :param name: the name of the file
        :param folder: the folder into which the data batches will be saved in
        """
        if not os.path.exists(self.directory + sub_folder):
            os.makedirs(self.directory + sub_folder)
        pd.to_pickle(df, self.directory + sub_folder + '/' + name)
        #filepath = sub_folder + '/' + name
        filepath = name
        row = pd.DataFrame.from_dict(data={'filename':[filepath]})
        return summary_file.append(row, ignore_index=True)

    def _replace_with_missing(self, df:pd.DataFrame, index:int):
        """
        replaces miss column with original values up until index
        """
        for col in df:
            orgcol = col + '_org'
            if orgcol in df:
                df[col][:index] = df[orgcol][:index]
        return df

    def _create_batch_split(self, df:pd.DataFrame,
                        batch_size:int, ild_extra_rows:int,
                        sub_folder:str, summary_str:str, shuffle:bool):         
        """
        splits a dataframe into an initially labeled dataset with each class label
        at least represented once plus ild_extra_rows rows extra
        divides the rest of data into batch_size rows
        logs all created batch files in a summary file and serializes all data
        via pickle into files
        """
        summary = pd.DataFrame({'filename':[]})

        data = pd.DataFrame()
        
        if shuffle:
            #give each label a representation in ild          
            for category in df[self.targetcol].unique():
                shuffled = df.loc[lambda x: df[self.targetcol] == category, :].sample(frac=1)
                data = data.append(shuffled[:1])
                df.drop(index=shuffled.index[0], inplace=True)
            data = self._replace_with_missing(df=data, index=data.shape[0])
            
            #shuffle all data
            df = df.sample(frac=1)            
            
        df = self._replace_with_missing(df=df, index=ild_extra_rows)

        #add extra data to ild
        data = data.append(df[:ild_extra_rows])
        #check if first batch actually contains all labels
        if set(df[self.targetcol]) != set(data[self.targetcol]):
            raise ValueError("The initial data must contain all possible labels.")
        
        df.drop(df.index[:ild_extra_rows], inplace=True)
        summary = self._save_df(data, summary, 'labeled_set.pkl.gzip', sub_folder)

        #create batch files
        for i in range(0,df.shape[0],batch_size):
            summary = self._save_df(df[i:i+batch_size], 
                            summary, 'data{0}.pkl.gzip'.format(i),
                            sub_folder)

        summary.reset_index(inplace=True,drop=True)
        pd.to_pickle(summary, summary_str)

    def do_preprocessing(self, miss_chances:dict, batch_size:int, 
                    ild_extra_rows:int, sub_folder:str = "prepared", 
                    summary_str:str = None, shuffle:bool = True):
        """
        automatically converts a csv file with header row and ',' as separator
        into batches of pickled pandas.DataFrames with their data in specified
        columns in miss_chances altered to include misses of the likelihood
        specified in miss_chances
        the complete column is readded to the DataFrame under the original name + '_org'
        returns the categorical values of all categorical features
        """
        if summary_str is None: summary_str = self.summary_str()
        df = self.load_df()
        df = self._encode_target_col(df=df)
        pd.to_pickle(df, self.directory + 'raw_data.pkl.gzip')

        df = self._add_miss_cols(df=df, 
                        miss_chances=miss_chances)
        self._create_batch_split(df=df, batch_size=batch_size, 
                        ild_extra_rows=ild_extra_rows, 
                        sub_folder=sub_folder, 
                        summary_str=summary_str,
                        shuffle=shuffle)

    def do_framework(self, window, base_estimator, active_learner = None, 
                     oracle = None, feature_pipeline = None,
                     feature_acquisition = None, budget_manager = None,
                     summary_str:str = None,
                     pre_folder_name:str = "", post_folder_name:str = "",
                     evaluation_strategy = None, ild_timepoint = None, 
                     post_label_call=None,
                     overwrite_summary:bool=False,
                     debug:bool=False):
        """
        does an active acquisition through the framework
        saves the summary.pkl.gzip also as tab separated csv file
        :param window: the window used for the stream
        :param budget_manager: the budget manager used by the AFA algorithm
        :param base_estimator: the final classifier to make predictions
        :param feature_pipeline: the pipeline used for the framework
        :param active_learner: the active learner to be used
        :param oracle: the oracle used in the framework
        :param feature_acquisition: the feature acquisition strategy used in the framework
        :param post_label_call: a function on the labeled data right before it is added to the window
        """
        # somehow directly starting a function through multiprocessing
        # removes window_data attribute from SlidingWindow despite being set to None
        # thus reset window_data back to None
        window.window_data = None
        if summary_str is None: summary_str = self.summary_str()
        if oracle is None:
            oracle = SimpleOracle()
        if active_learner is None:
            active_learner = NoActiveLearner(budget = 1.0, 
                                             oracle=oracle, 
                                             target_col_name=self.targetcol)
        if budget_manager is None:
            budget_manager = NoBudgetManager()
        if feature_acquisition is None:
            feature_acquisition = NoActiveFeatureAcquisition(budget_manager=budget_manager,
                                                            target_col_name=self.targetcol,
                                                            acquisition_costs={},
                                                            debug=True)
        if feature_pipeline is None:
            feature_pipeline = self.get_default_pipeline()

        framework = FrameWork(
            summary_file=summary_str,
            base_estimator=base_estimator, 
            feature_pipeline=feature_pipeline,
            target_col_name=self.targetcol,
            ild_timepoint=ild_timepoint,
            feature_acquisition=feature_acquisition,
            active_learner=active_learner,
            window=window,
            evaluation_strategy=evaluation_strategy,
            results_path=self.directory,
            post_label_call=post_label_call,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name,
            overwrite_summary=overwrite_summary,
            debug=debug)

        framework.process_data_stream()

        os.rename(os.path.join(framework.dir_result, framework.summary_filename), os.path.join(framework.dir_result, 'summary.pkl.gzip'))
        pd.read_pickle(os.path.join(framework.dir_result, 'summary.pkl.gzip')).to_csv(os.path.join(framework.dir_result, 'summary.csv'), sep="\t")

    def do_AFA_lower_bound(self, window, base_estimator, summary_str=None,
                           pre_folder_name="", post_folder_name="", 
                           overwrite_summary=False):
        """
        does the lower bound calculation for the AFA task
        """
        print("Starting AFA lower bound on " + self.name)
        budget_manager = NoBudgetManager()
        feature_acquisition = NoActiveFeatureAcquisition(
                                target_col_name=self.targetcol,
                                acq_set_size=0,
                                budget_manager=budget_manager)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager,
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name,
            overwrite_summary=overwrite_summary)
        
    def do_AFA_pairreg_bound(self, window, base_estimator, summary_str=None,
                             pre_folder_name="", post_folder_name="",
                             overwrite_summary=False):
        """
        does the lower bound calculation with a feature pair imputer for AFA task
        """
        print("Starting AFA pairreg bound on " + self.name)
        if not any(self.cat_cols): pipeline, imputer = self.get_pairreg_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
        else: pipeline, imputer = self.get_featurepair_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
        budget_manager = SimpleBudgetManager(0.0, initial_budget=0, default_budget_gain=0)
        # def add_examples(X):
        #     print(imputer.num_feature_pairs[0, 1])
        #     imputer.add_examples(X[self.num_cols + self.cat_cols])
        # post_label_call = add_examples
        post_label_call = lambda X: imputer.add_examples(X[self.num_cols + self.cat_cols])
        
        
        lambda X: imputer.add_examples(X[self.num_cols + self.cat_cols])
        feature_acquisition = RandomFeatureAcquisition(
                                target_col_name=self.targetcol, 
                                budget_manager=budget_manager,
                                budget_option=[('once', 0.0)],
                                acq_set_size=0,
                                columns=self.cat_cols + self.num_cols)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name,
            post_label_call=post_label_call,
            overwrite_summary=overwrite_summary)


    def do_AFA_upper_bound(self, window, base_estimator, summary_str=None,
                           pre_folder_name="", post_folder_name="",
                           overwrite_summary=False):
        """
        does the upper bound calculation for the AFA task
        """
        print("Starting AFA upper bound on " + self.name)
        budget_manager = NoBudgetManager()
        feature_acquisition = RandomFeatureAcquisition(
                                target_col_name=self.targetcol, 
                                budget_manager=budget_manager,
                                acq_set_size=len(self.cat_cols + self.num_cols),
                                columns=self.cat_cols + self.num_cols)

        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager,
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name,
            overwrite_summary=overwrite_summary)

    def do_AFA_random(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size,
                      summary_str=None, pre_folder_name="", post_folder_name="",
                      overwrite_summary=False):
        """
        does an AFA task with random acquisition strategy
        """
        print("Starting random AFA on " + self.name)
        feature_acquisition = RandomFeatureAcquisition(
                                target_col_name=self.targetcol, 
                                budget_manager=budget_manager,
                                budget_option=budget_option,
                                acq_set_size=acq_set_size,
                                acquisition_costs=acquisition_costs,
                                columns=self.cat_cols+self.num_cols)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name,
            overwrite_summary=overwrite_summary)

    def do_AFA_SAED(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="",
                    overwrite_summary=False, debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAED on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SingleWindowAED(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name,
            overwrite_summary=overwrite_summary)

    def do_AFA_MAED(self, window, base_estimator, budget_manager, budget_option, acquisition_costs,  acq_set_size, feature_set_selector, aed_window_size, 
                    summary_str=None, pre_folder_name="", post_folder_name="",
                    overwrite_summary=False, debug=True):
        """
        does an AFA task with a multi window
        average euclidean distance acquisition strategy
        """
        print("Starting MAED on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = MultiWindowAED(ild=window,
                                window_size=aed_window_size,
                                target_col_name=self.targetcol, 
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name,
            overwrite_summary=overwrite_summary)
    
    def do_AFA_SIG(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector, 
                    summary_str=None, pre_folder_name="", post_folder_name="",
                    overwrite_summary=False, debug=True):
        """
        does an AFA task with a single window 
        information gain acquisition strategy
        """
        print("Starting SIG on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SingleWindowIG(window=window,
                                target_col_name=self.targetcol, 
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),#self.get_digitized_pipeline(bins),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager,
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name,
            overwrite_summary=overwrite_summary)

    def do_AFA_SSU(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector, 
                    summary_str=None, pre_folder_name="", post_folder_name="",
                    overwrite_summary=False, debug=True):
        """
        does an AFA task with a single window 
        symmetric uncertainty acquisition strategy
        """
        print("Starting SSU on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SingleWindowSU(window=window,
                                target_col_name=self.targetcol, 
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),#self.get_digitized_pipeline(bins),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name,
            overwrite_summary=overwrite_summary)        

    def do_AFA_PCFI(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting PCFI on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        pipeline = self.get_default_pipeline()
        feature_acquisition = PoolClassifierFeatureImportanceAFA(window=window,
                                fi_classifier=sk_clone(base_estimator),
                                pipeline=pipeline,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)

    def do_AFA_PCFI_DTC(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting PCFI_DTC on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        pipeline = self.get_default_pipeline()
        feature_acquisition = PCFI_DTC(window=window,
                                fi_classifier=DecisionTreeClassifier(),
                                pipeline=pipeline,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)

    def do_AFA_RA_FPI(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="",
                    overwrite_summary=False, debug=True):
        """
        does an AFA task with random acquisition strategy
        """
        print("Starting RA_FPI on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        if any(self.cat_cols): 
            pipeline, imputer = self.get_featurepair_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
        else: 
            pipeline, imputer = self.get_pairreg_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
        post_label_call = lambda X: imputer.add_examples(X[self.num_cols + self.cat_cols])
        feature_acquisition = RA_FPI(
                                target_col_name=self.targetcol, 
                                budget_manager=budget_manager,
                                budget_option=budget_option,
                                acq_set_size=acq_set_size,
                                acquisition_costs=acquisition_costs,
                                columns=self.cat_cols+self.num_cols)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name,
            post_label_call=post_label_call,
            overwrite_summary=overwrite_summary)

    def do_AFA_SAED_FPI(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="",
                    overwrite_summary=False, debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAED_FPI on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        if any(self.cat_cols): 
            pipeline, imputer = self.get_featurepair_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
        else: 
            pipeline, imputer = self.get_pairreg_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
        post_label_call = lambda X: imputer.add_examples(X[self.num_cols + self.cat_cols])
        feature_acquisition = SWAED_FPI(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name,
            post_label_call=post_label_call,
            overwrite_summary=overwrite_summary)

    def do_AFA_SAED_NDB(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="",
                    overwrite_summary=False, debug=True):
        """
        various alternative runs
        """
        print("Starting SAED_NDB on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SWAED_NDB(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=False,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name,
            overwrite_summary=overwrite_summary)

    def do_AFA_SAED_IMAX(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="",
                    overwrite_summary=False, debug=True):
        """
        various alternative runs
        """
        print("Starting SAED_IMAX on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SWAED_IMAX(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name,
            overwrite_summary=overwrite_summary)
            
    def do_AFA_SAED_IEXP(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="",
                    overwrite_summary=False, debug=True):
        """
        various alternative runs
        """
        print("Starting SAED_IEXP on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SWAED_IEXP(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name,
            overwrite_summary=overwrite_summary)

    def do_AFA_SAED_SSBQ(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="",
                    overwrite_summary=False, debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAED_SSBQ on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SWAED_SSBQ(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name,
            overwrite_summary=overwrite_summary)

    def do_AFA_SAED_SSBQ_FPI(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="",
                    overwrite_summary=False, debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAED_SSBQ_FPI on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        if any(self.cat_cols): 
            pipeline, imputer = self.get_featurepair_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
        else: 
            pipeline, imputer = self.get_pairreg_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
        post_label_call = lambda X: imputer.add_examples(X[self.num_cols + self.cat_cols])
        feature_acquisition = SWAED_SSBQ_FPI(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name,
            post_label_call=post_label_call,
            overwrite_summary=overwrite_summary)

    def do_AFA_SAED_IMPQ(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="",
                    overwrite_summary=False, debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAED_IMPQ on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        if any(self.cat_cols): 
            pipeline, imputer = self.get_featurepair_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
        else: 
            pipeline, imputer = self.get_pairreg_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
        post_label_call = lambda X: imputer.add_examples(X[self.num_cols + self.cat_cols])
        feature_acquisition = SWAED_IMPQ(window=window,
                                imputer=imputer,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name,
            post_label_call=post_label_call,
            overwrite_summary=overwrite_summary)
        if hasattr(feature_set_selector, 'stats_lenInst'):
            print('inst = ', feature_set_selector.stats_lenInst)
            print('A = ', feature_set_selector.stats_lenA)
            print('inst_A = ', feature_set_selector.stats_lenInstA)
            for key, (i, a, ia) in enumerate(zip(feature_set_selector.stats_lenInst.values(), feature_set_selector.stats_lenA.values(), feature_set_selector.stats_lenInstA.values())):
                print(key, i, a, ia)
        
    def do_AFA_SAED_IMPQ2(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="",
                    overwrite_summary=False, debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAED_IMPQ2 on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        if any(self.cat_cols): 
            pipeline, imputer = self.get_featurepair_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
        else: 
            pipeline, imputer = self.get_pairreg_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
        post_label_call = lambda X: imputer.add_examples(X[self.num_cols + self.cat_cols])
        feature_acquisition = SWAED_IMPQ2(window=window,
                                imputer=imputer,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name,
            post_label_call=post_label_call,
            overwrite_summary=overwrite_summary)
        if hasattr(feature_set_selector, 'stats_lenInst'):
            print('inst = ', feature_set_selector.stats_lenInst)
            print('A = ', feature_set_selector.stats_lenA)
            print('inst_A = ', feature_set_selector.stats_lenInstA)
            for key, (i, a, ia) in enumerate(zip(feature_set_selector.stats_lenInst.values(), feature_set_selector.stats_lenA.values(), feature_set_selector.stats_lenInstA.values())):
                print(key, i, a, ia)

    def do_AFA_SAED_FPITS(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector, skip_percentile,
                    summary_str=None, pre_folder_name="", post_folder_name="",
                    overwrite_summary=False, debug=True):
        """
        does an AFA task with a single window
        average euclidean distance acquisition strategy
        """
        print("Starting SAED_FPITS on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        if any(self.cat_cols): 
            pipeline, imputer = self.get_featurepair_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
        else: 
            pipeline, imputer = self.get_pairreg_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
        post_label_call = lambda X: imputer.add_examples(X[self.num_cols + self.cat_cols])
        feature_acquisition = SWAED_FPITS(window=window,
                                fpi=imputer,
                                skip_percentile=skip_percentile,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name,
            post_label_call=post_label_call,
            overwrite_summary=overwrite_summary)    

    def do_AFA_SAED_FPITS_LOG(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector, skip_percentile,
                    summary_str=None, pre_folder_name="", post_folder_name="",
                    overwrite_summary=False, debug=True):
        """
        does an AFA task with a single window
        average euclidean distance acquisition strategy
        """
        print("Starting SAED_FPITS_LOG on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        if any(self.cat_cols): 
            pipeline, imputer = self.get_featurepair_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
        else: 
            pipeline, imputer = self.get_pairreg_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
        post_label_call = lambda X: imputer.add_examples(X[self.num_cols + self.cat_cols])
        feature_acquisition = SWAED_FPITS_LOG(window=window,
                                fpi=imputer,
                                skip_percentile=skip_percentile,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name,
            post_label_call=post_label_call,
            overwrite_summary=overwrite_summary)    

















    # TESTMETHODS BELOW















    # def do_AFA_SAED_IMPTS(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
    #                 summary_str=None, pre_folder_name="", post_folder_name="",
    #                 overwrite_summary=False, debug=True):
    #     """
    #     does an AFA task with a single window 
    #     average euclidean distance acquisition strategy
    #     """
    #     print("Starting SAED_IMPTS on " + self.name)
    #     categories = self._get_categories_dict()
    #     for col in self.num_cols:
    #         categories[col] = []
    #     if any(self.cat_cols): 
    #         pipeline, imputer = self.get_featurepair_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
    #     else: 
    #         pipeline, imputer = self.get_pairreg_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
    #     post_label_call = lambda X: imputer.add_examples(X[self.num_cols + self.cat_cols])
    #     feature_acquisition = SWAED_IMPTS(window=window,
    #                             imputer=imputer,
    #                             max_thr=.2,
    #                             skip_percentile=0.1,
    #                             target_col_name=self.targetcol,
    #                             budget_manager=budget_manager,
    #                             dynamic_budget_threshold=True,
    #                             budget_option=budget_option,
    #                             acquisition_costs=acquisition_costs,
    #                             acq_set_size=acq_set_size,
    #                             feature_selection=feature_set_selector,
    #                             categories=categories,
    #                             debug=debug)
    #     self.do_framework(window=window, base_estimator=base_estimator,
    #         active_learner=None, oracle=None,
    #         feature_pipeline = pipeline,
    #         feature_acquisition=feature_acquisition, 
    #         budget_manager=budget_manager, 
    #         summary_str=summary_str,
    #         pre_folder_name=pre_folder_name,
    #         post_folder_name=post_folder_name,
    #         post_label_call=post_label_call,
    #         overwrite_summary=overwrite_summary)

    # def do_AFA_SAED_IMPTS_2(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
    #                 summary_str=None, pre_folder_name="", post_folder_name="",
    #                 overwrite_summary=False, debug=True):
    #     """
    #     does an AFA task with a single window 
    #     average euclidean distance acquisition strategy
    #     """
    #     print("Starting SAED_IMPTS_SD1 on " + self.name)
    #     categories = self._get_categories_dict()
    #     for col in self.num_cols:
    #         categories[col] = []
    #     if any(self.cat_cols): 
    #         pipeline, imputer = self.get_featurepair_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
    #     else: 
    #         pipeline, imputer = self.get_pairreg_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
    #     post_label_call = lambda X: imputer.add_examples(X[self.num_cols + self.cat_cols])
    #     feature_acquisition = SWAED_IMPTS_2(window=window,
    #                             imputer=imputer,
    #                             max_thr=10,
    #                             skip_percentile=0.02,
    #                             target_col_name=self.targetcol,
    #                             budget_manager=budget_manager,
    #                             dynamic_budget_threshold=True,
    #                             budget_option=budget_option,
    #                             acquisition_costs=acquisition_costs,
    #                             acq_set_size=acq_set_size,
    #                             feature_selection=feature_set_selector,
    #                             categories=categories,
    #                             debug=debug)
    #     self.do_framework(window=window, base_estimator=base_estimator,
    #         active_learner=None, oracle=None,
    #         feature_pipeline = pipeline,
    #         feature_acquisition=feature_acquisition, 
    #         budget_manager=budget_manager, 
    #         summary_str=summary_str,
    #         pre_folder_name=pre_folder_name,
    #         post_folder_name=post_folder_name,
    #         post_label_call=post_label_call,
    #         overwrite_summary=overwrite_summary)

    # def do_AFA_SAED_IMPTS_6(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
    #                 summary_str=None, pre_folder_name="", post_folder_name="",
    #                 overwrite_summary=False, debug=True):
    #     """
    #     does an AFA task with a single window 
    #     average euclidean distance acquisition strategy
    #     """
    #     print("Starting SAED_IMPTS_SD1 on " + self.name)
    #     categories = self._get_categories_dict()
    #     for col in self.num_cols:
    #         categories[col] = []
    #     if any(self.cat_cols): 
    #         pipeline, imputer = self.get_featurepair_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
    #     else: 
    #         pipeline, imputer = self.get_pairreg_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
    #     post_label_call = lambda X: imputer.add_examples(X[self.num_cols + self.cat_cols])
    #     feature_acquisition = SWAED_IMPTS_6(window=window,
    #                             imputer=imputer,
    #                             max_thr=10,
    #                             skip_percentile=0.06,
    #                             target_col_name=self.targetcol,
    #                             budget_manager=budget_manager,
    #                             dynamic_budget_threshold=True,
    #                             budget_option=budget_option,
    #                             acquisition_costs=acquisition_costs,
    #                             acq_set_size=acq_set_size,
    #                             feature_selection=feature_set_selector,
    #                             categories=categories,
    #                             debug=debug)
    #     self.do_framework(window=window, base_estimator=base_estimator,
    #         active_learner=None, oracle=None,
    #         feature_pipeline = pipeline,
    #         feature_acquisition=feature_acquisition, 
    #         budget_manager=budget_manager, 
    #         summary_str=summary_str,
    #         pre_folder_name=pre_folder_name,
    #         post_folder_name=post_folder_name,
    #         post_label_call=post_label_call,
    #         overwrite_summary=overwrite_summary)

    # def do_AFA_SAED_IMPTS_10(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
    #                 summary_str=None, pre_folder_name="", post_folder_name="",
    #                 overwrite_summary=False, debug=True):
    #     """
    #     does an AFA task with a single window 
    #     average euclidean distance acquisition strategy
    #     """
    #     print("Starting SAED_IMPTS_SD1 on " + self.name)
    #     categories = self._get_categories_dict()
    #     for col in self.num_cols:
    #         categories[col] = []
    #     if any(self.cat_cols): 
    #         pipeline, imputer = self.get_featurepair_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
    #     else: 
    #         pipeline, imputer = self.get_pairreg_pipeline(pr_pairs=25, pr_importance='best', manual_fit=True)
    #     post_label_call = lambda X: imputer.add_examples(X[self.num_cols + self.cat_cols])
    #     feature_acquisition = SWAED_IMPTS_10(window=window,
    #                             imputer=imputer,
    #                             max_thr=10,
    #                             skip_percentile=0.1,
    #                             target_col_name=self.targetcol,
    #                             budget_manager=budget_manager,
    #                             dynamic_budget_threshold=True,
    #                             budget_option=budget_option,
    #                             acquisition_costs=acquisition_costs,
    #                             acq_set_size=acq_set_size,
    #                             feature_selection=feature_set_selector,
    #                             categories=categories,
    #                             debug=debug)
    #     self.do_framework(window=window, base_estimator=base_estimator,
    #         active_learner=None, oracle=None,
    #         feature_pipeline = pipeline,
    #         feature_acquisition=feature_acquisition, 
    #         budget_manager=budget_manager, 
    #         summary_str=summary_str,
    #         pre_folder_name=pre_folder_name,
    #         post_folder_name=post_folder_name,
    #         post_label_call=post_label_call,
    #         overwrite_summary=overwrite_summary)

    def do_AFA_SAEDFCC(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAEDFCC on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        pipeline = self.get_default_pipeline()
        feature_acquisition = SWAEDFeatureCorrelationCorrected(window=window,
                                pipeline=pipeline,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)

    def do_AFA_SAED_AQ(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAED_AQ on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SWAED_AQ(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)

    def do_AFA_SAED_QT(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAED_QT on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        pipeline, imputer = self.get_pairreg_pipeline(pr_importance='reciprocal')
        feature_acquisition = SWAED_QT(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
        if hasattr(feature_set_selector, 'stats'):
            print(feature_set_selector.stats)

    def do_AFA_SAED_TRPRI(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAED_TRPRI on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        pipeline, imputer = self.get_pairreg_pipeline(pr_importance='reciprocal')
        feature_acquisition = SWAED_TRPRI(window=window,
                                target_col_name=self.targetcol,
                                pipeline=pipeline,
                                imp_conf_stddev=-1,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
    def do_AFA_SAED_TRPRI_0(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAED_TRPRI_0 on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        pipeline, imputer = self.get_pairreg_pipeline(pr_importance='reciprocal')
        feature_acquisition = SWAED_TRPRI_0(window=window,
                                target_col_name=self.targetcol,
                                pipeline=pipeline,
                                imp_conf_stddev=0,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
    def do_AFA_SAED_TRPRI_25(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAED_TRPRI_25 on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        pipeline, imputer = self.get_pairreg_pipeline(pr_importance='reciprocal')
        feature_acquisition = SWAED_TRPRI_25(window=window,
                                target_col_name=self.targetcol,
                                pipeline=pipeline,
                                imp_conf_stddev=-0.25,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
    def do_AFA_SAED_TRPRI_50(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAED_TRPRI_50 on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        pipeline, imputer = self.get_pairreg_pipeline(pr_importance='reciprocal')
        feature_acquisition = SWAED_TRPRI_50(window=window,
                                target_col_name=self.targetcol,
                                pipeline=pipeline,
                                imp_conf_stddev=-0.5,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
    def do_AFA_SAED_TRPRI_75(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAED_TRPRI_75 on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        pipeline, imputer = self.get_pairreg_pipeline(pr_importance='reciprocal')
        feature_acquisition = SWAED_TRPRI_75(window=window,
                                target_col_name=self.targetcol,
                                pipeline=pipeline,
                                imp_conf_stddev=-0.75,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
    def do_AFA_SAED_TRPRI_100(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAED_TRPRI_100 on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        pipeline, imputer = self.get_pairreg_pipeline(pr_importance='reciprocal')
        feature_acquisition = SWAED_TRPRI_100(window=window,
                                target_col_name=self.targetcol,
                                pipeline=pipeline,
                                imp_conf_stddev=-1,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
    def do_AFA_SAED_TRPRI_200(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAED_TRPRI_200 on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        pipeline, imputer = self.get_pairreg_pipeline(pr_importance='reciprocal')
        feature_acquisition = SWAED_TRPRI_200(window=window,
                                target_col_name=self.targetcol,
                                pipeline=pipeline,
                                imp_conf_stddev=-2,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)

    def do_AFA_SAED_RPRI(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAED_RPRI on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        pipeline, imputer = self.get_pairreg_pipeline(pr_importance='reciprocal')
        if hasattr(feature_set_selector, "imputer"):
            feature_set_selector.imputer = imputer
        feature_acquisition = SWAED_RPRI(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
        print(feature_set_selector.stats)

    def do_AFA_SAEDFCC_RPRI(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAEDFCC_RPRI on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        pipeline, imputer = self.get_pairreg_pipeline(pr_importance='reciprocal')
        if hasattr(feature_set_selector, "imputer"):
            feature_set_selector.imputer = imputer
        feature_acquisition = SWAEDFCC_RPRI(window=window,
                                imputer=imputer,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)

    def do_AFA_SAED_SPRI(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAED_SPRI on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        pipeline, imputer = self.get_pairreg_pipeline(pr_importance='sum_difference')
        if hasattr(feature_set_selector, "imputer"):
            feature_set_selector.imputer = imputer
        feature_acquisition = SWAED_SPRI(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)

    def do_AFA_SAEDFCC_SPRI(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAEDFCC_SPRI on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        
        pipeline, imputer = self.get_pairreg_pipeline(pr_importance='sum_difference')
        if hasattr(feature_set_selector, "imputer"):
            feature_set_selector.imputer = imputer
        feature_acquisition = SWAEDFCC_SPRI(window=window,
                                imputer=imputer,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)

    def do_AFA_SAED_BPRI(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAED_BPRI on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        pipeline, imputer = self.get_pairreg_pipeline(pr_importance='best')
        if hasattr(feature_set_selector, "imputer"):
            feature_set_selector.imputer = imputer
        feature_acquisition = SWAED_BPRI(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)

    def do_AFA_SAED_II(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAED_II on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SWAED_II(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_iterative_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)

    def do_AFA_SAEDFCC_II(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        does an AFA task with a single window 
        average euclidean distance acquisition strategy
        """
        print("Starting SAEDFCC_II on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        pipeline = self.get_iterative_pipeline()
        feature_acquisition = SWAEDFCC_II(window=window,
                                pipeline=pipeline,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = pipeline,
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)

    def do_AFA_SAED_Q(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        various alternative runs
        """
        print("Starting SAED_Q on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SWAED_Q(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
            
    def do_AFA_SAED_I(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        various alternative runs
        """
        print("Starting SAED_I on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SWAED_I(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
    
    def do_AFA_SAED_D(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        various alternative runs
        """
        print("Starting SAED_D on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SWAED_D(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
            
    def do_AFA_SAED_C(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        various alternative runs
        """
        print("Starting SAED_C on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SWAED_C(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
    
    def do_AFA_SAED_P(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        various alternative runs
        """
        print("Starting SAED_P on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SWAED_P(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
    
    def do_AFA_SAED_E(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        various alternative runs
        """
        print("Starting SAED_E on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SWAED_E(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
    
    def do_AFA_SAED_H(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        various alternative runs
        """
        print("Starting SAED_H on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SWAED_H(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
            
    def do_AFA_SAED_H2(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        various alternative runs
        """
        print("Starting SAED_H2 on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SWAED_H2(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
            
    def do_AFA_SIG_H2(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        various alternative runs
        """
        print("Starting SIG_H2 on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SWIG_H2(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
            
    def do_AFA_SSU_H2(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        various alternative runs
        """
        print("Starting SSU_H2 on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SWSU_H2(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
    
    
    def do_AFA_test(self, window, base_estimator, budget_manager, budget_option, acquisition_costs, acq_set_size, feature_set_selector,
                    summary_str=None, pre_folder_name="", post_folder_name="", debug=True):
        """
        various alternative runs
        """
        print("Starting TEST on " + self.name)
        categories = self._get_categories_dict()
        for col in self.num_cols:
            categories[col] = []
        feature_acquisition = SWAEDTest3(window=window,
                                target_col_name=self.targetcol,
                                budget_manager=budget_manager,
                                dynamic_budget_threshold=True,
                                budget_option=budget_option,
                                acquisition_costs=acquisition_costs,
                                acq_set_size=acq_set_size,
                                feature_selection=feature_set_selector,
                                categories=categories,
                                debug=debug)
        self.do_framework(window=window, base_estimator=base_estimator,
            active_learner=None, oracle=None,
            feature_pipeline = self.get_default_pipeline(),
            feature_acquisition=feature_acquisition, 
            budget_manager=budget_manager, 
            summary_str=summary_str,
            pre_folder_name=pre_folder_name,
            post_folder_name=post_folder_name)
                     
def abalone():
    #4177 instances
    #8 features + 1 class
    #3 classes -> 
    return Dataset(
        name='abalone',
        directory=consts.DIR_CSV + '/abalone/',
        filename='abalone.csv',
        cat_cols=[],
        num_cols=['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
                'Viscera weight', 'Shell weight', 'Rings'],
        pid_mins=[0, 0, 0, 0, 0, 0, 0, 0],
        pid_maxs=[1, 1, 1, 3, 2, 1, 1, 30],
        targetcol='Sex')
def adult():
    #32561 instances
    #13 features + 1 class
    #2 classes -> 
    return Dataset(
        name='adult',
        directory=consts.DIR_CSV + '/adult/',
        filename='adult.csv',
        cat_cols=['workclass', 'education', 'marital-status', 'occupation', 
                'relationship', 'race', 'sex', 'native-country'],
        num_cols=['age', 'capital-gain', 'capital-loss', 'hours-per-week'],
        pid_mins=[18, 0, 0, 0],
        pid_maxs=[100, 100000, 5000, 100],
        targetcol='label')
def airlines():
    #539383 instances
    #7 features + 1 class
    #2 classes ->
    return Dataset(
        name='airlines',
        directory=consts.DIR_CSV + '/airlines/',
        filename='airlines.csv',
        cat_cols=['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek'],
        num_cols=['Flight', 'Time', 'Length'],
        pid_mins=[1, 10, 23],
        pid_maxs=[7814, 1439, 655],
        targetcol='Delay')
def cfpdss(): #correlated feature pairs drifting synthetic stream
    #13000 instances
    #10 features + 1 class
    # 2 classes ->
    return Dataset(
        name='cfpdss',
        directory=consts.DIR_CSV + '/cfpdss/',
        filename='raw_data.pkl.gzip.csv',
        cat_cols=['c5', 'c6', 'c7', 'c8', 'c9'],
        num_cols=['n0', 'n1', 'n2', 'n3', 'n4'],
        pid_mins=[0, 0, 0, 0, 0],
        pid_maxs=[0, 0, 0, 0, 0],
        targetcol='class')
def electricity():
    #45312 instances
    #8 features + 1 class
    #2 classes -> 
    return Dataset(
        name='electricity',
        directory=consts.DIR_CSV + '/electricity/',
        filename='elecNormNew.csv',
        cat_cols=['day'],
        num_cols=['date', 'period', 'nswprice', 'nswdemand', 'vicprice', 'vicdemand',
        'transfer'],
        pid_mins=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        pid_maxs=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        targetcol='class')
def forest():
    #581012 instances
    #53 features + 1 class
    #7 classes -> 
    return Dataset(
        name='forest',
        directory=consts.DIR_CSV + '/forest/',
        filename='covtype.csv',
        cat_cols=['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
        'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
        'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
        'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15',
        'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
        'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25',
        'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
        'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35',
        'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40'],
        num_cols=['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'],
        pid_mins=[2000, 0, 0, 0, -500, 0, 50, 50, 50, 0],
        pid_maxs=[3000, 360, 50, 1000, 500, 5000, 300, 300, 300, 7000],
        targetcol='class')
def intrusion():
    #494021 instances
    #41 features + 1 class
    #2 classes ->
    return Dataset(
        name='intrusion',
        directory=consts.DIR_CSV + '/intrusion/',
        filename='intrusion_10percent.csv',
        cat_cols=['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login',
        'is_guest_login'],
        num_cols=['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent',
        'hot', 'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 
        'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'],
        pid_mins=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0],
        pid_maxs=[1000, 100000, 100000, 5, 10, 10, 10, 10, 10, 10,
                  10, 10, 10, 10, 10, 300, 300, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 300, 300, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0],
        targetcol='class')
def magic():
    #19020 instances
    #10 features + 1 class
    #2 classes -> 
    return Dataset(
        name='magic',
        directory=consts.DIR_CSV + '/magic/',
        filename='magic04.csv',
        cat_cols=[],
        num_cols=['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 
                'fM3Long', 'fM3Trans', 'fAlpha', 'fDist'],
        pid_mins=[5, 0, 2, 0, 0, -400, -300, -200, 0, 1],
        pid_maxs=[330, 250, 5, 1, 1, 500, 250, 200, 90, 500],
        targetcol='class')
def nursery():
    #12960 instances
    #8 features + 1 class
    #5 classes -> 
    return Dataset(
        name='nursery',
        directory=consts.DIR_CSV + '/nursery/',
        filename='nursery.csv',
        cat_cols=['parents', 'has_nurs', 'form', 'children', 'housing', 'finance',
                'social', 'health'],
        num_cols=[],
        pid_mins=[],
        pid_maxs=[],
        targetcol='label')
def occupancy():
    #20560 instances
    #6 features + 1 class
    #2 classes ->
    return Dataset(
        name='occupancy',
        directory=consts.DIR_CSV + '/occupancy/',
        filename='data.csv',
        cat_cols=['Weekday'],
        num_cols=['Hour', 'Minute', 'Temperature', 'Humidity', 'Light', 'CO2', 
                'HumidityRatio'],
        pid_mins=[0, 0, 15, 0, 0, 400, 0],
        pid_maxs=[23, 59, 35, 100, 2000, 2000, 1],
        targetcol='Occupancy')
def pendigits():
    #10992 instances
    #16 features + 1 class
    #10 classes -> 
    return Dataset(
        name='pendigits',
        directory=consts.DIR_CSV + '/pendigits/',
        filename='pendigits.csv',
        cat_cols=[],
        num_cols=['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'X5', 'Y5',
                'X6', 'Y6', 'X7', 'Y7', 'X8', 'Y8'],
        pid_mins=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        pid_maxs=[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        targetcol='Digit')
"""
def poker():
    #25010 instances
    #10 features + 1 class
    #10 classes -> 
    return Dataset(
        name='poker',
        directory=consts.DIR_CSV + '/poker_hand/',
        filename='poker-hand.csv',
        cat_cols=['S1', 'S2', 'S3', 'S4', 'S5'],
        num_cols=['C1', 'C2', 'C3', 'C4', 'C5'],
        targetcol='CLASS')
def popularity():
    #REGRESSION
    #39644 instances
    # features + 1 class
    #10 classes -> 
    raise NotImplementedError("Regression Tasks aren't possible yet")
    return Dataset(
        name='popularity',
        directory = consts.DIR_CSV + '/online_news_popularity/',
        filename = 'OnlineNewsPopularity.csv',
        cat_cols = [],
        num_cols = [],
        targetcol = 'shares')
"""
def sea():
    #60000 instances; snythetic
    # 3 features + 1 class
    # 2 classes -> [0, 1]; 4 concepts at 15k insts. each; fx + fy > [8, 9, 7, 9.5] +- 10% noise
    return Dataset(
        name='sea',
        directory=consts.DIR_CSV + '/sea/',
        filename='sea.csv',
        cat_cols=[],
        num_cols=['f1', 'f2', 'f3'],
        pid_mins=[0, 0, 0],
        pid_maxs=[10, 10, 10],
        targetcol='label')
def xor():
    #500000 instances, synthetic
    # 5 features + 1 class
    # 2 classes -> [0, 1]; Xor operation on all determinant features (D)
    return Dataset(
        name='xor',
        directory=consts.DIR_CSV + '/xor_test/',
        filename='n_500000_f_5_d_3.csv',
        cat_cols=[],
        num_cols=['D0', 'D1', 'F2', 'D3', 'F4'],
        pid_mins=[0, 0, 0, 0, 0],
        pid_maxs=[1, 1, 1, 1, 1],
        targetcol='Label')
def xorf3d2():
    #10000 instances, synthetic
    # 3 features + 1 class
    # 2 classes -> [0, 1]; Xor operation on all determinant features (D)
    return Dataset(
        name='xorf3d2',
        directory=consts.DIR_CSV + '/xor_test/f3d2/',
        filename='n_10000_f_3_d_2.csv',
        cat_cols=[],
        num_cols=['D0', 'F1', 'D2'],
        pid_mins=[0]*3,
        pid_maxs=[1]*3,
        targetcol='Label')
def xorf5d3():
    #10000 instances, synthetic
    # 5 features + 1 class
    # 2 classes -> [0, 1]; Xor operation on all determinant features (D)
    return Dataset(
        name='xorf5d3',
        directory=consts.DIR_CSV + '/xor_test/f5d3/',
        filename='n_10000_f_5_d_3.csv',
        cat_cols=[],
        num_cols=['F0', 'D1', 'F2', 'D3', 'D4'],
        pid_mins=[0]*5,
        pid_maxs=[1]*5,
        targetcol='Label')
def xorf7d4():
    #10000 instances, synthetic
    # 7 features + 1 class
    # 2 classes -> [0, 1]; Xor operation on all determinant features (D)
    return Dataset(
        name='xorf7d4',
        directory=consts.DIR_CSV + '/xor_test/f7d4/',
        filename='n_10000_f_7_d_4.csv',
        cat_cols=[],
        num_cols=['F0', 'F1', 'D2', 'D3', 'D4', 'D5', 'F6'],
        pid_mins=[0]*7,
        pid_maxs=[1]*7,
        targetcol='Label')
def xorf9d5():
    #10000 instances, synthetic
    # 9 features + 1 class
    # 2 classes -> [0, 1]; Xor operation on all determinant features (D)
    return Dataset(
        name='xorf9d5',
        directory=consts.DIR_CSV + '/xor_test/f9d5/',
        filename='n_10000_f_9_d_5.csv',
        cat_cols=[],
        num_cols=['D0', 'F1', 'D2', 'F3', 'F4', 'F5', 'D6', 'D7', 'D8'],
        pid_mins=[0]*9,
        pid_maxs=[1]*9,
        targetcol='Label')

def evenoddf9d2():
    #10000 instances, synthetic
    # 9 features + 1 class
    # 2 classes -> [0, 1]; EvenOdd operation on the sum all dependent features (D); 0: even, 1: odd
    return Dataset(
        name='evenoddf9d2',
        directory=consts.DIR_CSV + '/evenodd/f9d2/',
        filename='n_10000_f_9_d_2.csv',
        cat_cols=[],
        num_cols=['D0', 'F1', 'F2', 'F3', 'F4', 'F5', 'D6', 'F7', 'F8'],
        pid_mins=[0]*9,
        pid_maxs=[1]*9,
        targetcol='Label')
def evenoddf9d3():
    #10000 instances, synthetic
    # 9 features + 1 class
    # 2 classes -> [0, 1]; EvenOdd operation on the sum all dependent features (D); 0: even, 1: odd
    return Dataset(
        name='evenoddf9d3',
        directory=consts.DIR_CSV + '/evenodd/f9d3/',
        filename='n_10000_f_9_d_3.csv',
        cat_cols=[],
        num_cols=['F0', 'D1', 'F2', 'D3', 'F4', 'F5', 'F6', 'D7', 'F8'],
        pid_mins=[0]*9,
        pid_maxs=[1]*9,
        targetcol='Label')
def evenoddf9d4():
    #10000 instances, synthetic
    # 9 features + 1 class
    # 2 classes -> [0, 1]; EvenOdd operation on the sum all dependent features (D); 0: even, 1: odd
    return Dataset(
        name='evenoddf9d5',
        directory=consts.DIR_CSV + '/evenodd/f9d4/',
        filename='n_10000_f_9_d_4.csv',
        cat_cols=[],
        num_cols=['D0', 'D1', 'D2', 'F3', 'F4', 'D5', 'F6', 'F7', 'F8'],
        pid_mins=[0]*9,
        pid_maxs=[1]*9,
        targetcol='Label')
def evenoddf9d5():
    #10000 instances, synthetic
    # 9 features + 1 class
    # 2 classes -> [0, 1]; EvenOdd operation on the sum all dependent features (D); 0: even, 1: odd
    return Dataset(
        name='evenoddf9d5',
        directory=consts.DIR_CSV + '/evenodd/f9d5/',
        filename='n_10000_f_9_d_5.csv',
        cat_cols=[],
        num_cols=['D0', 'F1', 'F2', 'D3', 'D4', 'F5', 'D6', 'F7', 'D8'],
        pid_mins=[0]*9,
        pid_maxs=[1]*9,
        targetcol='Label')
def evenoddf3d3():
    #10000 instances, synthetic
    # 3 features + 1 class
    # 2 classes -> [0, 1]; EvenOdd operation on the sum all dependent features (D); 0: even, 1: odd
    return Dataset(
        name='evenoddf3d3',
        directory=consts.DIR_CSV + '/evenodd/f3d3/',
        filename='n_10000_f_3_d_3.csv',
        cat_cols=[],
        num_cols=['D0', 'D1', 'D2'],
        pid_mins=[0]*3,
        pid_maxs=[1]*3,
        targetcol='Label')
def evenoddf5d3():
    #10000 instances, synthetic
    # 5 features + 1 class
    # 2 classes -> [0, 1]; EvenOdd operation on the sum all dependent features (D); 0: even, 1: odd
    return Dataset(
        name='evenoddf5d3',
        directory=consts.DIR_CSV + '/evenodd/f5d3/',
        filename='n_10000_f_5_d_3.csv',
        cat_cols=[],
        num_cols=['D0', 'D1', 'F2', 'D3', 'F4'],
        pid_mins=[0]*5,
        pid_maxs=[1]*5,
        targetcol='Label')
def evenoddf7d3():
    #10000 instances, synthetic
    # 7 features + 1 class
    # 2 classes -> [0, 1]; EvenOdd operation on the sum all dependent features (D); 0: even, 1: odd
    return Dataset(
        name='evenoddf7d3',
        directory=consts.DIR_CSV + '/evenodd/f7d3/',
        filename='n_10000_f_7_d_3.csv',
        cat_cols=[],
        num_cols=['D0', 'F1', 'D2', 'F3', 'D4', 'F5', 'F6'],
        pid_mins=[0]*7,
        pid_maxs=[1]*7,
        targetcol='Label')

def generator(filename):
    #??? instances
    #?? features + 1 class
    #2 classes -> 0, 1
    directory = os.path.join(consts.DIR_GEN, filename) + '\\'
    f = open(os.path.join(directory, "params.txt"), "r")
    features = 0
    for line in f:
        if "features" in line:
            features = int(line.replace("features\t", ""))#'ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz\t\r\n'))
            break
    f.close()

    return Dataset(
        name='gen',
        directory=directory, #+ '/gen_or_25_3_True_50_20_10/',
        filename='raw_data.csv',
        cat_cols=list(map(str, range(features))),
        num_cols=[],
        pid_mins=[],
        pid_maxs=[],
        targetcol='label')
def get_data_set(name:str):
    """
    returns the dataset corresponding to its name
    returns None if no match found
    """
    if name == 'abalone': return abalone()
    elif name == 'adult': return adult()
    elif name == 'airlines': return airlines()
    elif name == 'cfpdss': return cfpdss()
    elif name == 'electricity': return electricity()
    elif name == 'forest': return forest()
    elif name == 'intrusion': return intrusion()
    elif name == 'magic': return magic()
    elif name == 'nursery': return nursery()
    elif name == 'occupancy': return occupancy()
    elif name == 'pendigits': return pendigits()
    elif name == 'sea': return sea()
    elif name == 'xor': return xor()
    elif name == 'xorf3d2': return xorf3d2()
    elif name == 'xorf5d3': return xorf5d3()
    elif name == 'xorf7d4': return xorf7d4()
    elif name == 'xorf9d5': return xorf9d5()
    elif name == 'evenoddf3d3': return evenoddf3d3()
    elif name == 'evenoddf5d3': return evenoddf5d3()
    elif name == 'evenoddf7d3': return evenoddf7d3()
    elif name == 'evenoddf9d2': return evenoddf9d2()
    elif name == 'evenoddf9d3': return evenoddf9d3()
    elif name == 'evenoddf9d4': return evenoddf9d4()
    elif name == 'evenoddf9d5': return evenoddf9d5()
    elif name[:3] == 'gen': return generator(name)
    else: raise ValueError(f'"{name}" is not part of used data sets')
      
def title(text:str):
    os.system("title " + text)

def get_slwin(window_size:int, forgetting_strategy=None):
    """
    returns a new sliding window
    """
    return SlidingWindow(window_size=window_size, forgetting_strategy=forgetting_strategy)

def get_flwin(window_size:int):
    """
    returns a new fixed length window
    """
    return FixedLengthWindow(window_size=window_size)
    
def get_ipf(budget:float, window_size:int):
    """
    returns a new incremental percentile filter budget manager
    """
    return IncrementalPercentileFilter(budget_threshold=budget, window_size=window_size)

def get_tcipf(budget:float, window_size:int):
    """
    returns a new trend corrected incremental percentile filter budget manager
    """
    return TrendCorrectedIncrementalPercentileFilter(budget_threshold=budget, window_size=window_size)

def get_sbm(budget:float):
    """
    returns a new simple budget manager
    """
    return SimpleBudgetManager(budget_threshold=budget)

def calc_ident_miss_chances(dataset:Dataset, miss_chance:float):
    miss_chances = {}
    for col in dataset.num_cols + dataset.cat_cols:
        miss_chances[col] = miss_chance
    return miss_chances

    
def _mp_do_all_task(dataset, aed_window_size, window_size, ipf_size, batch_size, ild_extra_rows, budgets, miss_chance, i):
    """
    called from do_tasks_all_afas_bms_fsss as subprocess
    thus freeing memory after completion as python refuses to free memory in free lists
    """
    sub_folder = 'prepared/' + str(i) + '/' + str(miss_chance)
    summary_str = dataset.summary_str('prepared', str(i), str(miss_chance))
    pre_folder_name=str(i)
    post_folder_name=str(miss_chance)
    """
    #preprocessing
    title(str(i) + "-th Preprocessing " + str(miss_chance) + " misses " + dataset.name)
    dataset.do_preprocessing(miss_chances=calc_ident_miss_chances(dataset, miss_chance),
                        shuffle=False,
                        batch_size=batch_size, ild_extra_rows=ild_extra_rows,
                        summary_str=summary_str,
                        sub_folder=sub_folder)
    #""
    #lower bound
    title(str(i) + "-th Lower Bound " + str(miss_chance) + " misses " + dataset.name)
    window = get_slwin(window_size)
    base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)#SGDPredictFix()#CalibratedClassifierCV(SGDClassifier(max_iter=100, tol=1e-3), cv=3)
    dataset.do_AFA_lower_bound(window=window, 
                            base_estimator=base_estimator,
                            summary_str=summary_str,
                            pre_folder_name=pre_folder_name,
                            post_folder_name=post_folder_name)
    #""
    #upper bound
    title(str(i) + "-th Upper Bound " + str(miss_chance) + " misses " + dataset.name)
    window = get_slwin(window_size)
    base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)
    dataset.do_AFA_upper_bound(window=window, 
                            base_estimator=base_estimator,
                            summary_str=summary_str,
                            pre_folder_name=pre_folder_name,
                            post_folder_name=post_folder_name)
    """                       
    for budget in budgets:
        #for skip_quality in [True, False]:
        print(dataset.name.upper() + ' ' + str(budget))
        
        #""
        #random AFA
        title(str(i) + "-th Random AFA + SBM " + str(miss_chance) + " misses " + str(budget) + " budget " + dataset.name)
        window = get_slwin(window_size)
        base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)
        budget_manager = get_sbm(budget)
        dataset.do_AFA_random(window=window,
                        base_estimator=base_estimator,
                        budget_manager=budget_manager,
                        summary_str=summary_str,
                        pre_folder_name=pre_folder_name,
                        post_folder_name=post_folder_name)
        #""
        #SAED
        #""
        title(str(i) + "-th SAED + IPF " + str(miss_chance) + " misses " + str(budget) + " budget " + dataset.name)
        window = get_slwin(window_size)
        base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)
        budget_manager = get_ipf(budget, ipf_size)
        dataset.do_AFA_SAED(window=window,
                        base_estimator=base_estimator,
                        budget_manager=budget_manager,
                        summary_str=summary_str,
                        pre_folder_name=pre_folder_name,
                        post_folder_name=post_folder_name)
                        #post_folder_name=post_folder_name + ' ' + str(skip_quality),
                        #debug=skip_quality)
        #""
        title(str(i) + "-th SAED + SBM " + str(miss_chance) + " misses " + str(budget) + " budget " + dataset.name)
        window = get_slwin(window_size)
        base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)
        budget_manager = get_sbm(budget)
        dataset.do_AFA_SAED(window=window,
                        base_estimator=base_estimator,
                        budget_manager=budget_manager,
                        summary_str=summary_str,
                        pre_folder_name=pre_folder_name,
                        post_folder_name=post_folder_name)        
        #""        
        #SIG
        #""
        title(str(i) + "-th SIG + IPF " + str(miss_chance) + " misses " + str(budget) + " budget " + dataset.name)
        window = get_slwin(window_size)
        base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)
        budget_manager = get_ipf(budget, ipf_size)
        dataset.do_AFA_SIG(window=window,
                        base_estimator=base_estimator,
                        budget_manager=budget_manager,
                        summary_str=summary_str,
                        pre_folder_name=pre_folder_name,
                        post_folder_name=post_folder_name)
        #""                
        title(str(i) + "-th SIG + SBM " + str(miss_chance) + " misses " + str(budget) + " budget " + dataset.name)
        window = get_slwin(window_size)
        base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)
        budget_manager = get_sbm(budget)
        dataset.do_AFA_SIG(window=window,
                        base_estimator=base_estimator,
                        budget_manager=budget_manager,
                        summary_str=summary_str,
                        pre_folder_name=pre_folder_name,
                        post_folder_name=post_folder_name)
                        
        #SSU
        #""
        title(str(i) + "-th SSU + IPF " + str(miss_chance) + " misses " + str(budget) + " budget " + dataset.name)
        window = get_slwin(window_size)
        base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)
        budget_manager = get_ipf(budget, ipf_size)
        dataset.do_AFA_SSU(window=window,
                        base_estimator=base_estimator,
                        budget_manager=budget_manager,
                        summary_str=summary_str,
                        pre_folder_name=pre_folder_name,
                        post_folder_name=post_folder_name)
        #""                
        title(str(i) + "-th SSU + SBM " + str(miss_chance) + " misses " + str(budget) + " budget " + dataset.name)
        window = get_slwin(window_size)
        base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)
        budget_manager = get_sbm(budget)
        dataset.do_AFA_SSU(window=window,
                        base_estimator=base_estimator,
                        budget_manager=budget_manager,
                        summary_str=summary_str,
                        pre_folder_name=pre_folder_name,
                        post_folder_name=post_folder_name)
        #""
        #MAED
        title(str(i) + "-th MAED + IPF " + str(miss_chance) + " misses " + str(budget) + " budget " + dataset.name)# + str(skip_quality))
        window = get_slwin(window_size)
        base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)
        budget_manager = get_ipf(budget, ipf_size)
        dataset.do_AFA_MAED(window=window,
                        base_estimator=base_estimator,
                        budget_manager=budget_manager,
                        aed_window_size=aed_window_size,
                        summary_str=summary_str,
                        pre_folder_name=pre_folder_name,
                        post_folder_name=post_folder_name)
                        #post_folder_name=post_folder_name + ' ' + str(skip_quality),
                        #debug=skip_quality)
        #""
        title(str(i) + "-th MAED + SBM " + str(miss_chance) + " misses " + str(budget) + " budget " + dataset.name)
        window = get_slwin(window_size)
        base_estimator = SGDClassifier(loss='log', max_iter=100, tol=1e-3)#DecisionTreeClassifier(max_depth=4)
        budget_manager = get_sbm(budget)
        dataset.do_AFA_MAED(window=window,
                        base_estimator=base_estimator,
                        budget_manager=budget_manager,
                        aed_window_size=aed_window_size,
                        summary_str=summary_str,
                        pre_folder_name=pre_folder_name,
                        post_folder_name=post_folder_name)
        #""
        
def do_tasks_all_afas_bms_fsss(datasets:list,
                       aed_window_size:int = 50, window_size:int = 10,
                       ipf_size:int = 50, batch_size:int = 50, ild_extra_rows:int = 50,
                       budgets:list = [0.5], miss_chances:list = [0.5], iterations = [0]):
    """
    Executes a list of datasets with the given tasks with all given Missignesses and Budgets
    combinations over the specified dataset iterations times
    """
    for dataset in datasets:
        for i in iterations:#range(iterations):
            #pool = mp.Pool(processes=4)
            #pool_args = []
            for miss_chance in miss_chances:
                #pool_args.append([dataset, aed_window_size, window_size,
                #    ipf_size, batch_size, ild_extra_rows, budgets, miss_chance, i])
            #pool.starmap(_mp_do_all_task, pool_args)
                p = mp.Process(target=_mp_do_all_task, 
                    args=(dataset, aed_window_size, window_size,
                    ipf_size, batch_size, ild_extra_rows, budgets, miss_chance, i))
                p.start()
                p.join()

def get_classifier(name:str):
    if name == 'SGD': return SGDClassifier(loss='log', max_iter=100, tol=1e-3)
    elif name == 'DTC': return DecisionTreeClassifier(max_depth=4)
    else: raise ValueError('Undefined classifier string')

def do_tasks(params):#:dict):
    """
    Starts a new process for each param
    :param params: Dictionary with parameters
    """
    # fail cases
    classifier = 'SGD' #'DTC'
    
    window_size = params.w#params['w']
    datasets = params.d#params['d']
    iterations = params.i#params['i']
    set_sizes = params.s#params['s']
    acquisition_costs = params.c#params['c']
    miss_chances = params.m#params['m']
    budgets = params.b#params['b']
    tasks = params.t#params['t']
    preparation = params.p#params['p']
    ipf_size = 100
    aed_size = 50
    if not window_size: raise ValueError("Window size has to be specified")
    if len(datasets) == 0: raise ValueError("At least one dataset has to be specified")
    if len(iterations) == 0: raise ValueError("An iteration value has to be specified")
    
    # prep, lower and upper
    prepare = preparation is not None
    if prepare: 
        shuffle, batch_size, ild_size = preparation
        shuffle = shuffle == 'True'
        batch_size = int(batch_size)
        ild_size = int(ild_size)
    lower = 'lower' in tasks
    if lower: tasks.remove('lower')
    fpilow = 'fpilow' in tasks
    if fpilow: tasks.remove('fpilow')
    upper = 'upper' in tasks
    if upper: tasks.remove('upper')
    
    post_folder_name=""
    
    for dataset in datasets:
        for iteration in iterations:
            for miss_chance in miss_chances:
                sub_folder = 'prepared/' + str(iteration) + '/' + str(miss_chance)
                summary_str = dataset.summary_str('prepared', str(iteration), str(miss_chance))
                pre_folder_name = f"{iteration}/{miss_chance}"
                #post_folder_name = str(miss_chance)
                
                #preprocessing
                if prepare:
                    text = "{} {}-th preprocessing {} misses".format(dataset, iteration, miss_chance)
                    title(text)
                    #log.write("{}: {}\n".format(datetime.now(), text))
                    p = mp.Process(target=dataset.do_preprocessing, 
                        args=(calc_ident_miss_chances(dataset, miss_chance), 
                            batch_size, ild_size, sub_folder, summary_str, shuffle))
                    p.start()
                    p.join()
                    
                #upper
                if upper and miss_chance == miss_chances[0]:
                    text = "{} upper bound".format(dataset)
                    title(text)
                    #log.write("{}: {}\n".format(datetime.now(), text))
                    p = mp.Process(target=dataset.do_AFA_upper_bound,
                       args=(get_slwin(window_size), get_classifier(classifier),
                           summary_str, f"{iteration}", post_folder_name))
                    p.start()
                    p.join()
                
                #feature pair imputer low
                if fpilow:
                    text = "{} {}-th {} misses fpilow bound".format(dataset, iteration, miss_chance)
                    title(text)
                    #log.write("{}: {}\n".format(datetime.now(), text))
                    p = mp.Process(target=dataset.do_AFA_pairreg_bound,
                        args=(get_slwin(window_size), get_classifier(classifier),
                            summary_str, f"{pre_folder_name}/equal", post_folder_name))
                    p.start()
                    p.join()

                #lower
                if lower:
                    text = "{} {}-th {} misses lower bound".format(dataset, iteration, miss_chance)
                    title(text)
                    #log.write("{}: {}\n".format(datetime.now(), text))
                    p = mp.Process(target=dataset.do_AFA_lower_bound,
                        args=(get_slwin(window_size), get_classifier(classifier),
                            summary_str, pre_folder_name, post_folder_name))
                    p.start()
                    p.join()
                
                for cost_strategy in acquisition_costs:
                    pre_folder_name = f"{iteration}/{miss_chance}/{cost_strategy}"
                    costs = dataset.generate_feature_costs(strategy=cost_strategy)
                    for thr, budget_option in budgets:
                        for set_size in set_sizes:
                            threshold = get_threshold(thr, budget_option, costs, set_size, batch_size=50)
                            for task in tasks:
                                #task
                                text = "{} {}-th {} misses {} costs {} budget {} max features {}".format(dataset, iteration, miss_chance, cost_strategy, budget_option, set_size, task)
                                title(text)
                                afa_s, bm_s, fss_s = task.split('+')
                                
                                bm = None
                                if bm_s == consts.IPF: bm = IncrementalPercentileFilter(budget_threshold=threshold, window_size=ipf_size)
                                elif bm_s == consts.SBM: bm = SimpleBudgetManager(budget_threshold=threshold)
                                elif bm_s == consts.NBM: bm = NoBudgetManager()
                                elif bm_s == consts.TCIPF: bm = TrendCorrectedIncrementalPercentileFilter(budget_threshold=threshold, window_size=ipf_size)
                                
                                fss = None
                                if fss_s == consts.KBSMRFSS: fss = KBestSMRFSS(k=set_size)
                                elif fss_s == consts.KPBSMRFSS: fss = KPickyBestSMRFSS(k=set_size, rank=set_size)
                                elif fss_s == consts.KRSMRFSS: fss = KRandomSMRFSS(k=set_size)
                                elif fss_s == consts.KQSMRFSS: fss = KQualitySMRFSS(k=set_size)
                                elif fss_s == consts.KQGSMRFSS: fss = KQGainSMRFSS(k=set_size)
                                elif fss_s == consts.KBAQGSMRFSS: fss = KBudgetAwareQGainSMRFSS(k=set_size)
                                elif fss_s == consts.KBIAMSMRFSS: fss = KBestImputerAlteredMeritSMRFSS(k=set_size, imputer=None)
                                elif fss_s == consts.KBITSMRFSS: fss = KBestImputerThresholdSMRFSS(k=set_size, imputer=None, nrmse_threshold=0.2)

                                p = None
                                # Regular Methods
                                if afa_s == consts.RA: p = mp.Process(target=dataset.do_AFA_random,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED: p = mp.Process(target=dataset.do_AFA_SAED,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWIG: p = mp.Process(target=dataset.do_AFA_SIG,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWSU: p = mp.Process(target=dataset.do_AFA_SSU,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.MWAED: p = mp.Process(target=dataset.do_AFA_MAED,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, aed_size, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.PCFI: p = mp.Process(target=dataset.do_AFA_PCFI,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.PCFI_DTC: p = mp.Process(target=dataset.do_AFA_PCFI_DTC,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))

                                # Feature Pair Imputer Regular Methods
                                elif afa_s == consts.RA_FPI: p = mp.Process(target=dataset.do_AFA_RA_FPI,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_FPI: p = mp.Process(target=dataset.do_AFA_SAED_FPI,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                
                                # Dynamic Budget Threshold Methods
                                elif afa_s == consts.SWAED_NDB: p = mp.Process(target=dataset.do_AFA_SAED_NDB,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_IMAX: p = mp.Process(target=dataset.do_AFA_SAED_IMAX,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_IEXP: p = mp.Process(target=dataset.do_AFA_SAED_IEXP,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))

                                # Altered Methods
                                elif afa_s == consts.SWAED_SSBQ: p = mp.Process(target=dataset.do_AFA_SAED_SSBQ,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_SSBQ_FPI: p = mp.Process(target=dataset.do_AFA_SAED_SSBQ_FPI,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))

                                # Feature Pair Imputer Information Integrated Methods
                                elif afa_s == consts.SWAED_IMPQ: p = mp.Process(target=dataset.do_AFA_SAED_IMPQ,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_IMPQ2: p = mp.Process(target=dataset.do_AFA_SAED_IMPQ2,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                    
                                # Feature Pair Imputer Threshold Skip Methods
                                elif afa_s == consts.SWAED_FPITS_10: p =mp.Process(target=dataset.do_AFA_SAED_FPITS,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, 0.1, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_FPITS_20: p =mp.Process(target=dataset.do_AFA_SAED_FPITS,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, 0.2, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_FPITS_30: p =mp.Process(target=dataset.do_AFA_SAED_FPITS,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, 0.3, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_FPITS_40: p =mp.Process(target=dataset.do_AFA_SAED_FPITS,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, 0.4, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_FPITS_50: p =mp.Process(target=dataset.do_AFA_SAED_FPITS,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, 0.5, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_FPITS_60: p =mp.Process(target=dataset.do_AFA_SAED_FPITS,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, 0.6, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_FPITS_70: p =mp.Process(target=dataset.do_AFA_SAED_FPITS,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, 0.7, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_FPITS_80: p =mp.Process(target=dataset.do_AFA_SAED_FPITS,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, 0.8, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_FPITS_90: p =mp.Process(target=dataset.do_AFA_SAED_FPITS,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, 0.9, summary_str, pre_folder_name, post_folder_name, params.overwrite))

                                # Feature Pair Imputer Threshold Skip Log Methods
                                elif afa_s == consts.SWAED_FPITS_LOG_10: p =mp.Process(target=dataset.do_AFA_SAED_FPITS_LOG,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, 0.1, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_FPITS_LOG_20: p =mp.Process(target=dataset.do_AFA_SAED_FPITS_LOG,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, 0.2, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_FPITS_LOG_30: p =mp.Process(target=dataset.do_AFA_SAED_FPITS_LOG,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, 0.3, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_FPITS_LOG_40: p =mp.Process(target=dataset.do_AFA_SAED_FPITS_LOG,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, 0.4, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_FPITS_LOG_50: p =mp.Process(target=dataset.do_AFA_SAED_FPITS_LOG,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, 0.5, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_FPITS_LOG_60: p =mp.Process(target=dataset.do_AFA_SAED_FPITS_LOG,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, 0.6, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_FPITS_LOG_70: p =mp.Process(target=dataset.do_AFA_SAED_FPITS_LOG,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, 0.7, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_FPITS_LOG_80: p =mp.Process(target=dataset.do_AFA_SAED_FPITS_LOG,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, 0.8, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_FPITS_LOG_90: p =mp.Process(target=dataset.do_AFA_SAED_FPITS_LOG,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, 0.9, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                        
                                # elif afa_s == consts.SWAEDFCC: p = mp.Process(target=dataset.do_AFA_SAEDFCC,
                                    # args=(get_slwin(window_size), get_classifier(classifier),
                                        # bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))


                                elif afa_s == consts.SWAED_QT: p = mp.Process(target=dataset.do_AFA_SAED_QT,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_AQ: p = mp.Process(target=dataset.do_AFA_SAED_AQ,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_Q: p = mp.Process(target=dataset.do_AFA_SAED_Q,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_II: p = mp.Process(target=dataset.do_AFA_SAED_II,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAEDFCC_II: p = mp.Process(target=dataset.do_AFA_SAEDFCC_II,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_RPRI: p = mp.Process(target=dataset.do_AFA_SAED_RPRI,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAEDFCC_RPRI: p = mp.Process(target=dataset.do_AFA_SAEDFCC_RPRI,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_SPRI: p = mp.Process(target=dataset.do_AFA_SAED_SPRI,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAEDFCC_SPRI: p = mp.Process(target=dataset.do_AFA_SAEDFCC_SPRI,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_BPRI: p = mp.Process(target=dataset.do_AFA_SAED_BPRI,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))

                                
                                elif afa_s == consts.SWAED_TRPRI: p = mp.Process(target=dataset.do_AFA_SAED_TRPRI,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_TRPRI_0: p = mp.Process(target=dataset.do_AFA_SAED_TRPRI_0,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_TRPRI_25: p = mp.Process(target=dataset.do_AFA_SAED_TRPRI_25,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_TRPRI_50: p = mp.Process(target=dataset.do_AFA_SAED_TRPRI_50,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_TRPRI_75: p = mp.Process(target=dataset.do_AFA_SAED_TRPRI_75,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_TRPRI_100: p = mp.Process(target=dataset.do_AFA_SAED_TRPRI_100,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_TRPRI_200: p = mp.Process(target=dataset.do_AFA_SAED_TRPRI_200,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))

                                elif afa_s == consts.SWAED_I: p = mp.Process(target=dataset.do_AFA_SAED_I,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_D: p = mp.Process(target=dataset.do_AFA_SAED_D,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_C: p = mp.Process(target=dataset.do_AFA_SAED_C,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_P: p = mp.Process(target=dataset.do_AFA_SAED_P,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_E: p = mp.Process(target=dataset.do_AFA_SAED_E,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_H: p = mp.Process(target=dataset.do_AFA_SAED_H,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWAED_H2: p = mp.Process(target=dataset.do_AFA_SAED_H2,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWIG_H2: p = mp.Process(target=dataset.do_AFA_SIG_H2,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                elif afa_s == consts.SWSU_H2: p = mp.Process(target=dataset.do_AFA_SSU_H2,
                                    args=(get_slwin(window_size), get_classifier(classifier),
                                        bm, budget_option, costs, set_size, fss, summary_str, pre_folder_name, post_folder_name, params.overwrite))
                                
                                
                                if p is not None:
                                    p.start()
                                    p.join()
    #log.close()

def get_budget_option(budget_option:list):
    threshold, once, batch, inst, acq = budget_option
    bo = []
    if float(once) > 0: bo.append((('once'), float(once)))
    if float(batch) > 0: bo.append((('batch'), float(batch)))
    if float(inst) > 0: bo.append((('inst'), float(inst)))
    if float(acq) > 0: bo.append((('acq'), float(acq)))
    if not any(bo): bo.append((('once'), 0.0))
    return threshold, bo

def get_threshold(threshold:str, budget_option:list, costs:dict, set_size:int, batch_size:int, exp_acqs_per_instance=1.0):
    if is_number(threshold):
        return min(max(0.0, float(threshold)), 1.0)
    else:
        budget_per_instance = 0
        for bo, val in budget_option:
            if bo == 'once': budget_per_instance += val / batch_size
            elif bo == 'batch': budget_per_instance += val / batch_size
            elif bo == 'inst': budget_per_instance += val
            elif bo == 'acq': budget_per_instance += val * exp_acqs_per_instance
        
        if threshold == 'imax':
            set_size = min(set_size, len(costs))
            cost_per_instance = sum(sorted(costs.values(), reverse=True)[:set_size])
        elif threshold == 'half': 
            set_size = max(1, min(set_size, len(costs)) // 2)
            cost_per_instance = sum(sorted(costs.values(), reverse=True)[:set_size])
        elif threshold == 'mean':
            set_size = max(1, min(set_size, len(costs)) // 2)
            cost_per_instance = sum(costs.values()) / len(costs) * set_size
        else: raise ValueError(f"Undefined auto threshold method: '{threshold}'")
        
    return min(max(0.0, budget_per_instance / cost_per_instance), 1.0)    

def remove_chars(text:str):
    return text.translate({ord(i): None for i in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,:;()[]{}+*-#~_ '})

def abbreviate(text:str, delim:str = '_'):
    retval = ''
    for word in text.split(delim):
        retval += word[0].upper()
    return retval

def valid_afa_bm_fs(text:str):
    """
    Returns whether the AFA, BM and FS are contained in constants
    """
    if text == 'upper': return True
    if text == 'lower': return True
    afa, bm, fss = text.split('+')
    return afa.upper() in consts.AFAS and bm.upper() in consts.BMS and fss.upper() in consts.FSS
    #return afa in list(map(abbreviate, consts.AFAS)) and bm in list(map(abbreviate, consts.BMS)) and fss in list(map(abbreviate, consts.FSS))

def is_number(s:str):
    try:
        float(s)
        return True
    except ValueError:
        return False

def main():
    p_arg = argparse.ArgumentParser(description='Performs test runs based on parameters')

    # p_arg.add_argument('-l', required=False, nargs=1, type=str,
    #     help='A file containing a list of explicit jobs to be performed')
    # p_arg.add_argument('-j', required=False, nargs=1, type=int,
    #     help='The number of jobs dedicated to the list of jobs')

    p_arg.add_argument('-d', required=True, nargs='*', 
        help='the data sets for the tasks')
    p_arg.add_argument('-i', required=True, nargs='*', type=int, 
        help='the iterations')
    p_arg.add_argument('-m', required=True, nargs='*', type=float, 
        help='the missingness options')

    p_arg.add_argument('-s', required=False, nargs='*', type=int, 
        help='the acquisition set size')
    p_arg.add_argument('-w', required=False, type=int, 
        help='the size (batches) of the sliding window')
    p_arg.add_argument('-c', required=False, nargs='*', choices=consts.COSTDISTS, 
        help='the feature cost distributions')

    p_arg.add_argument('-t', required=False, nargs='*', 
        help='the special task name or alternatively the active feature acquisition moniker+budget manager moniker+feature set selection moniker')
    p_arg.add_argument('-b', required=False, nargs='*', 
        help='the budget options in the order of initial threshold, budget gained once, budget gained per batch, budget gained per instance, budget gained per acquisition')
    p_arg.add_argument('-p', required=False, nargs=3, 
        help='prepares data when specified in order of shuffle, batch size and initially labeled instances')
    
    p_arg.add_argument('--noconfirm', required=False, default=False, action='store_true',
        help='skips the confirm prompt')
    p_arg.add_argument('--overwrite', required=False, default=False, action='store_true',
        help='skips checking and raising an error if the test result summary file already exists')

    args = p_arg.parse_args()

    #dataset
    args.d = [get_data_set(d) for d in args.d]

    #fix budget
    if len(args.b) % 5 != 0:
        raise ValueError(f"budget inputs have wrong format. Provide 5 values per budget option. Given string: {args.b}")
    args.b = [get_budget_option(args.b[i*5:i*5+5]) for i in range(len(args.b)//5)]

    if not args.noconfirm:
        print(f"data sets        : {args.d}")
        print(f"iterations       : {args.i}")
        print(f"miss_chances     : {args.m}")
        if args.p is not None: print(f"preparation      : shuffle {args.p[0]}, batch size {args.p[1]}, ild extra instances {args.p[2]}")
        print(f"acquisition_costs: {args.c}")
        print(f"tasks            : {args.t}")
        print(f"acq_set_sizes    : {args.s}")
        print("budgets          : {}".format([b[1] for b in args.b]))
        print(f"window size      : {args.w}")
        print("")
        #do_tasks(params)
        c = input("To continue enter \'y\'\n\n")
        if c == 'y': do_tasks(args)
    else:
        do_tasks(args)

if __name__ == '__main__':
    """
    executes runs given hardcoded parameters
    contains dataset informations and Dataset class
    
    the following space separated args are defined:
    -d                          - checks following args for defined dataset names
    after -d [dataset]          - adds dataset to the tasks
    -t                          - checks following args for defined afa+bm+fs combinations
    after -t [task or afa+bm+fss]
                                - the special task name or alternatively
                                - the active feature acquisition method moniker
                                - the budget manager moniker
                                - the acquisition feature set selection method moniker
    -s                          - checks following args for maximum feature set sizes to be acquired
    after -s                    -
    after -i [iterations]       - adds specific iterations to the process, accepts val..val as ranges
    -b                          - checks following args for budget values
    after -b [threshold] [gain once] [gain batch] [gain inst] [gain acq]
                                - (initial) threshold for all budget managers it applies to
                                    set to float value to manually set initial threshold
                                    set to "imax" to automatically choose the low bar threshold
                                    set to 'half' to automatically choose the low bar threshold for half the set size
                                    set to 'mean' to automatically choose setsize / 2 * meancost as threshold
                                - budget gained once before execution of task
                                - budget gained before each batch to be processed
                                - budget gained before each instance to be processed
                                - budget gained before each acquisition decision
    -c                          - sets the acquisition costs methods for the features
    after -c                    -
    -m                          - checks following args for missingness values
    after -m                    -
    -p                          - enables the preprocessing step for all data sets
    after -p [shuffle] [batch size] [additional ild instances]
                                - whether to randomize the order of the data set 
                                    if shuffled, adds one representative instance for each label
                                - how large the batch size is
                                - the amount of additional instances in the initial data
    -l                          - checks the following args for the name of a log file
    after -l [log filename]     - the name the log writes into
    -w                          - checks the following args for the window size
    after -w [window size]      - the size (in batches) of the sliding window
    
    use example:
        -d nursery -p True 50 100 -t lower upper RA+SBM SWAED+SBM SWAED+IPF -i 0..3 9 -m 0.25 0.5 0.75 -b 0.25 0.5 0.75 1.0 -w 10 -l tasklogfile.log
    """
    main()

def old_methods():
    datasets = []
    #budgets = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    set_sizes = []
    budgets = []
    #budgets = [1.0]
    #miss_chances = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    miss_chances = []
    #miss_chances = [0.75]
    acquisition_costs = []
    iterations = []
    tasks = []
    preparation = []
    
    params = {'d':datasets, 't':tasks, 'i':iterations, 's':set_sizes, 'b':budgets, 'm':miss_chances, 'c':acquisition_costs, 'p':preparation, 'w':None}#, 'l':"tasks.log"}
    
    read_mode = ''
    for arg in sys.argv[1:]:
        if arg[0] == '-' and arg[1] in params: read_mode = arg[1]
        else:
            #if read_mode == 'l': params['l'] = arg
            if read_mode == 'd':
                dataset = get_data_set(arg)
                if dataset: datasets.append(dataset)
            elif read_mode == 't':
                if valid_afa_bm_fs(arg): tasks.append(arg)
            elif read_mode == 'p':
                if arg in ['T', 't', 'True', 'true']: preparation.append(True)
                elif arg in ['F', 'f', 'False', 'false']: preparation.append(False)
                else: preparation.append(int(remove_chars(arg)))
            elif read_mode == 'w': params[read_mode] = int(remove_chars(arg))                
            elif read_mode == 'c': acquisition_costs.append(arg)
            elif read_mode == 's': params[read_mode].append(int(arg))
            elif read_mode == 'b':
                if is_number(arg): params[read_mode].append(float(arg))
                else: params[read_mode].append(arg)
            else:
                if '..' in arg:
                    args = arg.split('..')
                    params[read_mode] += range(int(args[0]), int(args[1]))
                else: params[read_mode].append(float(remove_chars(arg)))
    
    #fix budget
    if len(budgets) % 5 != 0:
        raise ValueError(f"budget inputs have wrong format. Provide 5 values per budget option. Given string: {budgets}")
    budgets = [get_budget_option(budgets[i*5:i*5+5]) for i in range(int(len(budgets)/5))]
    params['b'] = budgets
    
    
    print("data sets        : {}".format(datasets))
    print("iterations       : {}".format(iterations))
    print("miss_chances     : {}".format(miss_chances))
    if len(preparation) != 0: print("preparation      : shuffle {}, batch size {}, ild extra instances {}".format(preparation[0], preparation[1], preparation[2]))
    print("acquisition_costs: {}".format(acquisition_costs))
    print("tasks            : {}".format(tasks))
    print("acq_set_sizes    : {}".format(set_sizes))
    print("budgets          : {}".format([b[1] for b in budgets]))
    print("window size      : {}".format(params['w']))
    #print("log file         : {}".format(params['l']))
    print("")
    #do_tasks(params)
    c = input("To continue enter \'y\'\n\n")
    if c == 'y': do_tasks(params)
    # else: do_tasks_all_afas_bms_fsss(datasets=[abalone()],
    #                    aed_window_size=50, window_size=10,
    #                    ipf_size=50, batch_size=50, ild_extra_rows=50,
    #                    budgets=[0.125], miss_chances=[0.125], iterations=[10])