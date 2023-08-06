from abc import ABC, abstractmethod

import numpy as np
import math
import pandas as pd
from datetime import datetime

from osm.data_streams.abstract_base_class import AbstractBaseClass
from osm.data_streams.budget_manager.abstract_budget_manager import AbstractBudgetManager
from osm.data_streams.budget_manager.no_budget_manager import NoBudgetManager
import osm.data_streams.constants as const

class AbstractActiveFeatureAcquisitionStrategy(AbstractBaseClass):
    def __init__(self, target_col_name, budget_manager, acq_set_size, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Abstract class to implement an active feature acquisition strategy
        :param budget_manager: An instance of type AbstractBudgetManager
        :param acq_set_size: The maximum set size that the method attempts to acquire per instance
        :param acquisition_costs: A dictionary with the names of columns as key and acquisition_cost as value
        if a field is left empty, handle like set to 1
        :param target_col_name: The name of the target column
        :param budget_option: specify a list of tuples consisting of points at which budget is distributed and budget gain function
        accepted strings as position: 'once', 'batch', 'inst', 'acq'
        :param debug: If True prints debug messages to console
        """
        super().__init__()

        #if not isinstance(budget, float):
        #    raise ValueError("The budget should be a float between [0,1]")

        #if budget < 0 or budget > 1:
        #    raise ValueError("The budget should be a float between [0,1]")

        if budget_manager is None or not isinstance(budget_manager, AbstractBudgetManager):
            raise ValueError("The budget_manager must be of type AbstractBudgetManager")

        self.queried = {}
        self.miss_counts = {}
        self.instances_processed = 0
        self.target_col_name = target_col_name
        self.acq_set_size = acq_set_size
        self.acquisition_costs = acquisition_costs
        self.budget_manager = budget_manager
        self._read_budget_options(budget_option)
        self.debug = debug

    def _initialize_queried(self, columns):
        """
        Call to initialize self.queried for use as statistic
        :param columns: The specified columns as keys for self.queried
        """
        for col in columns:
            self.queried[col] = 0
            self.miss_counts[col] = 0

    def _read_budget_options(self, budget_option):
        """
        reads budget_option and translates it into the budget gain options
        """
        self._budget_once = False
        self._budget_batch = False
        self._budget_inst = False
        self._budget_acq = False
        for btime, bgain in budget_option:
            if btime == 'once':
                self._budget_once = True
                self._bgain_once = bgain
            elif btime == 'batch':
                self._budget_batch = True
                self._bgain_batch = bgain
            elif btime == 'inst':
                self._budget_inst = True
                self._bgain_inst = bgain
            elif btime == 'acq':
                self._budget_acq = True
                self._bgain_acq = bgain

    def _isCategorical(self, data, col):
        """
        returns true if column col of data is categorical data
        """
        return data[col].dtype.kind not in "iufcb"
        
    def _isNumerical(self, data, col):
        """
        returns true if column col of data is numerical data
        """
        return data[col].dtype.kind in "iufcb"

    def get_budget_string(self):
        if self._budget_once or self._budget_batch or self._budget_inst or self._budget_acq:
            return f"{'o'+str(self._bgain_once) if self._budget_once else ''}{'b'+str(self._bgain_batch) if self._budget_batch else ''}{'i'+str(self._bgain_inst) if self._budget_inst else ''}{'a'+str(self._bgain_acq) if self._budget_acq else ''}"
        else:
            return str(self.budget_manager.budget_threshold)

    def get_columns_with_nan(self, data, columns=None):
        """
        returns a list of column names that contain missing data
        :param data: the data to check for missing entries
        :param columns: specify which columns to check
        """
        if columns is None:
            return data.columns[data.isna().any()].tolist()
        else:
            return data.columns[data[columns].isna().any()].tolist()

    def isnan(self, value):
        """
        checks whether a value is nan
        """
        if not isinstance(value, float):
            return False
        return math.isnan(value)
			
    def has_nan_value(self, data, columns=None):
        """
        returns true if data has nan values
        :param data: the data to check for missing entries
        :param columns: specify which columns to check
        """
        if columns is None:
            return data.isna().any().any()
        else:
            return data[columns].isna().any().any()

    def label_known(self, inst):
        """
        returns whether the label of inst is known
        """
        return self.target_col_name in inst and not math.isnan(inst[self.target_col_name])

    def get_feature(self, inst, column):
        """
        returns the value for the requested feature
        currently just returns a column with same name + "_org"
        TODO: export get_feature as separate module or make choice to use separate data
        """
        return inst[column + '_org']

    def get_dynamic_threshold(self, i_exp, penalty_step_size=1/32, penalty_gain=1):
        """
        returns a new dynamic threshold based on
        the expected cost of total feature set acquisition,
        the current budget usage of the budget manager,
        and a penalty factor
        :param i_exp: the expected cost of total feature set acquisition
        :param penalty_step_size: increases denominator by the number of steps over the budget usage ratio to counteract biases in expected feature set acquisitions
        :param penalty_gain: increases denominator by this amount per the number of steps over the budget usage
        """
        if i_exp == 0: return self.budget_manager.budget_threshold
        
        exp_threshold = self._bgain_inst / i_exp
        used_budget_ratio = self.budget_manager.used_budget()
        new_threshold = exp_threshold / used_budget_ratio
        over_budget_rate = 1 + (math.floor((used_budget_ratio - 1) / penalty_step_size) + 1) * penalty_gain
        # e.g. [1, 1+p) -> 1/2, [1+p, 1+2p) -> 1/3, ...
        if used_budget_ratio > 1: new_threshold /= over_budget_rate
        return(min(new_threshold, 1))

    @abstractmethod
    def get_data(self, data):
        """
        gets the data with additionally requested features
        """
        pass
        
    def get_stats(self, index=0):
        pd_index = pd.MultiIndex.from_product([[const.missing_features, const.active_feature_acquisition_queries], [x for x in self.queried.keys()]])
        stats = pd.DataFrame(index=pd_index).T.copy()

        for col, count in self.miss_counts.items():
            stats.loc[index, (const.missing_features, col)] = count
        for col, count in self.queried.items():
            stats.loc[index, (const.active_feature_acquisition_queries, col)] = count

        return stats

        #if self.debug:
            #print(str.format("{0}: Index: {1}\tQueried: {2}\tAnswered: {3}\tCost: {4}",
                  #str(datetime.now()),
                  #index,
                  #self.oracle.get_total_queried(),
                  #self.oracle.get_total_answered(),
                  #self.oracle.get_cost()))