from abc import ABC, abstractmethod
from multiprocessing.sharedctypes import Value
import pandas as pd
from pandas.api.types import is_numeric_dtype
import math
import numpy as np
import operator
import copy
import osm.data_streams.constants as const

from osm.data_streams.windows.abstract_window import AbstractWindow

from osm.data_streams.active_feature_acquisition.supervised_merit_ranking.supervised_merit_ranking import AbstractSMR#, AbstractSingleWindowSMR, AbstractMultiWindowSMR

class AbstractAED(AbstractSMR):
    #TODO: get function to tell if label known
    #TODO: get the acquisition module
    #TODO: normalization with only 1 value: behavior
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size,  dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Active Feature Acquisition strategy that ranks features accoring to average euclidean distance
        Requires the exact categories and possible category values beforehand
        Abstract basis for other different counting mechanisms AED algorithms might implement
        :param window: the framework window
        :param feature_selection: the method by which the feature set for acquisition is selected
        :param dynamic_budget_threshold: whether the budget threshold should be dynamically adjusted once per batch
        :param categories: provide categories for all categorical features as dict 
        empty categorical names are interpreted as numerical columns
        alternatively if left None make initally labeled data in window represent 
        all categorical features at least once 
        """
        super().__init__(target_col_name=target_col_name,
                            window=window,
                            budget_manager=budget_manager,
                            acq_set_size=acq_set_size,
                            acquisition_costs=acquisition_costs,
                            feature_selection=feature_selection,
                            dynamic_budget_threshold=dynamic_budget_threshold,
                            categories=categories,
                            budget_option=budget_option,
                            debug=debug)
        self.initialized = False

    def _initialize(self, data):
        """
        determines labels and numerical / categorical columns on first run and creates dicts for calculations
        execute once before starting process to initialize all dicts and labels
        Sets self.initialized to true
        :param data: the data all columns and categories will be based on
        """
        super()._initialize(data)

        #init normalize dicts
        self.num_max = {}
        self.num_min = {}

        #init feature dicts
        self.num_counts = {}
        self.num_sums = {}
        self.cat_counts = {}

        #set cat_col dicts
        for col in self.cat_cols:
            self.cat_counts[col] = {}
            for label in self.labels:
                self.cat_counts[col][label] = {}

        #set num_col dicts
        for col in self.num_cols:
            self.num_counts[col] = {}
            self.num_sums[col] = {}

    def _normalize(self, min_val, max_val, val, count=1):
        """
        unity-based normalization for a single value
        :param count: for calculating normalized sums 
        """
        if (min_val >= max_val): return 0.5
        return (val - (min_val * count)) / (max_val - min_val)
    
    def _get_rank_values(self):
        """
        returns a dict containing all features and their average euclidean distances
        """

        distances = {}

        #numerical features
        for col in self.num_cols:
            means = {}
            for label in self.labels:
                norm_sum = self._normalize(val=self.num_sums[col][label],
                                            min_val=self.num_min[col],
                                            max_val=self.num_max[col],
                                            count=self.num_counts[col][label])
                means[label] = norm_sum / self.num_counts[col][label]
            
                #handle complete nan classes by removing them
                if self.isnan(means[label]):
                    means.pop(label)

            #TODO: possibly more elegant solution? itertools?
            means2 = copy.deepcopy(means)
            squared_sums = 0
            for dkey, dvalue in means.items():
                means2.pop(dkey)
                for dkey2, dvalue2 in means2.items():
                    squared_sums += (dvalue - dvalue2) ** 2
            
            distances[col] = math.sqrt(squared_sums)

        #categorical features
        for col in self.cat_cols:
            #get count of features per label
            L = {}
            for label in self.labels:
                L[label] = sum(self.cat_counts[col][label].values())

                #handle no occurences of features for class by popping label from dict
                if L[label] == 0:
                    L.pop(label)
            
            L2 = copy.deepcopy(L)
            label_sums = 0
            for l1 in L.keys():
                L2.pop(l1)
                for l2 in L2.keys():
                    value_sums = 0
                    for v in self.categories[col]:
                        value_sums += math.sqrt(((self.cat_counts[col][l1][v] / L[l1]) - (self.cat_counts[col][l2][v] / L[l2])) ** 2)

                    label_sums += value_sums / len(self.categories[col])
            
            distances[col] = label_sums
        
        return distances
        
class MultiWindowAED(AbstractAED):
    def __init__(self, ild, window_size, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        """
        super().__init__(target_col_name=target_col_name,
                             window=ild,
                             budget_manager=budget_manager,
                             feature_selection=feature_selection,
                             dynamic_budget_threshold=dynamic_budget_threshold,
                             categories=categories,
                             acq_set_size=acq_set_size,
                             acquisition_costs=acquisition_costs,
                             budget_option=budget_option,
                             debug=debug)
                             
        self.window_size = window_size
    
    def _initialize(self, data):
        """
        resets sums and counts of data to 0 and recounts and resums all
        occurences for the given data
        adds all data into windows
        data is normalized after receiving it
        """
        super()._initialize(data)
        
        #set windows
        self.windows = {}
        self.windows_counts = {}
        for col in self.cat_cols + self.num_cols:
            self.windows[col] = {}
            self.windows_counts[col] = {}
        
        #first entry - set up num_min and num_max
        first = data.iloc[0]
        for col in self.num_cols:
            self.num_min[col] = first[col]
            self.num_max[col] = first[col]
        
        #reset all counts, sum and windows to 0
        for label in self.labels:
            for col in self.cat_cols:
                self.windows[col][label] = []
                self.windows_counts[col][label] = 0
                for val in self.categories[col]:
                    self.cat_counts[col][label][val] = 0
            for col in self.num_cols:
                self.num_sums[col][label] = 0
                self.num_counts[col][label] = 0
                self.windows[col][label] = []
                self.windows_counts[col][label] = 0

        #add all initial data to windows
        for index, row in data.iterrows():
            self._add_to_counts(row)
    
    def _add_to_counts(self, inst):
        """
        adds an instance to the counting statistics
        :param inst: labeled inst
        """
        label = inst[self.target_col_name]

        #numerical columns
        for col in self.num_cols:
            val = inst[col]
            #skip missing values
            if self.isnan(val):
                continue

            if self.num_min[col] > val: self.num_min[col] = val
            if self.num_max[col] < val: self.num_max[col] = val

            #add to it
            self.windows[col][label].append(val)
            self.windows_counts[col][label] += 1
            self.num_counts[col][label] += 1
            self.num_sums[col][label] += val

            #check for windows
            if self.windows_counts[col][label] > self.window_size:
                val = self.windows[col][label][0]

                self.windows[col][label].pop(0)
                self.windows_counts[col][label] -= 1
                self.num_counts[col][label] -= 1
                self.num_sums[col][label] -= val

                #handle new normalization
                if self.num_min[col] == val:
                    min_val = self.num_min[col]
                    for l in self.windows[col].keys():
                        if min(self.windows[col][l]) < min_val: min_val = min(self.windows[col][l])
                    self.num_min[col] = min_val

                if self.num_max[col] == val: 
                    max_val = self.num_max[col]
                    for l in self.windows[col].keys():
                        if max(self.windows[col][l]) > max_val: max_val = max(self.windows[col][l])
                    self.num_max[col] = max_val

        #categorical columns
        for col in self.cat_cols:
            val = inst[col]
            if not self.isnan(inst[col]):
                self.cat_counts[col][label][val] += 1
                self.windows[col][label].append(val)
                self.windows_counts[col][label] += 1

                #check for windows
                if self.windows_counts[col][label] > self.window_size:
                    self.windows_counts[col][label] -= 1
                    self.cat_counts[col][label][val] -= 1
                    self.windows[col][label].pop(0)
    
    def _on_new_batch(self, data):
        pass
    
    def _update_window(self, inst):      
        """
        adds an instance to the counting statistics
        :param inst: labeled inst
        """
        label = inst[self.target_col_name]

        #numerical columns
        for col in self.num_cols:
            val = inst[col]
            #skip missing values
            if self.isnan(val):
                continue

            if self.num_min[col] > val: self.num_min[col] = val
            if self.num_max[col] < val: self.num_max[col] = val

            #add to it
            self.windows[col][label].append(val)
            self.windows_counts[col][label] += 1
            self.num_counts[col][label] += 1
            self.num_sums[col][label] += val

            #check for windows
            if self.windows_counts[col][label] > self.window_size:
                val = self.windows[col][label][0]

                self.windows[col][label].pop(0)
                self.windows_counts[col][label] -= 1
                self.num_counts[col][label] -= 1
                self.num_sums[col][label] -= val

                #handle new normalization
                if self.num_min[col] == val:
                    min_val = self.num_min[col]
                    for l in self.windows[col].keys():
                        if min(self.windows[col][l]) < min_val: min_val = min(self.windows[col][l])
                    self.num_min[col] = min_val

                if self.num_max[col] == val: 
                    max_val = self.num_max[col]
                    for l in self.windows[col].keys():
                        if max(self.windows[col][l]) > max_val: max_val = max(self.windows[col][l])
                    self.num_max[col] = max_val

        #categorical columns
        for col in self.cat_cols:
            val = inst[col]
            if not self.isnan(inst[col]):
                self.cat_counts[col][label][val] += 1
                self.windows[col][label].append(val)
                self.windows_counts[col][label] += 1

                #check for windows
                if self.windows_counts[col][label] > self.window_size:
                    self.windows_counts[col][label] -= 1
                    self.cat_counts[col][label][val] -= 1
                    self.windows[col][label].pop(0)      

    def get_name(self):
        return "MWAED+" + self.feature_selection.get_name()
                                
class SingleWindowAED(AbstractAED):
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size,dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Active Feature Acquisition strategy that ranks features accoring to average euclidean distance
        Requires the exact categories and possible category values beforehand
        :param window: set as same window as used in framework
        :param feature_selection: the method by which the feature set for acquisition is selected
        :param dynamic_budget_threshold: whether the budget threshold should be dynamically adjusted before each batch
        :param categories: provide categories for all categorical features as dict 
        alternatively if left None make initally labeled data in window represent 
        all categorical features at least once 
        """
        super().__init__(target_col_name=target_col_name,
                            window=window,
                            feature_selection=feature_selection,
                            dynamic_budget_threshold=dynamic_budget_threshold,
                            budget_manager=budget_manager,
                            acq_set_size=acq_set_size,
                            acquisition_costs=acquisition_costs,
                            categories=categories,
                            budget_option=budget_option,
                            debug=debug)
        
    def _on_new_batch(self, data):
        """
        resets sums and counts of data to 0 and counts and adds data
        sets max and min
        """
        #Mean values for each feature given label
        #  sum of features given label for numerical features
        #  count of features given label for numerical features
        #  count of features given label and value for categorical features
        
        #normalize for sum and 0 to 1 distances
        #wd = self._normalize(data=data)

        #set sums and counts back to 0
        for label in self.labels:
            for col in self.num_cols:
                self.num_counts[col][label] = 0
                self.num_sums[col][label] = 0
            for col in self.cat_cols:
                for val in self.categories[col]:
                    self.cat_counts[col][label][val] = 0

        #set max and min
        for col in self.num_cols:
            self.num_max[col] = data[col].max()
            self.num_min[col] = data[col].min()

        #add each row to the counts and sums
        for index, row in data.iterrows():
            #assume each entry in window to have label
            label = row[self.target_col_name]
            for col in self.num_cols:
                #skip nans
                if self.isnan(row[col]):
                    continue
                self.num_counts[col][label] += 1
                self.num_sums[col][label] += row[col]
            for col in self.cat_cols:
                #skip nans
                if self.isnan(row[col]):
                    continue
                val = row[col]
                self.cat_counts[col][label][val] += 1
    
    def _update_window(self, inst):
        """
        adds an inst to the counting statistics
        :param inst: labeled inst
        """
        label = inst[self.target_col_name]

        #numerical columns
        for col in self.num_cols:
            val = inst[col]
            #skip missing values
            if self.isnan(val):
                continue

            if self.num_min[col] > val: self.num_min[col] = val
            if self.num_max[col] < val: self.num_max[col] = val

            #add to it
            self.num_counts[col][label] += 1
            self.num_sums[col][label] += val

        #categorical columns
        for col in self.cat_cols:
            val = inst[col]
            if not self.isnan(inst[col]):
                self.cat_counts[col][label][val] += 1
    
    def get_name(self):
        return "SWAED+" + self.feature_selection.get_name()

class SWAEDFeatureCorrelationCorrected(SingleWindowAED):
    def __init__(self, window, pipeline, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Active Feature Acquisition strategy that ranks features accoring to average euclidean distance
        Requires the exact categories and possible category values beforehand
        :param window: set as same window as used in framework
        :param feature_selection: the method by which the feature set for acquisition is selected
        :param dynamic_budget_threshold: whether the budget threshold should be dynamically adjusted before each batch
        :param categories: provide categories for all categorical features as dict 
        alternatively if left None make initally labeled data in window represent 
        all categorical features at least once 
        """
        super().__init__(window=window, 
                            target_col_name=target_col_name,
                            budget_manager=budget_manager,
                            feature_selection=feature_selection,
                            acq_set_size=acq_set_size, 
                            dynamic_budget_threshold=dynamic_budget_threshold, 
                            categories=categories, 
                            acquisition_costs=acquisition_costs, 
                            budget_option=budget_option, 
                            debug=debug)
        self.pipeline = pipeline
        self.correlations = None
    
    def _get_miss_merits(self, inst, merits):
        """
        returns all merits of features missing in an instance
        :param merits: the global feature merits
        """
        miss_merits = {}
        known_features = [(1 if not self.isnan(inst[col]) else 0) for col in self.num_cols]
        for i, col in enumerate(self.num_cols):
            if not known_features[i]:
                min_vals = np.extract(known_features, self.correlations[i])
                miss_merits[col] = merits[col] * np.min(min_vals) if any(min_vals) else merits[col]
        return miss_merits
        #return {col:merits[col] * np.min(np.extract(known_features, self.correlations[i])) for i, col in enumerate(self.num_cols) if not known_features[i]}

    def _on_new_batch(self, data):
        super()._on_new_batch(data)
        dt = self.pipeline.fit_transform(data, data[self.target_col_name])
        self.correlations = 1 - np.abs(np.corrcoef(dt.T))
        #np.corrcoef(data[self.num_cols].T)

    def get_name(self):
        return "SWAEDFCC+" + self.feature_selection.get_name()
    