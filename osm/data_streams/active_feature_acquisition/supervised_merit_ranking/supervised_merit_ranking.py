from abc import ABC, abstractmethod
import pandas as pd
from pandas.api.types import is_numeric_dtype
import math
import operator
import copy
import osm.data_streams.constants as const

from osm.data_streams.windows.abstract_window import AbstractWindow

from osm.data_streams.active_feature_acquisition.abstract_feature_acquisition import AbstractActiveFeatureAcquisitionStrategy
from osm.data_streams.active_feature_acquisition.supervised_merit_ranking.smr_feature_set_selection import AbstractSMRFeatureSetSelection

# RA wirft alle Ergebnisse in selbes Verzeichnis ohne Rücksicht auf feature set groesse
# (fixed?) KPickyBest laesst die k+1 merits durch
# (fixed?) KPickyBest indexiert falsch und wirft Fehler, wenn k groesser gleich feature anzahl

#Exec order: Dataset-Iteration-MissChance-Costs-Budget-SetSize-Task
#Path order: Dataset-results-Iteration-MissChance-Costs-AFA-FSS-BM-Budget

#AbstractActiveFeatureAcquistionStrategy
#  queried:dict  (featureName:count)
#  miss_counts:dict    (featureName:count)
#  instances_processed:int
#  target_col_name:string
#  acq_set_size:int
#  acquisition_costs:dict    (featureName:cost)
#  budget_manager:BudgetManager
#  _budget_once:bool
#  _budget_batch:bool
#  _budget_inst:bool
#  _budget_acq:bool
#  _bgain_once:int
#  _bgain_batch:int
#  _bgain_inst:int
#  _bgain_acq:int
#  
#  AbstractSMR
#    dynamic_budget_threshold:bool
#    feature_selection:AbstractSMRFeatureSetSelection
#    merits:dict (featureName, merit)
#    window:AbstractWindow
#    categories:dict  (featureName:[possibleValues])
#    initialized:bool
#    
#    AbstractEntropy
#      initialized:bool
#      pid:PartitionIncrementalDiscretizer
#      
#      SingleWindowIG
#        
#      SingleWindowSU
#
#    AbstractAED
#      initilized:bool
#      
#      SingleWindowAED
#
#      MultiWindowAED
#
#
#AbstractBudgetManager
#  budget_threshold:float
#  budget:int
#  budget_spent:int
#  default_budget_gain:int
#  called:int
#  queried:int
#  
#  IncrementalPercentileFilter
#    counter:int
#    window_size:int
#    values_list:list(float)
#    values:sortedlist(float)
#
#AbstractSMRFeatureSetSelection
#  set_cost_total:int 
#  k:int
#
#  KBestSMRFSS
#
#  KPickyBestSMRFSS
#    rank:int
#
#  KRandomSMRFSS
#
#  KQualityBestSMRFSS
#


class AbstractSMR(AbstractActiveFeatureAcquisitionStrategy):
    #TODO: integrate differing costs
    #TODO: get function to tell if label known
    #TODO: get the acquisition module
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Abstract Active Feature Acquisition strategies that rank entire features accoring to calculated merits
        and chooses features to acquire iteratively
        Requires the exact categories and possible category values beforehand
        Abstract basis for other different merit mechanisms
        Implement get_data(self, data) to be used
        :param window: the framework window containing the initially labeled data
        :param feature_selection: the method by which the feature set for acquisition is selected
        :param dynamic_budget_threshold: whether the budget threshold should be dynamically adjusted before each batch
        :param categories: provide categories for all categorical features as dict         
        empty categorical names are interpreted as numerical columns
        alternatively if left None make initally labeled data in window represent 
        all categorical features at least once 
        :param budget_option: specify a list of tuples consisting of points at which budget is distributed and budget gain function
        accepted strings as position: 'once', 'batch', 'inst', 'acq'
        """
        if feature_selection is None or not isinstance(feature_selection, AbstractSMRFeatureSetSelection):
            raise ValueError("The feature_selection must be of type AbstractSMRFeatureSetSelection")
        if not categories == None and not isinstance(categories, dict):
            raise ValueError("categories is not a dictionary nor left none")
        if not isinstance(window, AbstractWindow):
            raise ValueError("The window containing the ild must be the same instance of AbstractWindow as the framework uses")
        super().__init__(target_col_name=target_col_name,
                            budget_manager=budget_manager,
                            acquisition_costs=acquisition_costs,
                            budget_option=budget_option,
                            acq_set_size=acq_set_size,
                            debug=debug)
        self.feature_selection = feature_selection
        self.dynamic_budget_threshold = dynamic_budget_threshold
        self.merits = {}
        self.window = window
        self.categories = categories        
        self.initialized = False

    def _initialize(self, data):
        """
        determines labels and numerical / categorical columns on first run
        execute once before starting process to initialize all dicts and labels
        Sets self.initialized to true
        is separated from __init__ as columns and other only become available once data is provided
        :param data: the data all columns and categories will be based on
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Provided data is not a pandas.DataFrame")

        #set columns
        self.cat_cols = []
        self.num_cols = []
        if self.categories == None:
            self.categories = {}
            for col in data:
                if self._isCategorical(data, col):
                    self.cat_cols.append(col)
                    self.categories[col] = data[col].unique()
                elif self._isNumerical(data, col):
                    self.num_cols.append(col)
        else:
            for col in self.categories:
                #all empty lists are num_cols
                if len(self.categories[col]) == 0:
                    self.num_cols.append(col)
                else:   
                    self.cat_cols.append(col)

        #remove target column
        try:
            self.cat_cols.remove(self.target_col_name)
        except ValueError:
            pass
        try:
            self.num_cols.remove(self.target_col_name)
        except ValueError:
            pass

        #get labels
        if self.categories == None or not self.target_col_name in self.categories.keys():
            self.labels = data[self.target_col_name].unique()
        else:
            self.labels = self.categories[self.target_col_name]

        self._initialize_queried(self.cat_cols + self.num_cols)
        self.initialized = True

    def _get_quality(self, inst, merits, acq_merits={}):
        """
        quality is the average of merits of all features known within an instance
        used by IPF when evaluating an acquisition candidate
        an empty instance returns a quality of 0
        the higher the better
        :param merits: the global feature merits
        :param inst: the instance
        :param acq_merits: the merits of further acquisitions
        """
        if not acq_merits:
            i_merit = 0
            known_features = 0
        else:
            i_merit = sum(acq_merits.values())
            known_features = len(acq_merits)
        for key, item in merits.items():
            if not self.isnan(inst[key]):
                known_features += 1
                i_merit += item
        #prevent 0 / 0
        if known_features == 0:
            return 0
        return i_merit / known_features

    def _get_quality_gain(self, inst, merits, acq_merits={}):
        """
        the difference of quality before acquiring additional features versus after 
        for a particular instance
        quality is the average of merits of all features known within an instance
        used by IPF when evaluating an acquisition candidate
        empty instances are handled as having quality 0
        :param merits: the global feature merits
        :param inst: the instance
        :param acq_merits: the merits of further acquisitions
        """
        if not acq_merits:
            return 0

        pre_merit = 0
        pre_known = 0
        for key, item in merits.items():
            if not self.isnan(inst[key]):
                pre_merit += item
                pre_known += 1
        pre_quali = pre_merit / pre_known if pre_known != 0 else 0
        
        pos_merit = sum(acq_merits.values()) + pre_merit
        pos_known = len(acq_merits) + pre_known
        pos_quali = pos_merit / pos_known
        
        return pos_quali - pre_quali
        
    def _get_miss_merits(self, inst, merits):
        """
        returns all merits of features missing in an instance
        :param merits: the global feature merits
        """
        miss_merits = {}
        for key, item in merits.items():
            if self.isnan(inst[key]):
                miss_merits[key] = item
                self.miss_counts[key] += 1
                
        return miss_merits
        
    def _get_merits(self, feature_ranks):
        """
        merits are feature_ranks divided by their acquisition cost
        returns a dict with all of them
        """
        merits = {}
        for key, item in feature_ranks.items():
            cost = self.acquisition_costs[key] if key in self.acquisition_costs else 1
            merits[key] = item / cost
        #print(feature_ranks)
        return merits

    @abstractmethod
    def _get_rank_values(self):
        """
        implement method for calculating ranking values for all features as dict here
        """
        pass
    
    @abstractmethod
    def _on_new_batch(self, data):
        """
        this method gets called at the beginning of each new batch
        this is useful for methods that implement a temporary window for instance-wise decision making
        thus allowing to resync the window
        :param data: the data of the framework window
        """
        pass
    
    @abstractmethod
    def _update_window(self, inst):
        """
        this method is called if the acquisition was successful and the label of 
        the corresponding instance is known
        :param inst: the instance in question
        """
        pass
    
    #def _get_keys(self, listlist, index):
    #    return [a[index] for a in listlist]
    
    def expected_total_batch_costs(self, batch_size=1):
        return sum([misses / self.instances_processed * self.acquisition_costs[feature] for feature, misses in self.miss_counts.items()]) * batch_size
    
    def get_data(self, data):
        """
        gets the data with additionally requested features
        advanced to incorporate gathering of more features simultaneously
        fixed acquisition budget increase
        approach 2, buy all features with merits greater than median merit
        """
        #initialization only possible after representative data in window
        if not self.initialized:                
            self._initialize(self.window.get_window_data())
            if self._budget_once:
                self.budget_manager.add_budget(self._bgain_once)
        
        if hasattr(data, "iterrows"):
            if self._budget_batch:
                self.budget_manager.add_budget(self._bgain_batch)
            
            if self.dynamic_budget_threshold and self.instances_processed > 0:
                self.budget_manager.budget_threshold = self.get_dynamic_threshold(self.feature_selection.get_expected_total_inst_cost(self))
                # i_exp = self.feature_selection.get_expected_total_inst_cost(self)
                # exp_threshold = self._bgain_inst / i_exp
                # used_budget_ratio = self.budget_manager.used_budget()
                # new_threshold = exp_threshold / used_budget_ratio
                # over_budget_rate = math.floor((used_budget_ratio - 1) / 0.0625) + 2
                # if used_budget_ratio > 1: new_threshold /= over_budget_rate
                # self.budget_manager.budget_threshold = min(new_threshold, 1)
                
                #print(i_exp, exp_threshold, new_threshold)
            
            #since window is batch based, reset temporary window to equal the batch window again
            self._on_new_batch(self.window.get_window_data())
            
            #calculate global feature merits, saved into class to allow statistics
            self.merits = self._get_merits(self._get_rank_values())
            
            for index, row in data.iterrows():
                if self._budget_inst:
                    self.budget_manager.add_budget(self._bgain_inst)
                    
                #handle instance
                #miss_feature_merits = {feature: merit}
                miss_feature_merits = self._get_miss_merits(inst=row, merits=self.merits)
                
                #acquisition strategy
                acq_merits, acq_costs = self.feature_selection.get_acquisition_feature_set(self, miss_feature_merits)
                
                if any(acq_merits):
                    quality = self._get_quality_gain(inst=row, merits=self.merits, acq_merits=acq_merits)
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in acq_merits:
                            self.queried[feature] += 1
                            #get feature and replace current inst in iteration and data
                            feature_val = self.get_feature(inst=row, column=feature)
                            row[feature] = feature_val
                            data.loc[[index], [feature]] = feature_val
                
                if self.label_known(inst=row):
                    #update window
                    self._update_window(inst=row)
                    self.merits = self._get_merits(self._get_rank_values())
                    
                self.instances_processed += 1
                    
        else:
            raise ValueError("A pandas.DataFrame expected")

        return data

    def get_stats(self, index=0):
        pd_index = pd.MultiIndex.from_product([[const.missing_features, const.active_feature_acquisition_queries, const.active_feature_acquisition_merits], [x for x in self.queried]])
        stats = pd.DataFrame(index=pd_index).T.copy()
        
        for col, count in self.miss_counts.items():
            stats.loc[index, (const.missing_features, col)] = count
        for col, count in self.queried.items():
            stats.loc[index, (const.active_feature_acquisition_queries, col)] = count        
        for col, merit in self.merits.items():
            stats.loc[index, (const.active_feature_acquisition_merits, col)] = merit
        
        return stats