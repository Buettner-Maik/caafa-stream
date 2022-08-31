import pandas as pd
import numpy as np
import math
import random
import osm.data_streams.constants as const

from osm.data_streams.active_feature_acquisition.abstract_feature_acquisition import AbstractActiveFeatureAcquisitionStrategy

class RandomFeatureAcquisition(AbstractActiveFeatureAcquisitionStrategy):
    def __init__(self, target_col_name, budget_manager, acq_set_size=1, columns=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        An active feature acquisition method that selects features to request at random
        :param acq_set_size: The maximum set size that the method attempts to acquire per instance
        :param columns: If None gets all columns to pick from on first call of get_data
        from provided data
        :param budget_manager: The Budgeting strategy used will only be given None as parameter
        for its acquisition
        :param budget_option: specify a list of tuples consisting of points at which budget is distributed and budget gain function
        accepted strings as position: 'once', 'batch', 'inst', 'acq'
        """
        super().__init__(target_col_name=target_col_name, 
                        budget_manager=budget_manager, 
                        acquisition_costs=acquisition_costs,
                        acq_set_size=acq_set_size,
                        budget_option=budget_option,
                        debug=debug)
        self.columns = columns
        self._initialized = False
    
    def _initialize(self, data):
        """
        Gets the columns the data has for use in the random number generator and
        prepares self.queried
        """
        if self.columns is None:
            self.columns = list(data.columns.values)
            self.columns.remove(self.target_col_name)

        self._initialize_queried(self.columns)
        self._initialized = True

    def get_data(self, data):
        """
        Returns the incoming data with additionally acquired features
        :param data: A pandas.DataFrame with the data to do active feature acquisition on
        """
        if not self._initialized:
            self._initialize(data)
            if self._budget_once:
                self.budget_manager.add_budget(self._bgain_once)

        if hasattr(data, "iterrows"):
            if self._budget_batch:
                self.budget_manager.add_budget(self._bgain_batch)
            for index, row in data.iterrows():
                if self._budget_inst:
                    self.budget_manager.add_budget(self._bgain_inst)
                #get all nan containing col indices in row
                nan_cols = []
                miss_fea = 0
                for col in self.columns:
                    if self.isnan(row[col]):
                        self.miss_counts[col] += 1
                        nan_cols.append(col)
                        miss_fea += 1
                if miss_fea > 0:
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    #Randomize which to pick
                    cols = random.sample(nan_cols, min(self.acq_set_size, miss_fea))
                    #rnd = math.floor(np.random.uniform(high=len(nan_cols)))
                    #col = self.columns[nan_cols[rnd]]

                    acq_costs = sum([self.acquisition_costs.get(col, 1) for col in cols])
                    #apply budgeting
                    if self.budget_manager.acquire(0.0, acq_costs):
                        for col in cols:
                            feature_val = self.get_feature(row, col)
                            data.loc[[index], [col]] = feature_val
                            self.queried[col] += 1
        else:
            raise ValueError("A pandas.DataFrame expected")

        return data

    def get_name(self):
        return "RA"

# class RandomDynamicBudgetFeatureAcquisition(RandomFeatureAcquisition):
    # def __init__(self, target_col_name, budget_manager, acq_set_size=1, columns=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        # """
        # Revised RandomFeatureAcquisition class intended for debugging and logging budget manager behavior
        # """
        # super().__init__(target_col_name=target_col_name,
                         # budget_manager=budget_manager,
                         # acq_set_size=acq_set_size,
                         # columns=columns,
                         # acquisition_costs=acquisition_costs,
                         # budget_option=budget_option,
                         # debug=debug)
        # self.i_max = sum(sorted(self.acquisition_costs.values(), reverse=True)[:min(self.acq_set_size, len(self.acquisition_costs))])
        
    # def get_data(self):
        # """
        # Returns the incoming data with additionally acquired features
        # :param data: A pandas.DataFrame with the data to do active feature acquisition on
        # """
        # if not self._initialized:
            # self._initialize(data)
            # if self._budget_once:
                # self.budget_manager.add_budget(self._bgain_once)

        # if hasattr(data, "iterrows"):
            # if self._budget_batch:
                # self.budget_manager.add_budget(self._bgain_batch)
                
            # #dynamic budget threshold
            # exp_threshold = self._bgain_inst / self.i_max
            # new_threshold = exp_threshold / self.budget_manager.used_budget()
            # self.budget_manager.budget_threshold = min(new_threshold, 1)
                
            # #instance logging
            # self.i_queried = {col: [] for col in self.merits}
            # self.i_answered = {col: [] for col in self.merits}
            # self.i_miss_counts = {col: [] for col in self.merits}
                
            # for index, row in data.iterrows():
                # if self._budget_inst:
                    # self.budget_manager.add_budget(self._bgain_inst)
                # #get all nan containing col indices in row
                # nan_cols = []
                # miss_fea = 0
                # for col in self.columns:
                    # if self.isnan(row[col]):
                        # self.miss_counts[col] += 1
                        # nan_cols.append(col)
                        # miss_fea += 1
                # if self._budget_acq:
                    # self.budget_manager.add_budget(self._bgain_acq)
                # #Randomize which to pick
                # cols = random.sample(nan_cols, min(self.acq_set_size, miss_fea))
                # #rnd = math.floor(np.random.uniform(high=len(nan_cols)))
                # #col = self.columns[nan_cols[rnd]]

                # acq_costs = sum([self.acquisition_costs.get(col, 1) for col in cols])
                # #apply budgeting
                # if self.budget_manager.acquire(0.0, acq_costs):
                    # for col in cols:
                        # feature_val = self.get_feature(row, col)
                        # data.loc[[index], [col]] = feature_val
                        # self.queried[col] += 1
        # else:
            # raise ValueError("A pandas.DataFrame expected")

        # return data
    
    # def get_stats(self, index=0):
        # pd_index = pd.MultiIndex.from_product([[const.missing_features, const.active_feature_acquisition_queries, const.active_feature_acquisition_answers], [x for x in self.queried]])
        # stats = pd.DataFrame(index=pd_index).T.copy()
        
        # for col, count in self.i_miss_counts.items():
            # stats[(const.missing_features, col)] = stats[(const.missing_features, col)].astype('object')
            # stats.loc[index, (const.missing_features, col)] = str(count)
        # for col, count in self.i_queried.items():
            # stats[(const.active_feature_acquisition_queries, col)] = stats[(const.active_feature_acquisition_queries, col)].astype('object')
            # stats.loc[index, (const.active_feature_acquisition_queries, col)] = str(count)
        # for col, count in self.i_answered.items():
            # stats[(const.active_feature_acquisition_answers, col)] = stats[(const.active_feature_acquisition_answers, col)].astype('object')
            # stats.loc[index, (const.active_feature_acquisition_answers, col)] = str(count)
        
        # return stats
    
    
    # def get_name(self):
        # return "RADB"

    # # def get_stats(self, index=0):
        # # pd_index = pd.MultiIndex.from_product([[const.active_feature_acquisition_merits, const.active_feature_acquisition_queries], [x for x in self.queried.keys()]])
        # # stats = pd.DataFrame(index=pd_index).T.copy()
        
        # # for col, count in self.queried.items():
            # # stats.loc[index, (const.active_feature_acquisition_queries, col)] = count        
        # # for col, count in self.queried.items():
            # # stats.loc[index, (const.active_feature_acquisition_merits, col)] = 1
        
        # # return stats
