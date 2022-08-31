import pandas as pd
import copy
import osm.data_streams.constants as const
import math

from osm.data_streams.active_feature_acquisition.supervised_merit_ranking.average_euclidean_distance import SingleWindowAED
from osm.data_streams.active_feature_acquisition.supervised_merit_ranking.entropy_based import SingleWindowIG, SingleWindowSU


class SWAED_T(SingleWindowAED): #single instance decision tracking with quality instead of quality_gain as bm info
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size,dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Copy of SingleWindowAED class modified to suit special presentation and or debug purposes
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
                #dynamic budget threshold
                i_exp = self.feature_selection.get_expected_total_inst_cost(self)
                exp_threshold = self._bgain_inst / i_exp
                new_threshold = exp_threshold / self.budget_manager.used_budget()
                self.budget_manager.budget_threshold = min(new_threshold, 1)
                
                # print(i_exp, exp_threshold, new_threshold)
            
            #since window is batch based, reset temporary window to equal the batch window again
            self._on_new_batch(self.window.get_window_data())
            
            #calculate global feature merits, saved into class to allow statistics
            self.merits = self._get_merits(self._get_rank_values())
            
            # TEST STUFF
            self.i_queried = {col: [] for col in self.merits}
            self.i_answered = {col: [] for col in self.merits}
            self.i_merits = {col: [] for col in self.merits}
            self.i_miss_counts = {col: [] for col in self.merits}
            # ----------
            
            for index, row in data.iterrows():
                if self._budget_inst:
                    self.budget_manager.add_budget(self._bgain_inst)
                    
                #handle instance
                miss_feature_merits = self._get_miss_merits(inst=row, merits=self.merits)
                
                #acquisition strategy
                acq_merits, acq_costs = self.feature_selection.get_acquisition_feature_set(self, miss_feature_merits)
                
                # TEST STUFF                
                for col, merit in self.merits.items():
                    if self.isnan(row[col]): self.i_miss_counts[col].append(1)
                    else: self.i_miss_counts[col].append(0)
                    self.i_merits[col].append(merit)
                    if col in [merit[0] for merit in acq_merits]: self.i_queried[col].append(1)
                    else: self.i_queried[col].append(0)
                    self.i_answered[col].append(0)
                # ----------
                
                if any(acq_merits):
                    quality = self._get_quality(inst=row, merits=self.merits, acq_merits=[m[1] for m in acq_merits])
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in self._get_keys(acq_merits, 0):     
                            self.queried[feature] += 1
                            
                            # TEST STUFF
                            self.i_answered[feature][-1] = 1
                            # ----------
                            
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
        pd_index = pd.MultiIndex.from_product([[const.missing_features, const.active_feature_acquisition_queries, const.active_feature_acquisition_answers, const.active_feature_acquisition_merits], [x for x in self.queried]])
        stats = pd.DataFrame(index=pd_index).T.copy()
        
        for col, count in self.i_miss_counts.items():
            stats[(const.missing_features, col)] = stats[(const.missing_features, col)].astype('object')
            stats.loc[index, (const.missing_features, col)] = str(count)
        for col, count in self.i_queried.items():
            stats[(const.active_feature_acquisition_queries, col)] = stats[(const.active_feature_acquisition_queries, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_queries, col)] = str(count)
        for col, count in self.i_answered.items():
            stats[(const.active_feature_acquisition_answers, col)] = stats[(const.active_feature_acquisition_answers, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_answers, col)] = str(count)
        for col, merit in self.i_merits.items():
            stats[(const.active_feature_acquisition_merits, col)] = stats[(const.active_feature_acquisition_merits, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_merits, col)] = str(merit)
        
        return stats
        
    def get_name(self):
        return "SWAED_T+" + self.feature_selection.get_name()

class SWAED_Q(SingleWindowAED): #single instance decision tracking
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size,dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Copy of SingleWindowAED class modified to suit special presentation and or debug purposes
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
                #dynamic budget threshold
                i_exp = self.feature_selection.get_expected_total_inst_cost(self)
                exp_threshold = self._bgain_inst / i_exp
                new_threshold = exp_threshold / self.budget_manager.used_budget()
                self.budget_manager.budget_threshold = min(new_threshold, 1)
                
                # print(i_exp, exp_threshold, new_threshold)
            
            #since window is batch based, reset temporary window to equal the batch window again
            self._on_new_batch(self.window.get_window_data())
            
            #calculate global feature merits, saved into class to allow statistics
            self.merits = self._get_merits(self._get_rank_values())
            
            # TEST STUFF
            self.i_queried = {col: [] for col in self.merits}
            self.i_answered = {col: [] for col in self.merits}
            self.i_merits = {col: [] for col in self.merits}
            self.i_miss_counts = {col: [] for col in self.merits}
            # ----------
            
            for index, row in data.iterrows():
                if self._budget_inst:
                    self.budget_manager.add_budget(self._bgain_inst)
                    
                #handle instance
                miss_feature_merits = self._get_miss_merits(inst=row, merits=self.merits)
                
                #acquisition strategy
                acq_merits, acq_costs = self.feature_selection.get_acquisition_feature_set(self, miss_feature_merits)
                
                # TEST STUFF                
                for col, merit in self.merits.items():
                    if self.isnan(row[col]): self.i_miss_counts[col].append(1)
                    else: self.i_miss_counts[col].append(0)
                    self.i_merits[col].append(merit)
                    if col in [merit[0] for merit in acq_merits]: self.i_queried[col].append(1)
                    else: self.i_queried[col].append(0)
                    self.i_answered[col].append(0)
                # ----------
                
                if any(acq_merits):
                    q_before = self._get_quality(inst=row, merits=self.merits, acq_merits=[])
                    q_after = self._get_quality(inst=row, merits=self.merits, acq_merits=[m[1] for m in acq_merits])
                    quality = q_after - q_before
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in self._get_keys(acq_merits, 0):     
                            self.queried[feature] += 1
                            
                            # TEST STUFF
                            self.i_answered[feature][-1] = 1
                            # ----------
                            
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
        pd_index = pd.MultiIndex.from_product([[const.missing_features, const.active_feature_acquisition_queries, const.active_feature_acquisition_answers, const.active_feature_acquisition_merits], [x for x in self.queried]])
        stats = pd.DataFrame(index=pd_index).T.copy()
        
        for col, count in self.i_miss_counts.items():
            stats[(const.missing_features, col)] = stats[(const.missing_features, col)].astype('object')
            stats.loc[index, (const.missing_features, col)] = str(count)
        for col, count in self.i_queried.items():
            stats[(const.active_feature_acquisition_queries, col)] = stats[(const.active_feature_acquisition_queries, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_queries, col)] = str(count)
        for col, count in self.i_answered.items():
            stats[(const.active_feature_acquisition_answers, col)] = stats[(const.active_feature_acquisition_answers, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_answers, col)] = str(count)
        for col, merit in self.i_merits.items():
            stats[(const.active_feature_acquisition_merits, col)] = stats[(const.active_feature_acquisition_merits, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_merits, col)] = str(merit)
        
        return stats
        
    def get_name(self):
        return "SWAED_Q+" + self.feature_selection.get_name()

class SWAED_I(SingleWindowAED): #single instance decision tracking and imax as dynamic budget threshold
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size,dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Copy of SingleWindowAED class modified to suit special presentation and or debug purposes
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
        self.i_max = sum(sorted(self.acquisition_costs.values(), reverse=True)[:min(self.feature_selection.k, len(self.acquisition_costs))])
    
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
                #dynamic budget threshold
                #i_exp = self.feature_selection.get_expected_total_inst_cost(self)
                exp_threshold = self._bgain_inst / self.i_max
                new_threshold = exp_threshold / self.budget_manager.used_budget()
                self.budget_manager.budget_threshold = min(new_threshold, 1)
                
                # print(i_exp, exp_threshold, new_threshold)
            
            #since window is batch based, reset temporary window to equal the batch window again
            self._on_new_batch(self.window.get_window_data())
            
            #calculate global feature merits, saved into class to allow statistics
            self.merits = self._get_merits(self._get_rank_values())
            
            # TEST STUFF
            self.i_queried = {col: [] for col in self.merits}
            self.i_answered = {col: [] for col in self.merits}
            self.i_merits = {col: [] for col in self.merits}
            self.i_miss_counts = {col: [] for col in self.merits}
            # ----------
            
            for index, row in data.iterrows():
                if self._budget_inst:
                    self.budget_manager.add_budget(self._bgain_inst)
                    
                #handle instance
                miss_feature_merits = self._get_miss_merits(inst=row, merits=self.merits)
                
                #acquisition strategy
                acq_merits, acq_costs = self.feature_selection.get_acquisition_feature_set(self, miss_feature_merits)
                
                # TEST STUFF                
                for col, merit in self.merits.items():
                    if self.isnan(row[col]): self.i_miss_counts[col].append(1)
                    else: self.i_miss_counts[col].append(0)
                    self.i_merits[col].append(merit)
                    if col in [merit[0] for merit in acq_merits]: self.i_queried[col].append(1)
                    else: self.i_queried[col].append(0)
                    self.i_answered[col].append(0)
                # ----------
                
                if any(acq_merits):
                    q_before = self._get_quality(inst=row, merits=self.merits, acq_merits=[])
                    q_after = self._get_quality(inst=row, merits=self.merits, acq_merits=[m[1] for m in acq_merits])
                    quality = q_after - q_before
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in self._get_keys(acq_merits, 0):     
                            self.queried[feature] += 1
                            
                            # TEST STUFF
                            self.i_answered[feature][-1] = 1
                            # ----------
                            
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
        pd_index = pd.MultiIndex.from_product([[const.missing_features, const.active_feature_acquisition_queries, const.active_feature_acquisition_answers, const.active_feature_acquisition_merits], [x for x in self.queried]])
        stats = pd.DataFrame(index=pd_index).T.copy()
        
        for col, count in self.i_miss_counts.items():
            stats[(const.missing_features, col)] = stats[(const.missing_features, col)].astype('object')
            stats.loc[index, (const.missing_features, col)] = str(count)
        for col, count in self.i_queried.items():
            stats[(const.active_feature_acquisition_queries, col)] = stats[(const.active_feature_acquisition_queries, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_queries, col)] = str(count)
        for col, count in self.i_answered.items():
            stats[(const.active_feature_acquisition_answers, col)] = stats[(const.active_feature_acquisition_answers, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_answers, col)] = str(count)
        for col, merit in self.i_merits.items():
            stats[(const.active_feature_acquisition_merits, col)] = stats[(const.active_feature_acquisition_merits, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_merits, col)] = str(merit)
        
        return stats
        
    def get_name(self):
        return "SWAED_I+" + self.feature_selection.get_name()

class SWAED_D(SingleWindowAED): #single instance decision tracking and dynamic budgeting until 500 instances
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Copy of SingleWindowAED class modified to suit special presentation and or debug purposes
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
            
            if self.dynamic_budget_threshold and self.instances_processed > 0 and self.instances_processed <= 500:
                #dynamic budget threshold
                i_exp = self.feature_selection.get_expected_total_inst_cost(self)
                exp_threshold = self._bgain_inst / i_exp
                new_threshold = exp_threshold / self.budget_manager.used_budget()
                self.budget_manager.budget_threshold = min(new_threshold, 1)
                
                # print(i_exp, exp_threshold, new_threshold)
            
            #since window is batch based, reset temporary window to equal the batch window again
            self._on_new_batch(self.window.get_window_data())
            
            #calculate global feature merits, saved into class to allow statistics
            self.merits = self._get_merits(self._get_rank_values())
            
            # TEST STUFF
            self.i_queried = {col: [] for col in self.merits}
            self.i_answered = {col: [] for col in self.merits}
            self.i_merits = {col: [] for col in self.merits}
            self.i_miss_counts = {col: [] for col in self.merits}
            # ----------
            
            for index, row in data.iterrows():
                if self._budget_inst:
                    self.budget_manager.add_budget(self._bgain_inst)
                    
                #handle instance
                miss_feature_merits = self._get_miss_merits(inst=row, merits=self.merits)
                
                #acquisition strategy
                acq_merits, acq_costs = self.feature_selection.get_acquisition_feature_set(self, miss_feature_merits)
                
                # TEST STUFF                
                for col, merit in self.merits.items():
                    if self.isnan(row[col]): self.i_miss_counts[col].append(1)
                    else: self.i_miss_counts[col].append(0)
                    self.i_merits[col].append(merit)
                    if col in [merit[0] for merit in acq_merits]: self.i_queried[col].append(1)
                    else: self.i_queried[col].append(0)
                    self.i_answered[col].append(0)
                # ----------
                
                if any(acq_merits):
                    q_before = self._get_quality(inst=row, merits=self.merits, acq_merits=[])
                    q_after = self._get_quality(inst=row, merits=self.merits, acq_merits=[m[1] for m in acq_merits])
                    quality = q_after - q_before
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in self._get_keys(acq_merits, 0):     
                            self.queried[feature] += 1
                            
                            # TEST STUFF
                            self.i_answered[feature][-1] = 1
                            # ----------
                            
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
        pd_index = pd.MultiIndex.from_product([[const.missing_features, const.active_feature_acquisition_queries, const.active_feature_acquisition_answers, const.active_feature_acquisition_merits], [x for x in self.queried]])
        stats = pd.DataFrame(index=pd_index).T.copy()
        
        for col, count in self.i_miss_counts.items():
            stats[(const.missing_features, col)] = stats[(const.missing_features, col)].astype('object')
            stats.loc[index, (const.missing_features, col)] = str(count)
        for col, count in self.i_queried.items():
            stats[(const.active_feature_acquisition_queries, col)] = stats[(const.active_feature_acquisition_queries, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_queries, col)] = str(count)
        for col, count in self.i_answered.items():
            stats[(const.active_feature_acquisition_answers, col)] = stats[(const.active_feature_acquisition_answers, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_answers, col)] = str(count)
        for col, merit in self.i_merits.items():
            stats[(const.active_feature_acquisition_merits, col)] = stats[(const.active_feature_acquisition_merits, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_merits, col)] = str(merit)
        
        return stats
        
    def get_name(self):
        return "SWAED_D+" + self.feature_selection.get_name()

class SWAED_C(SingleWindowAED): #single instance decision tracking and dynamic budgeting before each instance
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size,dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Copy of SingleWindowAED class modified to suit special presentation and or debug purposes
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
            
            #since window is batch based, reset temporary window to equal the batch window again
            self._on_new_batch(self.window.get_window_data())
            
            #calculate global feature merits, saved into class to allow statistics
            self.merits = self._get_merits(self._get_rank_values())
            
            # TEST STUFF
            self.i_queried = {col: [] for col in self.merits}
            self.i_answered = {col: [] for col in self.merits}
            self.i_merits = {col: [] for col in self.merits}
            self.i_miss_counts = {col: [] for col in self.merits}
            # ----------
            
            for index, row in data.iterrows():            
                if self.dynamic_budget_threshold and self.instances_processed > 49:
                    #dynamic budget threshold
                    i_exp = self.feature_selection.get_expected_total_inst_cost(self)
                    exp_threshold = self._bgain_inst / i_exp
                    new_threshold = exp_threshold / self.budget_manager.used_budget()
                    self.budget_manager.budget_threshold = min(new_threshold, 1)
            
                if self._budget_inst:
                    self.budget_manager.add_budget(self._bgain_inst)
                    
                #handle instance
                miss_feature_merits = self._get_miss_merits(inst=row, merits=self.merits)
                
                #acquisition strategy
                acq_merits, acq_costs = self.feature_selection.get_acquisition_feature_set(self, miss_feature_merits)
                
                # TEST STUFF                
                for col, merit in self.merits.items():
                    if self.isnan(row[col]): self.i_miss_counts[col].append(1)
                    else: self.i_miss_counts[col].append(0)
                    self.i_merits[col].append(merit)
                    if col in [merit[0] for merit in acq_merits]: self.i_queried[col].append(1)
                    else: self.i_queried[col].append(0)
                    self.i_answered[col].append(0)
                # ----------
                
                if any(acq_merits):
                    q_before = self._get_quality(inst=row, merits=self.merits, acq_merits=[])
                    q_after = self._get_quality(inst=row, merits=self.merits, acq_merits=[m[1] for m in acq_merits])
                    quality = q_after - q_before
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in self._get_keys(acq_merits, 0):     
                            self.queried[feature] += 1
                            
                            # TEST STUFF
                            self.i_answered[feature][-1] = 1
                            # ----------
                            
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
        pd_index = pd.MultiIndex.from_product([[const.missing_features, const.active_feature_acquisition_queries, const.active_feature_acquisition_answers, const.active_feature_acquisition_merits], [x for x in self.queried]])
        stats = pd.DataFrame(index=pd_index).T.copy()
        
        for col, count in self.i_miss_counts.items():
            stats[(const.missing_features, col)] = stats[(const.missing_features, col)].astype('object')
            stats.loc[index, (const.missing_features, col)] = str(count)
        for col, count in self.i_queried.items():
            stats[(const.active_feature_acquisition_queries, col)] = stats[(const.active_feature_acquisition_queries, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_queries, col)] = str(count)
        for col, count in self.i_answered.items():
            stats[(const.active_feature_acquisition_answers, col)] = stats[(const.active_feature_acquisition_answers, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_answers, col)] = str(count)
        for col, merit in self.i_merits.items():
            stats[(const.active_feature_acquisition_merits, col)] = stats[(const.active_feature_acquisition_merits, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_merits, col)] = str(merit)
        
        return stats
        
    def get_name(self):
        return "SWAED_C+" + self.feature_selection.get_name()

class SWAED_P(SingleWindowAED): #single instance decision tracking and dynamic budgeting penalty term
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size,dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Copy of SingleWindowAED class modified to suit special presentation and or debug purposes
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
        
        b = 1
        if self._budget_inst: b = self._bgain_inst
        max_acq_set = min(self.acq_set_size, len(self.acquisition_costs))
        avg_per_feat_cost = sum(self.acquisition_costs.values()) / len(self.acquisition_costs) # 1, 4.5
        p = b / (avg_per_feat_cost * max_acq_set) # 0.. # 0.25, 0.5, 0.75, 0.06, 0.11, 0.17
        self.penalty_term = min(0.6875 + p / (p + 1), 1)
    
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
                #dynamic budget threshold
                i_exp = self.feature_selection.get_expected_total_inst_cost(self)
                exp_threshold = self._bgain_inst / i_exp
                new_threshold = exp_threshold / self.budget_manager.used_budget() * self.penalty_term
                self.budget_manager.budget_threshold = min(new_threshold, 1)
                
                # print(i_exp, exp_threshold, new_threshold)
            
            #since window is batch based, reset temporary window to equal the batch window again
            self._on_new_batch(self.window.get_window_data())
            
            #calculate global feature merits, saved into class to allow statistics
            self.merits = self._get_merits(self._get_rank_values())
            
            # TEST STUFF
            self.i_queried = {col: [] for col in self.merits}
            self.i_answered = {col: [] for col in self.merits}
            self.i_merits = {col: [] for col in self.merits}
            self.i_miss_counts = {col: [] for col in self.merits}
            # ----------
            
            for index, row in data.iterrows():
                if self._budget_inst:
                    self.budget_manager.add_budget(self._bgain_inst)
                    
                #handle instance
                miss_feature_merits = self._get_miss_merits(inst=row, merits=self.merits)
                
                #acquisition strategy
                acq_merits, acq_costs = self.feature_selection.get_acquisition_feature_set(self, miss_feature_merits)
                
                # TEST STUFF                
                for col, merit in self.merits.items():
                    if self.isnan(row[col]): self.i_miss_counts[col].append(1)
                    else: self.i_miss_counts[col].append(0)
                    self.i_merits[col].append(merit)
                    if col in [merit[0] for merit in acq_merits]: self.i_queried[col].append(1)
                    else: self.i_queried[col].append(0)
                    self.i_answered[col].append(0)
                # ----------
                
                if any(acq_merits):
                    q_before = self._get_quality(inst=row, merits=self.merits, acq_merits=[])
                    q_after = self._get_quality(inst=row, merits=self.merits, acq_merits=[m[1] for m in acq_merits])
                    quality = q_after - q_before
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in self._get_keys(acq_merits, 0):     
                            self.queried[feature] += 1
                            
                            # TEST STUFF
                            self.i_answered[feature][-1] = 1
                            # ----------
                            
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
        pd_index = pd.MultiIndex.from_product([[const.missing_features, const.active_feature_acquisition_queries, const.active_feature_acquisition_answers, const.active_feature_acquisition_merits], [x for x in self.queried]])
        stats = pd.DataFrame(index=pd_index).T.copy()
        
        for col, count in self.i_miss_counts.items():
            stats[(const.missing_features, col)] = stats[(const.missing_features, col)].astype('object')
            stats.loc[index, (const.missing_features, col)] = str(count)
        for col, count in self.i_queried.items():
            stats[(const.active_feature_acquisition_queries, col)] = stats[(const.active_feature_acquisition_queries, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_queries, col)] = str(count)
        for col, count in self.i_answered.items():
            stats[(const.active_feature_acquisition_answers, col)] = stats[(const.active_feature_acquisition_answers, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_answers, col)] = str(count)
        for col, merit in self.i_merits.items():
            stats[(const.active_feature_acquisition_merits, col)] = stats[(const.active_feature_acquisition_merits, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_merits, col)] = str(merit)
        
        return stats
        
    def get_name(self):
        return "SWAED_P+" + self.feature_selection.get_name()

class SWAED_E(SingleWindowAED): #single instance decision tracking and expected dynamic budgeting correction
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size,dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Copy of SingleWindowAED class modified to suit special presentation and or debug purposes
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
                #dynamic budget threshold
                i_exp = self.feature_selection.get_expected_total_inst_cost(self)
                exp_threshold = self._bgain_inst / i_exp
                #new_threshold = exp_threshold / self.budget_manager.used_budget()
                self.budget_manager.budget_threshold = min(exp_threshold, 1)
                
                # print(i_exp, exp_threshold, new_threshold)
            
            #since window is batch based, reset temporary window to equal the batch window again
            self._on_new_batch(self.window.get_window_data())
            
            #calculate global feature merits, saved into class to allow statistics
            self.merits = self._get_merits(self._get_rank_values())
            
            # TEST STUFF
            self.i_queried = {col: [] for col in self.merits}
            self.i_answered = {col: [] for col in self.merits}
            self.i_merits = {col: [] for col in self.merits}
            self.i_miss_counts = {col: [] for col in self.merits}
            # ----------
            
            for index, row in data.iterrows():
                if self._budget_inst:
                    self.budget_manager.add_budget(self._bgain_inst)
                    
                #handle instance
                miss_feature_merits = self._get_miss_merits(inst=row, merits=self.merits)
                
                #acquisition strategy
                acq_merits, acq_costs = self.feature_selection.get_acquisition_feature_set(self, miss_feature_merits)
                
                # TEST STUFF                
                for col, merit in self.merits.items():
                    if self.isnan(row[col]): self.i_miss_counts[col].append(1)
                    else: self.i_miss_counts[col].append(0)
                    self.i_merits[col].append(merit)
                    if col in [merit[0] for merit in acq_merits]: self.i_queried[col].append(1)
                    else: self.i_queried[col].append(0)
                    self.i_answered[col].append(0)
                # ----------
                
                if any(acq_merits):
                    q_before = self._get_quality(inst=row, merits=self.merits, acq_merits=[])
                    q_after = self._get_quality(inst=row, merits=self.merits, acq_merits=[m[1] for m in acq_merits])
                    quality = q_after - q_before
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in self._get_keys(acq_merits, 0):     
                            self.queried[feature] += 1
                            
                            # TEST STUFF
                            self.i_answered[feature][-1] = 1
                            # ----------
                            
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
        pd_index = pd.MultiIndex.from_product([[const.missing_features, const.active_feature_acquisition_queries, const.active_feature_acquisition_answers, const.active_feature_acquisition_merits], [x for x in self.queried]])
        stats = pd.DataFrame(index=pd_index).T.copy()
        
        for col, count in self.i_miss_counts.items():
            stats[(const.missing_features, col)] = stats[(const.missing_features, col)].astype('object')
            stats.loc[index, (const.missing_features, col)] = str(count)
        for col, count in self.i_queried.items():
            stats[(const.active_feature_acquisition_queries, col)] = stats[(const.active_feature_acquisition_queries, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_queries, col)] = str(count)
        for col, count in self.i_answered.items():
            stats[(const.active_feature_acquisition_answers, col)] = stats[(const.active_feature_acquisition_answers, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_answers, col)] = str(count)
        for col, merit in self.i_merits.items():
            stats[(const.active_feature_acquisition_merits, col)] = stats[(const.active_feature_acquisition_merits, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_merits, col)] = str(merit)
        
        return stats
        
    def get_name(self):
        return "SWAED_E+" + self.feature_selection.get_name()

class SWAED_H(SingleWindowAED): #single instance decision tracking and one-sided harsher dynamic budget correction
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size,dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Copy of SingleWindowAED class modified to suit special presentation and or debug purposes
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
                #dynamic budget threshold
                i_exp = self.feature_selection.get_expected_total_inst_cost(self)
                exp_threshold = self._bgain_inst / i_exp
                used_budget_ratio = self.budget_manager.used_budget()
                new_threshold = exp_threshold / used_budget_ratio
                if used_budget_ratio > 1: new_threshold /= 2
                self.budget_manager.budget_threshold = min(new_threshold, 1)
                
                # print(i_exp, exp_threshold, new_threshold)
            
            #since window is batch based, reset temporary window to equal the batch window again
            self._on_new_batch(self.window.get_window_data())
            
            #calculate global feature merits, saved into class to allow statistics
            self.merits = self._get_merits(self._get_rank_values())
            
            # TEST STUFF
            self.i_queried = {col: [] for col in self.merits}
            self.i_answered = {col: [] for col in self.merits}
            self.i_merits = {col: [] for col in self.merits}
            self.i_miss_counts = {col: [] for col in self.merits}
            # ----------
            
            for index, row in data.iterrows():
                if self._budget_inst:
                    self.budget_manager.add_budget(self._bgain_inst)
                    
                #handle instance
                miss_feature_merits = self._get_miss_merits(inst=row, merits=self.merits)
                
                #acquisition strategy
                acq_merits, acq_costs = self.feature_selection.get_acquisition_feature_set(self, miss_feature_merits)
                
                # TEST STUFF                
                for col, merit in self.merits.items():
                    if self.isnan(row[col]): self.i_miss_counts[col].append(1)
                    else: self.i_miss_counts[col].append(0)
                    self.i_merits[col].append(merit)
                    if col in [merit[0] for merit in acq_merits]: self.i_queried[col].append(1)
                    else: self.i_queried[col].append(0)
                    self.i_answered[col].append(0)
                # ----------
                
                if any(acq_merits):
                    q_before = self._get_quality(inst=row, merits=self.merits, acq_merits=[])
                    q_after = self._get_quality(inst=row, merits=self.merits, acq_merits=[m[1] for m in acq_merits])
                    quality = q_after - q_before
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in self._get_keys(acq_merits, 0):     
                            self.queried[feature] += 1
                            
                            # TEST STUFF
                            self.i_answered[feature][-1] = 1
                            # ----------
                            
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
        pd_index = pd.MultiIndex.from_product([[const.missing_features, const.active_feature_acquisition_queries, const.active_feature_acquisition_answers, const.active_feature_acquisition_merits], [x for x in self.queried]])
        stats = pd.DataFrame(index=pd_index).T.copy()
        
        for col, count in self.i_miss_counts.items():
            stats[(const.missing_features, col)] = stats[(const.missing_features, col)].astype('object')
            stats.loc[index, (const.missing_features, col)] = str(count)
        for col, count in self.i_queried.items():
            stats[(const.active_feature_acquisition_queries, col)] = stats[(const.active_feature_acquisition_queries, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_queries, col)] = str(count)
        for col, count in self.i_answered.items():
            stats[(const.active_feature_acquisition_answers, col)] = stats[(const.active_feature_acquisition_answers, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_answers, col)] = str(count)
        for col, merit in self.i_merits.items():
            stats[(const.active_feature_acquisition_merits, col)] = stats[(const.active_feature_acquisition_merits, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_merits, col)] = str(merit)
        
        return stats
        
    def get_name(self):
        return "SWAED_H+" + self.feature_selection.get_name()

class SWAED_H2(SingleWindowAED): #single instance decision tracking and one-sided harsher dynamic budget correction
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size,dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Copy of SingleWindowAED class modified to suit special presentation and or debug purposes
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
                #dynamic budget threshold
                i_exp = self.feature_selection.get_expected_total_inst_cost(self)
                exp_threshold = self._bgain_inst / i_exp
                used_budget_ratio = self.budget_manager.used_budget()
                new_threshold = exp_threshold / used_budget_ratio
                over_budget_rate = math.floor((used_budget_ratio - 1) / 0.0625) + 2
                if used_budget_ratio > 1: new_threshold /= over_budget_rate
                self.budget_manager.budget_threshold = min(new_threshold, 1)
                
                # print(i_exp, exp_threshold, new_threshold)
            
            #since window is batch based, reset temporary window to equal the batch window again
            self._on_new_batch(self.window.get_window_data())
            
            #calculate global feature merits, saved into class to allow statistics
            self.merits = self._get_merits(self._get_rank_values())
            
            # TEST STUFF
            self.i_queried = {col: [] for col in self.merits}
            self.i_answered = {col: [] for col in self.merits}
            self.i_merits = {col: [] for col in self.merits}
            self.i_miss_counts = {col: [] for col in self.merits}
            # ----------
            
            for index, row in data.iterrows():
                if self._budget_inst:
                    self.budget_manager.add_budget(self._bgain_inst)
                    
                #handle instance
                miss_feature_merits = self._get_miss_merits(inst=row, merits=self.merits)
                
                #acquisition strategy
                acq_merits, acq_costs = self.feature_selection.get_acquisition_feature_set(self, miss_feature_merits)
                
                # TEST STUFF                
                for col, merit in self.merits.items():
                    if self.isnan(row[col]): self.i_miss_counts[col].append(1)
                    else: self.i_miss_counts[col].append(0)
                    self.i_merits[col].append(merit)
                    if col in [merit[0] for merit in acq_merits]: self.i_queried[col].append(1)
                    else: self.i_queried[col].append(0)
                    self.i_answered[col].append(0)
                # ----------
                
                if any(acq_merits):
                    q_before = self._get_quality(inst=row, merits=self.merits, acq_merits=[])
                    q_after = self._get_quality(inst=row, merits=self.merits, acq_merits=[m[1] for m in acq_merits])
                    quality = q_after - q_before
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in self._get_keys(acq_merits, 0):     
                            self.queried[feature] += 1
                            
                            # TEST STUFF
                            self.i_answered[feature][-1] = 1
                            # ----------
                            
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
        pd_index = pd.MultiIndex.from_product([[const.missing_features, const.active_feature_acquisition_queries, const.active_feature_acquisition_answers, const.active_feature_acquisition_merits], [x for x in self.queried]])
        stats = pd.DataFrame(index=pd_index).T.copy()
        
        for col, count in self.i_miss_counts.items():
            stats[(const.missing_features, col)] = stats[(const.missing_features, col)].astype('object')
            stats.loc[index, (const.missing_features, col)] = str(count)
        for col, count in self.i_queried.items():
            stats[(const.active_feature_acquisition_queries, col)] = stats[(const.active_feature_acquisition_queries, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_queries, col)] = str(count)
        for col, count in self.i_answered.items():
            stats[(const.active_feature_acquisition_answers, col)] = stats[(const.active_feature_acquisition_answers, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_answers, col)] = str(count)
        for col, merit in self.i_merits.items():
            stats[(const.active_feature_acquisition_merits, col)] = stats[(const.active_feature_acquisition_merits, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_merits, col)] = str(merit)
        
        return stats
        
    def get_name(self):
        return "SWAED_H2+" + self.feature_selection.get_name()

class SWIG_H2(SingleWindowIG): #single instance decision tracking and one-sided harsher dynamic budget correction
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size,dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Copy of SingleWindowAED class modified to suit special presentation and or debug purposes
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
                #dynamic budget threshold
                i_exp = self.feature_selection.get_expected_total_inst_cost(self)
                exp_threshold = self._bgain_inst / i_exp
                used_budget_ratio = self.budget_manager.used_budget()
                new_threshold = exp_threshold / used_budget_ratio
                over_budget_rate = math.floor((used_budget_ratio - 1) / 0.0625) + 2
                if used_budget_ratio > 1: new_threshold /= over_budget_rate
                self.budget_manager.budget_threshold = min(new_threshold, 1)
                
                # print(i_exp, exp_threshold, new_threshold)
            
            #since window is batch based, reset temporary window to equal the batch window again
            self._on_new_batch(self.window.get_window_data())
            
            #calculate global feature merits, saved into class to allow statistics
            self.merits = self._get_merits(self._get_rank_values())
            
            # TEST STUFF
            self.i_queried = {col: [] for col in self.merits}
            self.i_answered = {col: [] for col in self.merits}
            self.i_merits = {col: [] for col in self.merits}
            self.i_miss_counts = {col: [] for col in self.merits}
            # ----------
            
            for index, row in data.iterrows():
                if self._budget_inst:
                    self.budget_manager.add_budget(self._bgain_inst)
                    
                #handle instance
                miss_feature_merits = self._get_miss_merits(inst=row, merits=self.merits)
                
                #acquisition strategy
                acq_merits, acq_costs = self.feature_selection.get_acquisition_feature_set(self, miss_feature_merits)
                
                # TEST STUFF                
                for col, merit in self.merits.items():
                    if self.isnan(row[col]): self.i_miss_counts[col].append(1)
                    else: self.i_miss_counts[col].append(0)
                    self.i_merits[col].append(merit)
                    if col in [merit[0] for merit in acq_merits]: self.i_queried[col].append(1)
                    else: self.i_queried[col].append(0)
                    self.i_answered[col].append(0)
                # ----------
                
                if any(acq_merits):
                    q_before = self._get_quality(inst=row, merits=self.merits, acq_merits=[])
                    q_after = self._get_quality(inst=row, merits=self.merits, acq_merits=[m[1] for m in acq_merits])
                    quality = q_after - q_before
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in self._get_keys(acq_merits, 0):     
                            self.queried[feature] += 1
                            
                            # TEST STUFF
                            self.i_answered[feature][-1] = 1
                            # ----------
                            
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
        pd_index = pd.MultiIndex.from_product([[const.missing_features, const.active_feature_acquisition_queries, const.active_feature_acquisition_answers, const.active_feature_acquisition_merits], [x for x in self.queried]])
        stats = pd.DataFrame(index=pd_index).T.copy()
        
        for col, count in self.i_miss_counts.items():
            stats[(const.missing_features, col)] = stats[(const.missing_features, col)].astype('object')
            stats.loc[index, (const.missing_features, col)] = str(count)
        for col, count in self.i_queried.items():
            stats[(const.active_feature_acquisition_queries, col)] = stats[(const.active_feature_acquisition_queries, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_queries, col)] = str(count)
        for col, count in self.i_answered.items():
            stats[(const.active_feature_acquisition_answers, col)] = stats[(const.active_feature_acquisition_answers, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_answers, col)] = str(count)
        for col, merit in self.i_merits.items():
            stats[(const.active_feature_acquisition_merits, col)] = stats[(const.active_feature_acquisition_merits, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_merits, col)] = str(merit)
        
        return stats
        
    def get_name(self):
        return "SWIG_H2+" + self.feature_selection.get_name()
        
class SWSU_H2(SingleWindowSU): #single instance decision tracking and one-sided harsher dynamic budget correction
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size,dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Copy of SingleWindowAED class modified to suit special presentation and or debug purposes
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
                #dynamic budget threshold
                i_exp = self.feature_selection.get_expected_total_inst_cost(self)
                exp_threshold = self._bgain_inst / i_exp
                used_budget_ratio = self.budget_manager.used_budget()
                new_threshold = exp_threshold / used_budget_ratio
                over_budget_rate = math.floor((used_budget_ratio - 1) / 0.0625) + 2
                if used_budget_ratio > 1: new_threshold /= over_budget_rate
                self.budget_manager.budget_threshold = min(new_threshold, 1)
                
                # print(i_exp, exp_threshold, new_threshold)
            
            #since window is batch based, reset temporary window to equal the batch window again
            self._on_new_batch(self.window.get_window_data())
            
            #calculate global feature merits, saved into class to allow statistics
            self.merits = self._get_merits(self._get_rank_values())
            
            # TEST STUFF
            self.i_queried = {col: [] for col in self.merits}
            self.i_answered = {col: [] for col in self.merits}
            self.i_merits = {col: [] for col in self.merits}
            self.i_miss_counts = {col: [] for col in self.merits}
            # ----------
            
            for index, row in data.iterrows():
                if self._budget_inst:
                    self.budget_manager.add_budget(self._bgain_inst)
                    
                #handle instance
                miss_feature_merits = self._get_miss_merits(inst=row, merits=self.merits)
                
                #acquisition strategy
                acq_merits, acq_costs = self.feature_selection.get_acquisition_feature_set(self, miss_feature_merits)
                
                # TEST STUFF                
                for col, merit in self.merits.items():
                    if self.isnan(row[col]): self.i_miss_counts[col].append(1)
                    else: self.i_miss_counts[col].append(0)
                    self.i_merits[col].append(merit)
                    if col in [merit[0] for merit in acq_merits]: self.i_queried[col].append(1)
                    else: self.i_queried[col].append(0)
                    self.i_answered[col].append(0)
                # ----------
                
                if any(acq_merits):
                    q_before = self._get_quality(inst=row, merits=self.merits, acq_merits=[])
                    q_after = self._get_quality(inst=row, merits=self.merits, acq_merits=[m[1] for m in acq_merits])
                    quality = q_after - q_before
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in self._get_keys(acq_merits, 0):     
                            self.queried[feature] += 1
                            
                            # TEST STUFF
                            self.i_answered[feature][-1] = 1
                            # ----------
                            
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
        pd_index = pd.MultiIndex.from_product([[const.missing_features, const.active_feature_acquisition_queries, const.active_feature_acquisition_answers, const.active_feature_acquisition_merits], [x for x in self.queried]])
        stats = pd.DataFrame(index=pd_index).T.copy()
        
        for col, count in self.i_miss_counts.items():
            stats[(const.missing_features, col)] = stats[(const.missing_features, col)].astype('object')
            stats.loc[index, (const.missing_features, col)] = str(count)
        for col, count in self.i_queried.items():
            stats[(const.active_feature_acquisition_queries, col)] = stats[(const.active_feature_acquisition_queries, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_queries, col)] = str(count)
        for col, count in self.i_answered.items():
            stats[(const.active_feature_acquisition_answers, col)] = stats[(const.active_feature_acquisition_answers, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_answers, col)] = str(count)
        for col, merit in self.i_merits.items():
            stats[(const.active_feature_acquisition_merits, col)] = stats[(const.active_feature_acquisition_merits, col)].astype('object')
            stats.loc[index, (const.active_feature_acquisition_merits, col)] = str(merit)
        
        return stats
        
    def get_name(self):
        return "SWSU_H2+" + self.feature_selection.get_name()
        
class SWAED_IMAX(SingleWindowAED): #no budgeting penalty and IMax budgeting
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size,dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Copy of SingleWindowAED class modified to suit special presentation and or debug purposes
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
        self.i_max = sum(sorted(self.acquisition_costs.values(), reverse=True)[:min(self.feature_selection.k, len(self.acquisition_costs))])

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
                #dynamic budget threshold
                #i_exp = self.feature_selection.get_expected_total_inst_cost(self)
                exp_threshold = self._bgain_inst / self.i_max
                used_budget_ratio = self.budget_manager.used_budget()
                new_threshold = exp_threshold / used_budget_ratio
                #over_budget_rate = math.floor((used_budget_ratio - 1) / 0.0625) + 2
                #if used_budget_ratio > 1: new_threshold /= over_budget_rate
                self.budget_manager.budget_threshold = min(new_threshold, 1)
                
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
                    quality = self._get_quality_gain(inst=row, merits=self.merits, acq_merits=[m[1] for m in acq_merits])                
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in self._get_keys(acq_merits, 0):     
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
        
    def get_name(self):
        return "SWAED_IMAX+" + self.feature_selection.get_name()

class SWAED_IEXP(SingleWindowAED): #no budgeting penalty and IMax budgeting
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size,dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Copy of SingleWindowAED class modified to suit special presentation and or debug purposes
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
                #dynamic budget threshold
                i_exp = self.feature_selection.get_expected_total_inst_cost(self)
                exp_threshold = self._bgain_inst / i_exp
                used_budget_ratio = self.budget_manager.used_budget()
                new_threshold = exp_threshold / used_budget_ratio
                #over_budget_rate = math.floor((used_budget_ratio - 1) / 0.0625) + 2
                #if used_budget_ratio > 1: new_threshold /= over_budget_rate
                self.budget_manager.budget_threshold = min(new_threshold, 1)
                
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
                    quality = self._get_quality_gain(inst=row, merits=self.merits, acq_merits=[m[1] for m in acq_merits])                
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in self._get_keys(acq_merits, 0):     
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
        
    def get_name(self):
        return "SWAED_IEXP+" + self.feature_selection.get_name()
        
