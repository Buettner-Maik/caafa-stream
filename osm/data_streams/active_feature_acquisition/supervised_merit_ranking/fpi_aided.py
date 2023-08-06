import pandas as pd
import numpy as np
import osm.data_streams.constants as const
import math
import datetime

from osm.data_streams.active_feature_acquisition.supervised_merit_ranking.average_euclidean_distance import SingleWindowAED
from osm.data_streams.imputer.PairImputer import PairRegImputer, FeaturePairImputer
from osm.data_streams.budget_manager.incremental_percentile_filter import IncrementalPercentileFilter

class SWAED_FPITS(SingleWindowAED):

    def __init__(self, window, fpi, target_col_name, budget_manager, feature_selection, acq_set_size, skip_percentile, dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Tracks the expected regression errors and skips the acquisition process for an instance if
        the imputer is confident enough to predict the values on its own
        An imputer is deemed confident if all missing values can be imputed with pair-models that
        have NRMSES are below the threshold
        The selection process of these models varies depending on the importance method of the
        imputer
        Simple models (i->i) are ignored
        :param fpi: The feature pair imputer providing rmses
        :param skip_percentile: How many acquisitions are skipped based on the imputer's ability
                                to impute correctly
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
        self.fpi = fpi
        self.skip_percentile = skip_percentile
        self.ipf = IncrementalPercentileFilter(self.skip_percentile, 100)
        self.budget_saved = 0
        self.acqs_skipped = 0
    
    def skip_acquisition(self, x):
        """
        skip the skip_percentile best imputable instances
        """
        known = pd.notna(x)
        unknown = pd.isna(x)
        if known.sum() == 0:
            return self.ipf.acquire(0.0)
        if unknown.sum() == 0:
            return self.ipf.acquire(1.0)

        e = self.fpi.imputation_rmses
        errors = e[known, :][:, unknown]

        if self.fpi.include_simple:
            simples = np.diag(e)[unknown]
            errors = np.concatenate((errors, np.expand_dims(simples, axis=0)), axis=0)
        if self.fpi.importance == 'best':
            errors = errors.min(axis=0)
        else:
            errors = errors.mean(axis=0)

        conf = 1 - errors.max()
        return self.ipf.acquire(conf) #and conf < self.max_thr

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
                    
                    fpi_skipped = self.skip_acquisition(row[self.cat_cols + self.num_cols])
                    ipf_acquire = self.budget_manager.acquire(quality, acq_costs)
                    if not fpi_skipped and ipf_acquire:
                        for feature in acq_merits:
                            self.queried[feature] += 1
                            #get feature and replace current inst in iteration and data
                            feature_val = self.get_feature(inst=row, column=feature)
                            row[feature] = feature_val
                            data.loc[[index], [feature]] = feature_val
                    elif fpi_skipped and ipf_acquire:
                        self.budget_saved += acq_costs
                        self.acqs_skipped += 1
                
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
        pd_index.insert(0, (const.feature_pair_imputer_stats, const.acquisitions_skipped))
        pd_index.insert(0, (const.feature_pair_imputer_stats, const.budget_saved))
        stats = pd.DataFrame(index=pd_index).T.copy()
        
        for col, count in self.miss_counts.items():
            stats.loc[index, (const.missing_features, col)] = count
        for col, count in self.queried.items():
            stats.loc[index, (const.active_feature_acquisition_queries, col)] = count        
        for col, merit in self.merits.items():
            stats.loc[index, (const.active_feature_acquisition_merits, col)] = merit
        stats.loc[index, (const.feature_pair_imputer_stats, const.acquisitions_skipped)] = self.acqs_skipped
        stats.loc[index, (const.feature_pair_imputer_stats, const.budget_saved)] = self.budget_saved

        return stats

    def get_name(self):
        return "SWAED_FPITS_" + str(self.skip_percentile) + "+" + self.feature_selection.get_name()

class SWAED_FPITS_LOG(SWAED_FPITS):
    def Now(): return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    def features(self):
        row.notna().astype(int)[:10]
    
    def __init__(self, window, fpi, target_col_name, budget_manager, feature_selection, acq_set_size, skip_percentile, dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        More verbose version of SWAED_FPITS that logs additional info
        See SWAED_FPITS for explanation of functionality
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
                            fpi=fpi,
                            skip_percentile=skip_percentile,
                            debug=debug)
        self.features = None
        self.logfilepath = f"logs/{SWAED_FPITS_LOG.Now()}_{self.get_name()}.csv"
        self.csvsep = '\t'
        
    def _write_log_header(self):
        # Nr BatchProcessingStartTime BatchBudgetThreshold 
        
        header = ["Nr", "InstID", "IPFBudgetThreshold", "BudgetReceived", "Class"] + [f"{feature}_known" for feature in self.features] + [f"{feature}_merit" for feature in self.features] + [f"{feature}_acq" for feature in self.features] + ["AcqSetCost", "FPIConfidence", "QualityGain", "FPIDecision", "IPFDecision"]
            
        logfile = open(self.logfilepath, "w")
        
        logfile.write(header[0])
        for col in header[1:]:
            logfile.write(f"{self.csvsep}{col}")
        logfile.close()
    
    def _initialize(self, data):
        super()._initialize(data)
        self.features = self.cat_cols + self.num_cols
        self._write_log_header()
    
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
            logfile = open(self.logfilepath, "a")
            if self._budget_batch:
                self.budget_manager.add_budget(self._bgain_batch)
            
            if self.dynamic_budget_threshold and self.instances_processed > 0:
                self.budget_manager.budget_threshold = self.get_dynamic_threshold(self.feature_selection.get_expected_total_inst_cost(self))
            
            #since window is batch based, reset temporary window to equal the batch window again
            self._on_new_batch(self.window.get_window_data())
            
            #calculate global feature merits, saved into class to allow statistics
            self.merits = self._get_merits(self._get_rank_values())
            
            for index, row in data.iterrows():
                if self._budget_inst:
                    self.budget_manager.add_budget(self._bgain_inst)
                
                logfile.write(f"\n{self.instances_processed}{self.csvsep}{index}{self.csvsep}{self.budget_manager.budget_threshold}{self.csvsep}{self.budget_manager.budget}{self.csvsep}{row[self.target_col_name]}")
                for k in row.notna().astype(int)[:len(self.features)]:
                    logfile.write(f"{self.csvsep}{k}")
                for feature in self.features:
                    logfile.write(f"{self.csvsep}{self.merits[feature]}")
                
                #handle instance
                #miss_feature_merits = {feature: merit}
                miss_feature_merits = self._get_miss_merits(inst=row, merits=self.merits)
                
                #acquisition strategy
                acq_merits, acq_costs = self.feature_selection.get_acquisition_feature_set(self, miss_feature_merits)
                for feature in self.features:
                    if feature in acq_merits:
                        logfile.write(f"{self.csvsep}1")
                    else:
                        logfile.write(f"{self.csvsep}0")
                logfile.write(f"{self.csvsep}{acq_costs}")
                
                if any(acq_merits):
                    quality = self._get_quality_gain(inst=row, merits=self.merits, acq_merits=acq_merits)
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    fpi_skipped = self.skip_acquisition(row[self.features])
                    ipf_acquire = self.budget_manager.acquire(quality, acq_costs)
                    logfile.write(f"{self.csvsep}{self.ipf.values_list[(self.ipf.counter - 1) % self.ipf.window_size]}{self.csvsep}{quality}{self.csvsep}{int(fpi_skipped)}{self.csvsep}{int(ipf_acquire)}")
                    
                    if not fpi_skipped and ipf_acquire:
                        for feature in acq_merits:
                            self.queried[feature] += 1
                            #get feature and replace current inst in iteration and data
                            feature_val = self.get_feature(inst=row, column=feature)
                            row[feature] = feature_val
                            data.loc[[index], [feature]] = feature_val
                    elif fpi_skipped and ipf_acquire:
                        self.budget_saved += acq_costs
                        self.acqs_skipped += 1
                else:
                    logfile.write(f"{self.csvsep}{self.csvsep}{self.csvsep}{self.csvsep}")
                    # TODO: fill in commas for CSV log
                
                if self.label_known(inst=row):
                    #update window
                    self._update_window(inst=row)
                    self.merits = self._get_merits(self._get_rank_values())
                    
                self.instances_processed += 1
            
            logfile.close()
        else:
            raise ValueError("A pandas.DataFrame expected")

        
        return data
    
        # tracks batch: new budget threshold
        # tracks inst : budget, known / unknown features, merits, selectedFeatureSet, set cost, FPI confidence, FPI decision, qualityGain, IPF decision, +I-F budget spent / +I+F budget saved / -I+F budget savedenied / budget denied -I-F, 
    
    def get_name(self):
        return "SWAED_FPITS_LOG_" + str(self.skip_percentile) + "+" + self.feature_selection.get_name()
    
