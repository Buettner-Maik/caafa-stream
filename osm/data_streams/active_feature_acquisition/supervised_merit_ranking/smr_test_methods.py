import pandas as pd
import numpy as np
import osm.data_streams.constants as const
import math

from osm.data_streams.active_feature_acquisition.supervised_merit_ranking.average_euclidean_distance import SingleWindowAED, SWAEDFeatureCorrelationCorrected
from osm.data_streams.active_feature_acquisition.supervised_merit_ranking.entropy_based import SingleWindowIG, SingleWindowSU
from osm.data_streams.active_feature_acquisition.random_acquisition import RandomFeatureAcquisition
from osm.data_streams.active_feature_acquisition.no_feature_acquisition import NoActiveFeatureAcquisition
from osm.data_streams.imputer.PairImputer import PairRegImputer, FeaturePairImputer
from osm.data_streams.budget_manager.incremental_percentile_filter import IncrementalPercentileFilter
from osm.data_streams.active_feature_acquisition.supervised_merit_ranking.classifier_feature_importance import PoolClassifierFeatureImportanceAFA

class NAFA_FPI(NoActiveFeatureAcquisition):
    """
    Tracker for Feature Pair Imputer Runs
    """
    def get_name(self):
        return "no_AFA_FPI"

class RA_FPI(RandomFeatureAcquisition):
    """
    Tracker for Feature Pair Imputer Runs
    """
    def get_name(self):
        return "RA_FPI"
class SWAED_FPI(SingleWindowAED):
    """
    Tracker for Feature Pair Imputer Runs
    """
    def get_name(self):
        return "SWAED_FPI+" + self.feature_selection.get_name()

class SWAED_NDB(SingleWindowAED):
    """
    No dynamic budgeting
    """
    def get_name(self):
        return "SWAED_NDB+" + self.feature_selection.get_name()

class SWAED_IMAX(SingleWindowAED):
    """
    Using I_max for dynamic budgeting without penalty
    """
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold=False, categories=None, acquisition_costs=..., budget_option=..., debug=True):
        super().__init__(window, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold, categories, acquisition_costs, budget_option, debug)
        self.i_max = sum(sorted(self.acquisition_costs.values(), reverse=True)[:min(self.feature_selection.k, len(self.acquisition_costs))])
    def get_dynamic_threshold(self, i_exp, penalty_step_size=1 / 32, penalty_gain=1):
        return super().get_dynamic_threshold(self.i_max, 1, 0)
    def get_name(self):
        return "SWAED_IMAX+" + self.feature_selection.get_name()

class SWAED_IEXP(SingleWindowAED):
    """
    Using I_exp for dynamic budgeting without penalty
    """
    def get_dynamic_threshold(self, i_exp, penalty_step_size=1 / 32, penalty_gain=1):
        return super().get_dynamic_threshold(i_exp, 1, 0)
    def get_name(self):
        return "SWAED_IEXP+" + self.feature_selection.get_name()

class SWAED_SSBQ(SingleWindowAED):
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Altered quality function with a set size bias
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
        self.q_setsize_mult = {}
        
        self.old_instances_processed = 0
        self.old_miss_sum = 0

        self.features = 0
        self.expected_acqs_per_inst = 0

    def _update_setsize_mult(self, target_set_size):
        if target_set_size < (self.features + 1) / 2:
            offset = self.features + 1 - (target_set_size * 2)
            self.q_setsize_mult = {i:(i + offset) * (self.features + 1 - i) for i in range(self.features + 1)}
        else:
            self.q_setsize_mult = {i:i * (target_set_size * 2 - i) for i in range(self.features + 1)}

    def _initialize(self, data):
        super()._initialize(data)

        self.batch_size = data.shape[0]
        self.features = len(self.cat_cols) + len(self.num_cols)
        self.expected_acqs_per_inst = self._bgain_inst / (sum(self.acquisition_costs.values()) / len(self.acquisition_costs))

        self.q_setsize_mult = {i:1 for i in range(self.features + 1)}

    def _on_new_batch(self, data):
        super()._on_new_batch(data)

        if self.instances_processed > 0:
            new_miss_sum = sum(self.miss_counts.values())

            batch_size = self.instances_processed - self.old_instances_processed
            total_features = self.features * batch_size
            missing_features = new_miss_sum - self.old_miss_sum
            average_known_set_size = (total_features - missing_features) / batch_size
            self._update_setsize_mult(average_known_set_size + self.expected_acqs_per_inst)

            self.old_instances_processed = self.instances_processed
            self.old_miss_sum = new_miss_sum

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
        return i_merit / known_features * self.q_setsize_mult[known_features]

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
        pre_quali = pre_merit / pre_known * self.q_setsize_mult[pre_known] if pre_known != 0 else 0
        
        pos_merit = sum(acq_merits.values()) + pre_merit
        pos_known = len(acq_merits) + pre_known
        pos_quali = pos_merit / pos_known * self.q_setsize_mult[pos_known]
        
        return pos_quali - pre_quali

    def get_name(self):
        return "SWAED_SSBQ+" + self.feature_selection.get_name()

class SWAED_SSBQ_FPI(SWAED_SSBQ):
    """
    Tracker for Feature Pair Imputer Runs
    """
    def get_name(self):
        return "SWAED_SSBQ_FPI+" + self.feature_selection.get_name()

class SWAED_IMPQ(SingleWindowAED):
    def __init__(self, window, imputer, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Altered quality function with feature specific imputer confidence
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
        if not isinstance(imputer, PairRegImputer) and not isinstance(imputer, FeaturePairImputer):
            raise ValueError(f'Imputer {imputer} is not of the PairRegImputer or FeaturePairImputer class')
        self.imputer = imputer
        self.q_setsize_mult = {}
        
        self.old_instances_processed = 0
        self.old_miss_sum = 0

        self.features = 0
        self.expected_acqs_per_inst = 0

    def _update_setsize_mult(self, target_set_size):
        dropoff = 5 / 6
        self.q_setsize_mult = {i:dropoff ** np.abs(i - target_set_size) for i in range(self.features + 1)}

        # if target_set_size < (self.features + 1) / 2:
        #     offset = self.features + 1 - (target_set_size * 2)
        #     self.q_setsize_mult = {i:(i + offset) * (self.features + 1 - i) for i in range(self.features + 1)}
        # else:
        #     self.q_setsize_mult = {i:i * (target_set_size * 2 - i) for i in range(self.features + 1)}

    def _initialize(self, data):
        super()._initialize(data)

        self.batch_size = data.shape[0]
        self.features = len(self.cat_cols) + len(self.num_cols)
        self.expected_acqs_per_inst = self._bgain_inst / (sum(self.acquisition_costs.values()) / len(self.acquisition_costs))

        self.q_setsize_mult = {i:1 for i in range(self.features + 1)}

    def _on_new_batch(self, data):
        super()._on_new_batch(data)

        if self.instances_processed > 0:
            new_miss_sum = sum(self.miss_counts.values())

            batch_size = self.instances_processed - self.old_instances_processed
            total_features = self.features * batch_size
            missing_features = new_miss_sum - self.old_miss_sum
            average_known_set_size = (total_features - missing_features) / batch_size
            self._update_setsize_mult(average_known_set_size + self.expected_acqs_per_inst)

            self.old_instances_processed = self.instances_processed
            self.old_miss_sum = new_miss_sum

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
        knowns = np.zeros((self.features, 1))
        for feature in self.cat_cols + self.num_cols:
            if not self.isnan(inst[feature]): knowns[self.imputer.f_name_to_index[feature], 0] = 1
        merit_sum = 0
        imp_sum = 0

        for feature, merit in acq_merits.items():
            knowns[self.imputer.f_name_to_index[feature], 0] = True
            merit_sum += merit
            imp_sum += 1

        nrmses = knowns * self.imputer.imputation_rmses

        for feature, merit in merits.items():
            if not self.isnan(inst[feature]):
                merit_sum += merit
                imp_sum += 1
            elif not feature in acq_merits:
                imp_sum += 1 - np.min(nrmses[:, self.imputer.f_name_to_index[feature]])

        return merit_sum * imp_sum * self.q_setsize_mult[knowns.sum()]

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
        pre_knowns = np.zeros((self.features, 1))
        for feature in self.cat_cols + self.num_cols:
            if not self.isnan(inst[feature]): pre_knowns[self.imputer.f_name_to_index[feature], 0] = 1
        pos_knowns = pre_knowns.copy()
        pre_merit_sum = 0
        pos_merit_sum = 0
        pre_imp_sum = 0
        pos_imp_sum = 0

        for key, item in acq_merits.items():
            pos_knowns[self.imputer.f_name_to_index[key], 0] = True
            pos_merit_sum += item
            pos_imp_sum += 1

        pre_nrmses = pre_knowns * self.imputer.imputation_rmses
        pos_nrmses = pos_knowns * self.imputer.imputation_rmses

        for key, item in merits.items():
            if not self.isnan(inst[key]):
                pre_merit_sum += item
                pos_merit_sum += item
                pre_imp_sum += 1
                pos_imp_sum += 1
            else:
                pre_imp_sum += 1 - np.min(pos_nrmses[:, self.imputer.f_name_to_index[key]])
                if not key in acq_merits:
                    pos_imp_sum += 1 - np.min(pre_nrmses[:, self.imputer.f_name_to_index[key]])

        pre_quality = pre_merit_sum * pre_imp_sum * self.q_setsize_mult[pre_knowns.sum()]
        pos_quality = pos_merit_sum * pos_imp_sum * self.q_setsize_mult[pos_knowns.sum()]

        #return self._get_quality(inst, merits, acq_merits) - self._get_quality(inst, merits, {})
        return pos_quality - pre_quality

    def get_name(self):
        return "SWAED_IMPQ+" + self.feature_selection.get_name()

class SWAED_IMPQ2(SingleWindowAED):
    def __init__(self, window, imputer, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Altered quality function with feature specific imputer confidence
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
        if not isinstance(imputer, PairRegImputer) and not isinstance(imputer, FeaturePairImputer):
            raise ValueError(f'Imputer {imputer} is not of the PairRegImputer or FeaturePairImputer class')
        self.imputer = imputer
        self.q_setsize_cost_mult = {}
        
        self.old_instances_processed = 0
        self.old_miss_sum = 0

        self.features = 0
        self.total_feature_costs = sum(self.acquisition_costs.values())
        self.average_feature_cost = self.total_feature_costs / len(self.acquisition_costs)
        self.expected_acqs_per_inst = 0

    def _update_setsize_cost_mult(self, target_set_size_cost):
        dropoff = (self.total_feature_costs * 0.875) / self.total_feature_costs
        self.q_setsize_cost_mult = {i:dropoff ** np.abs(i - target_set_size_cost) for i in range(self.features + 1)}
        #dropoff = 5 / 6
        #self.q_setsize_mult = {i:dropoff ** np.abs(i - target_set_size_cost) for i in range(self.features + 1)}

        # if target_set_size < (self.features + 1) / 2:
        #     offset = self.features + 1 - (target_set_size * 2)
        #     self.q_setsize_mult = {i:(i + offset) * (self.features + 1 - i) for i in range(self.features + 1)}
        # else:
        #     self.q_setsize_mult = {i:i * (target_set_size * 2 - i) for i in range(self.features + 1)}

    def _initialize(self, data):
        super()._initialize(data)

        self.batch_size = data.shape[0]
        self.features = len(self.cat_cols) + len(self.num_cols)
        self.expected_acqs_per_inst = self._bgain_inst / (sum(self.acquisition_costs.values()) / len(self.acquisition_costs))

        self.q_setsize_cost_mult = {i:1 for i in range(self.features + 1)}

    def _on_new_batch(self, data):
        super()._on_new_batch(data)

        if self.instances_processed > 0:
            new_miss_sum = sum(self.miss_counts.values())

            batch_size = self.instances_processed - self.old_instances_processed
            total_features = self.features * batch_size
            missing_features = new_miss_sum - self.old_miss_sum
            average_known_set_size = (total_features - missing_features) / batch_size
            self._update_setsize_cost_mult((average_known_set_size + self.expected_acqs_per_inst) * self.average_feature_cost)

            self.old_instances_processed = self.instances_processed
            self.old_miss_sum = new_miss_sum

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
        knowns = np.zeros((self.features, 1))
        for feature in self.cat_cols + self.num_cols:
            if not self.isnan(inst[feature]): knowns[self.imputer.f_name_to_index[feature], 0] = 1
        merit_sum = 0
        imp_sum = 0
        cost_sum = 0

        for feature, merit in acq_merits.items():
            knowns[self.imputer.f_name_to_index[feature], 0] = True
            merit_sum += merit
            imp_sum += 1
            cost_sum += self.acquisition_costs[feature]

        nrmses = knowns * self.imputer.imputation_rmses

        for feature, merit in merits.items():
            if not self.isnan(inst[feature]):
                merit_sum += merit
                imp_sum += 1
                cost_sum += self.acquisition_costs[feature]
            elif not feature in acq_merits:
                imp_sum += 1 - np.min(nrmses[:, self.imputer.f_name_to_index[feature]])

        return merit_sum * imp_sum * self.q_setsize_cost_mult[cost_sum]

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
        pre_knowns = np.zeros((self.features, 1))
        for feature in self.cat_cols + self.num_cols:
            if not self.isnan(inst[feature]): pre_knowns[self.imputer.f_name_to_index[feature], 0] = 1
        pos_knowns = pre_knowns.copy()
        pre_merit_sum = 0
        pos_merit_sum = 0
        pre_imp_sum = 0
        pos_imp_sum = 0
        pre_cost_sum = 0
        pos_cost_sum = 0

        for key, item in acq_merits.items():
            pos_knowns[self.imputer.f_name_to_index[key], 0] = True
            pos_merit_sum += item
            pos_imp_sum += 1
            pos_cost_sum += self.acquisition_costs[key]

        pre_nrmses = pre_knowns * self.imputer.imputation_rmses
        pos_nrmses = pos_knowns * self.imputer.imputation_rmses

        for key, item in merits.items():
            if not self.isnan(inst[key]):
                pre_merit_sum += item
                pos_merit_sum += item
                pre_imp_sum += 1
                pos_imp_sum += 1
                pre_cost_sum += self.acquisition_costs[key]
                pos_cost_sum += self.acquisition_costs[key]
            else:
                pre_imp_sum += 1 - np.min(pos_nrmses[:, self.imputer.f_name_to_index[key]])
                if not key in acq_merits:
                    pos_imp_sum += 1 - np.min(pre_nrmses[:, self.imputer.f_name_to_index[key]])

        pre_quality = pre_merit_sum * pre_imp_sum * self.q_setsize_cost_mult[pre_cost_sum]
        pos_quality = pos_merit_sum * pos_imp_sum * self.q_setsize_cost_mult[pos_cost_sum]

        #return self._get_quality(inst, merits, acq_merits) - self._get_quality(inst, merits, {})
        return pos_quality - pre_quality

    def get_name(self):
        return "SWAED_IMPQ2+" + self.feature_selection.get_name()

class SWAED_IMPTS(SingleWindowAED):
    # threshold values for
    # -> magic
    #   -> mean: 0.261645072
    #   ->  var: 0.001581752
    #   ->  min: 0.141492352
    #   ->  max: 0.375063275

    def __init__(self, window, imputer, target_col_name, budget_manager, feature_selection, acq_set_size, skip_percentile, max_thr=10, dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Tracks the expected regression errors and skips the acquisition process for an instance if
        the imputer is confident enough to predict the values on its own
        An imputer is deemed confident if all missing values can be imputed with pair-models that
        have NRMSES are below the threshold
        The selection process of these models varies depending on the importance method of the
        imputer
        Simple models (i->i) are ignored
        :param imputer: The feature pair imputer providing rmses
        :param skip_percentile: How many acquisitions are skipped based on the imputer's ability
                                to impute correctly
        :param max_thr: The imputer's minimum confidence to skip an instance
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
        self.imputer = imputer
        #self.max_thr = max_thr
        self.ipf = IncrementalPercentileFilter(skip_percentile, 100)
    
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

        e = self.imputer.imputation_rmses
        errors = e[known, :][:, unknown]

        if self.imputer.include_simple:
            simples = np.diag(e)[unknown]
            errors = np.concatenate((errors, np.expand_dims(simples, axis=0)), axis=0)
        if self.imputer.importance == 'best':
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
                    
                if not self.skip_acquisition(row[self.cat_cols + self.num_cols]):
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

    def get_name(self):
        return "SWAED_IMPCS+" + self.feature_selection.get_name()

class SWAED_IMPTS_2(SWAED_IMPTS):
    def get_name(self):
        return "SWAED_IMPTS_2+" + self.feature_selection.get_name()
class SWAED_IMPTS_6(SWAED_IMPTS):
    def get_name(self):
        return "SWAED_IMPTS_6+" + self.feature_selection.get_name()
class SWAED_IMPTS_10(SWAED_IMPTS):
    def get_name(self):
        return "SWAED_IMPTS_10+" + self.feature_selection.get_name()

class PCFI_DTC(PoolClassifierFeatureImportanceAFA):
    def get_name(self):
        return "PCFI_DTC+" + self.feature_selection.get_name()


class SWAED_QT(SingleWindowAED):
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Altered quality function with feature specific imputer confidence
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
        self.q_setsize_mult = {}
        
        self.old_instances_processed = 0
        self.old_miss_sum = 0

        self.features = 0
        self.expected_acqs_per_inst = 0

    def _update_setsize_mult(self, target_set_size):
        if target_set_size < (self.features + 1) / 2:
            offset = self.features + 1 - (target_set_size * 2)
            self.q_setsize_mult = {i:(i + offset) * (self.features + 1 - i) for i in range(self.features + 1)}
        else:
            self.q_setsize_mult = {i:i * (target_set_size * 2 - i) for i in range(self.features + 1)}

    def _initialize(self, data):
        super()._initialize(data)

        self.batch_size = data.shape[0]
        self.features = len(self.cat_cols) + len(self.num_cols)
        self.expected_acqs_per_inst = sum(self.acquisition_costs.values()) / len(self.acquisition_costs) / self._bgain_inst

        self._update_setsize_mult(self.features / 2)

    def _on_new_batch(self, data):
        super()._on_new_batch(data)

        if self.instances_processed > 0:
            new_miss_sum = sum(self.miss_counts.values())

            batch_size = self.instances_processed - self.old_instances_processed
            total_features = self.features * batch_size
            missing_features = new_miss_sum - self.old_miss_sum
            average_known_set_size = (total_features - missing_features) / batch_size
            self._update_setsize_mult(average_known_set_size + self.expected_acqs_per_inst)

            self.old_instances_processed = self.instances_processed
            self.old_miss_sum = new_miss_sum

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
        return i_merit / known_features * self.q_setsize_mult[known_features]

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
        pre_quali = pre_merit / pre_known * self.q_setsize_mult[pre_known] if pre_known != 0 else 0
        
        pos_merit = sum(acq_merits.values()) + pre_merit
        pos_known = len(acq_merits) + pre_known
        pos_quali = pos_merit / pos_known * self.q_setsize_mult[pos_known]
        
        return pos_quali - pre_quali

    def get_name(self):
        return "SWAED_QT+" + self.feature_selection.get_name()

class SWAED_IQ(SingleWindowAED):
    def __init__(self, window, imputer, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Altered quality function with feature specific imputer confidence
        """
        raise NotImplementedError()
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
        self.imputer = imputer

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

        known = ~np.isnan(inst)
        a_merits = np.array([np.nan]*self.imputer.num_features)
        for feature in acq_merits:
            known[self.imputer.f_name_to_index[feature]] = 1
        if known.any():
            for feature, merit in merits.items():
                a_merits[self.imputer.f_name_to_index[feature]] = merit
        else:
            return 0

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
        
        
        if known.any():
            e = self.imputer.imputation_rmses
            nrmse = e[known == 1, :][:, known == 0].min(axis=0)
            np.place(known, known == 0, nrmse)

            for feature in miss_feature_merits:
                miss_feature_merits[feature] *= known[self.imputer.f_name_to_index[feature]]
        
        

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

    def get_name(self):
        return "SWAED_IQ+" + self.feature_selection.get_name()

class SWAED_AQ(SingleWindowAED): #Acquisition quality function
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size,dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Copy of SingleWindowAED class modified to only give the quality of the acquisition set to the budget manager
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
    def _get_quality_gain(self, inst, merits, acq_merits=[]):
        return sum(acq_merits) / len(acq_merits) if any(acq_merits) else 0

    def get_name(self):
        return "SWAED_AQ+" + self.feature_selection.get_name()

class SWAED_TRPRI(SingleWindowAED):
    # threshold values for
    # -> magic
    #   -> mean: 0.261645072
    #   ->  var: 0.001581752
    #   ->  min: 0.141492352
    #   ->  max: 0.375063275

    def __init__(self, window, pipeline, imp_conf_stddev, target_col_name, budget_manager, feature_selection, acq_set_size,dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Copy of SingleWindowAED class modified to only give the quality of the acquisition set to the budget manager
        Tracks the expected regression errors 
        :param pipeline: The pipeline containing a PairRegImputer
        :param imp_conf_stddev: The stddev added to the rmse list serving as a threshold,
                                acts as a confidence level where 0 is equivalent to
                                skipping half the acquisitions to leave for the imputer
                                Negative values increase the necessary confidence to
                                skip the acquisition for a particular instance
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
        self.pipeline = pipeline
        self.imp_thr = 0
        self.imp_conf_stddev = imp_conf_stddev
        self.imp_thr_stddev = 0
        self.imp_thr_mean = 0
        self.imp_eerror_list = []

    # def get_pair_reg_imputer(self):
    #     return self.pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['imputer']
    
    def imputation_confident(self, x):
        e = self.pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['imputer'].imputation_rmses
        means = e[~np.isnan(x), :][:, np.isnan(x)].mean(axis=0)
        if means.any():
            eerror = means.max()
            self.imp_eerror_list.append(eerror)
            return eerror < self.imp_thr
        else: #no missing features thus skip
            return True

    def update_imp_thr(self):
        self.imp_thr_mean = np.mean(self.imp_eerror_list)
        self.imp_thr_stddev = np.sqrt(np.var(self.imp_eerror_list))
        self.imp_thr = self.imp_thr_mean + self.imp_conf_stddev * self.imp_thr_stddev
        self.imp_eerror_list = []

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
                self.update_imp_thr()
            
            #since window is batch based, reset temporary window to equal the batch window again
            self._on_new_batch(self.window.get_window_data())
            
            #calculate global feature merits, saved into class to allow statistics
            self.merits = self._get_merits(self._get_rank_values())
            
            for index, row in data.iterrows():
                if self._budget_inst:
                    self.budget_manager.add_budget(self._bgain_inst)
                    
                if self.imputation_confident(row[self.num_cols]):
                    continue

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

    def get_name(self):
        return "SWAED_TRPRI+" + self.feature_selection.get_name()

class SWAED_TRPRI_0(SWAED_TRPRI):
    def __init__(self, window, pipeline, imp_conf_stddev, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        super().__init__(window, pipeline, imp_conf_stddev, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold, categories, acquisition_costs, budget_option, debug)
    def get_name(self):
        return "SWAED_TRPRI_0+" + self.feature_selection.get_name()
class SWAED_TRPRI_25(SWAED_TRPRI):
    def __init__(self, window, pipeline, imp_conf_stddev, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        super().__init__(window, pipeline, imp_conf_stddev, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold, categories, acquisition_costs, budget_option, debug)
    def get_name(self):
        return "SWAED_TRPRI_25+" + self.feature_selection.get_name()
class SWAED_TRPRI_50(SWAED_TRPRI):
    def __init__(self, window, pipeline, imp_conf_stddev, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        super().__init__(window, pipeline, imp_conf_stddev, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold, categories, acquisition_costs, budget_option, debug)
    def get_name(self):
        return "SWAED_TRPRI_50+" + self.feature_selection.get_name()
class SWAED_TRPRI_75(SWAED_TRPRI):
    def __init__(self, window, pipeline, imp_conf_stddev, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        super().__init__(window, pipeline, imp_conf_stddev, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold, categories, acquisition_costs, budget_option, debug)
    def get_name(self):
        return "SWAED_TRPRI_75+" + self.feature_selection.get_name()
class SWAED_TRPRI_100(SWAED_TRPRI):
    def __init__(self, window, pipeline, imp_conf_stddev, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        super().__init__(window, pipeline, imp_conf_stddev, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold, categories, acquisition_costs, budget_option, debug)
    def get_name(self):
        return "SWAED_TRPRI_100+" + self.feature_selection.get_name()
class SWAED_TRPRI_200(SWAED_TRPRI):
    def __init__(self, window, pipeline, imp_conf_stddev, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        super().__init__(window, pipeline, imp_conf_stddev, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold, categories, acquisition_costs, budget_option, debug)
    def get_name(self):
        return "SWAED_TRPRI_200+" + self.feature_selection.get_name()

class SWAED_RPRI(SingleWindowAED): #reciprocal pair-reg-imputer
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size,dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Copy of SingleWindowAED class modified to only give the quality of the acquisition set to the budget manager
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

    def get_name(self):
        return "SWAED_RPRI+" + self.feature_selection.get_name()

class SWAED_SPRI(SingleWindowAED):
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size,dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Copy of SingleWindowAED class modified to only give the quality of the acquisition set to the budget manager
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

    def get_name(self):
        return "SWAED_SPRI+" + self.feature_selection.get_name()

class SWAED_BPRI(SingleWindowAED): #reciprocal pair-reg-imputer
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size,dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Copy of SingleWindowAED class modified to only give the quality of the acquisition set to the budget manager
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

    def get_name(self):
        return "SWAED_BPRI+" + self.feature_selection.get_name()

class SWAEDFCC_RPRI(SingleWindowAED):
    def __init__(self, window, imputer, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
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
        self.imputer = imputer
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

    def _on_new_batch(self, data):
        super()._on_new_batch(data)
        self.correlations = 1 - np.abs(self.imputer.get_correlation_matrix())

    def get_name(self):
        return "SWAEDFCC_RPRI+" + self.feature_selection.get_name()

class SWAEDFCC_SPRI(SingleWindowAED):
    def __init__(self, window, imputer, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
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
        self.imputer = imputer
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

    def _on_new_batch(self, data):
        super()._on_new_batch(data)
        self.correlations = 1 - np.abs(self.imputer.get_correlation_matrix())

    def get_name(self):
        return "SWAEDFCC_SPRI+" + self.feature_selection.get_name()

class SWAED_II(SingleWindowAED): #New name for iterative Imputer pipeline
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size,dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Copy of SingleWindowAED class modified to only give the quality of the acquisition set to the budget manager
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
    
    def get_name(self):
        return "SWAED_II+" + self.feature_selection.get_name()

class SWAEDFCC_II(SWAEDFeatureCorrelationCorrected): #New name for iterative Imputer pipeline
    def __init__(self, window, pipeline, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold=False, categories=None, acquisition_costs=..., budget_option=..., debug=True):
        super().__init__(window, pipeline, target_col_name, budget_manager, feature_selection, acq_set_size, dynamic_budget_threshold, categories, acquisition_costs, budget_option, debug)
    
    def get_name(self):
        return "SWAEDFCC_II+" + self.feature_selection.get_name()

class SWAED_LOGI(SingleWindowAED):
    def __init__(self, window, target_col_name, budget_manager, feature_selection, acq_set_size,dynamic_budget_threshold=False, categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Copy of SingleWindowAED class modified to only give the quality of the acquisition set to the budget manager
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
    
    def get_name(self):
        return "SWAED_LOGI+" + self.feature_selection.get_name()

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
                    quality = self._get_quality(inst=row, merits=self.merits, acq_merits=acq_merits)
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in acq_merits:
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
                    q_before = self._get_quality(inst=row, merits=self.merits, acq_merits={})
                    q_after = self._get_quality(inst=row, merits=self.merits, acq_merits=acq_merits)
                    quality = q_after - q_before
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in acq_merits:
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
                    q_before = self._get_quality(inst=row, merits=self.merits, acq_merits={})
                    q_after = self._get_quality(inst=row, merits=self.merits, acq_merits=acq_merits)
                    quality = q_after - q_before
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in acq_merits:
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
                    q_before = self._get_quality(inst=row, merits=self.merits, acq_merits={})
                    q_after = self._get_quality(inst=row, merits=self.merits, acq_merits=acq_merits)
                    quality = q_after - q_before
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in acq_merits:
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
                    q_before = self._get_quality(inst=row, merits=self.merits, acq_merits={})
                    q_after = self._get_quality(inst=row, merits=self.merits, acq_merits=acq_merits)
                    quality = q_after - q_before
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in acq_merits:
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
                    q_before = self._get_quality(inst=row, merits=self.merits, acq_merits={})
                    q_after = self._get_quality(inst=row, merits=self.merits, acq_merits=acq_merits)
                    quality = q_after - q_before
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in acq_merits:
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
                    q_before = self._get_quality(inst=row, merits=self.merits, acq_merits={})
                    q_after = self._get_quality(inst=row, merits=self.merits, acq_merits=acq_merits)
                    quality = q_after - q_before
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in acq_merits:
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
                    q_before = self._get_quality(inst=row, merits=self.merits, acq_merits={})
                    q_after = self._get_quality(inst=row, merits=self.merits, acq_merits=acq_merits)
                    quality = q_after - q_before
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in acq_merits:
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
                    q_before = self._get_quality(inst=row, merits=self.merits, acq_merits={})
                    q_after = self._get_quality(inst=row, merits=self.merits, acq_merits=acq_merits)
                    quality = q_after - q_before
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in acq_merits:
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
                    q_before = self._get_quality(inst=row, merits=self.merits, acq_merits={})
                    q_after = self._get_quality(inst=row, merits=self.merits, acq_merits=acq_merits)
                    quality = q_after - q_before
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in acq_merits:
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
                    q_before = self._get_quality(inst=row, merits=self.merits, acq_merits={})
                    q_after = self._get_quality(inst=row, merits=self.merits, acq_merits=acq_merits)
                    quality = q_after - q_before
                
                    if self._budget_acq:
                        self.budget_manager.add_budget(self._bgain_acq)
                    
                    if self.budget_manager.acquire(quality, acq_costs):
                        for feature in acq_merits:
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
