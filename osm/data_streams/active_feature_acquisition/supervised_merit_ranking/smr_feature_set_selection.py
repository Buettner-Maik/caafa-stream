from abc import ABC, abstractmethod

from osm.data_streams.abstract_base_class import AbstractBaseClass
import operator
import random
from numpy import nan


class AbstractSMRFeatureSetSelection(AbstractBaseClass):
    def __init__(self, k):
        """
        :param k: the maximum feature set size to be returned
        """
        super().__init__()
        self.set_cost_total = 0
        self.k = k
        
    def get_expected_total_inst_cost(self, smr):
        """
        gets the expected cost for a total purchase of an average instance according to the selection strategy and miss statistics of the features
        :param smr: the parent supervised merit ranking active feature acquisition strategy this feature selection is used in
        """
        return self.set_cost_total / smr.instances_processed
        
    # def get_feature_set(self, context):
        # """
        # avoid usage, abstract SMRFSS are explicitly designed to make use of merit information
        # redirects to the get_acquisition_feature_set method
        # """
        # return get_acquisition_feature_set(context, None)
    
    @abstractmethod
    def get_acquisition_feature_set(self, smr, miss_feature_merits):
        """
        gets the feature set to be acquired according to some selection strategy and its corresponding total cost
        :param smr: the parent supervised merit ranking active feature acquisition strategy this feature selection is used in
        :param miss_feature_merits: the set of missing features and their merits of an instance in form of a dictionary
        returns a tuple of a list of (feature, merit) structures and the total cost of the set
        """
        pass
        
class KBestSMRFSS(AbstractSMRFeatureSetSelection):
    def __init__(self, k):
        """
        selects a set of feature acquisitions based on the k best missing merits
        :param k: how many acquisitions to add to the set at maximum
        """
        super().__init__(k=k)
        
    def get_acquisition_feature_set(self, smr, miss_feature_merits):
        """
        """
        acq_set = sorted(miss_feature_merits.items(), key=operator.itemgetter(1), reverse=True)[:self.k]
        acq_cost = sum([smr.acquisition_costs[feature] for feature in [a[0] for a in acq_set]])
        self.set_cost_total += acq_cost
        return acq_set, acq_cost
        
    def get_name(self):
        return "KBSMRFSS"

class KPickyBestSMRFSS(AbstractSMRFeatureSetSelection):        
    def __init__(self, k, rank=None):
        """
        selects a set of feature acquisitions based on the k best missing merits whose merit is equal or better than the specified merit rank at the time of acquisition
        :param k: how many acquisitions to add to the set at maximum
        :param rank: the maximum rank of a merit to be permitted as a consideration for the set, defaults to k
        """
        super().__init__(k=k)
        self.rank = rank if rank != None else k

    def get_acquisition_feature_set(self, smr, miss_feature_merits):
        """
        """
        min_merit = sorted(smr.merits.items(), key=operator.itemgetter(1), reverse=True)[min(self.rank, len(smr.merits.keys())) - 1][1]
        pot_feats = sorted(miss_feature_merits.items(), key=operator.itemgetter(1), reverse=True)
        acq_set = [[f, m] for f, m in pot_feats if m >= min_merit][:self.k]
        acq_cost = sum([smr.acquisition_costs[feature] for feature in [a[0] for a in acq_set]])
        self.set_cost_total += acq_cost
        return acq_set, acq_cost
        
    def get_name(self):
        return "KPBSMRFSS"

class KRandomSMRFSS(AbstractSMRFeatureSetSelection):
    def __init__(self, k):
        """
        selects a random set of maximum k missing features
        :param k: how many acquisitions to add to the set at maximum
        """
        super().__init__(k=k)
        
    def get_acquisition_feature_set(self, smr, miss_feature_merits):
        """
        """
        acq_set = random.sample(miss_feature_merits.items(), min(self.k, len(miss_feature_merits)))
        acq_cost = sum([smr.acquisition_costs[feature] for feature in [a[0] for a in acq_set]])
        self.set_cost_total += acq_cost
        return acq_set, acq_cost
        
    def get_name(self):
        return "KRSMRFSS"
        
class KQualitySMRFSS(AbstractSMRFeatureSetSelection):
    def __init__(self, k):
        """
        selects a set of feature acquisitions based on a greedy search of the highest quality per iterative acquisition of a feature until k features are selected
        :param k: how many acquisitions to add to the set at maximum
        """
        super().__init__(k=k)
        
    def get_acquisition_feature_set(self, smr, miss_feature_merits):
        """
        """
        inst = {}
        for feature in smr.merits:
            inst[feature] = nan if feature in miss_feature_merits else 0
        rem_set = list(miss_feature_merits.keys())
        acq_set = {}
        
        max_iter = min(self.k, len(rem_set))
        for i in range(max_iter):
            qualities = {feature:smr._get_quality(inst, merits=smr.merits, acq_merits=[miss_feature_merits[feature]]) for feature in rem_set}
            best_feature = sorted(qualities.items(), key=operator.itemgetter(1), reverse=True)[0][0]
            
            inst[best_feature] = 0
            acq_set[best_feature] = miss_feature_merits[best_feature]
            rem_set.remove(best_feature)
        
        acq_cost = sum([smr.acquisition_costs[feature] for feature in acq_set])
        self.set_cost_total += acq_cost
        return [(key, merit) for key, merit in acq_set.items()], acq_cost
        
    def get_name(self):
        return "KQSMRFSS"

class KQGainSMRFSS(AbstractSMRFeatureSetSelection):
    def __init__(self, k):
        """
        selects a set of feature acquisitions based on a greedy search of the highest quality gain per iterative acquisition of a feature until k features are selected or no gain is had
        :param k: how many acquisitions to add to the set at maximum
        """
        super().__init__(k=k)
        
    def get_acquisition_feature_set(self, smr, miss_feature_merits):
        """
        """
        inst = {}
        for feature in smr.merits:
            inst[feature] = nan if feature in miss_feature_merits else 0
        rem_set = list(miss_feature_merits.keys())
        acq_set = {}
        
        old_quality = smr._get_quality(inst, merits=smr.merits, acq_merits=[])
        max_iter = min(self.k, len(rem_set))
        for i in range(max_iter):
            qualities = {feature:smr._get_quality(inst, merits=smr.merits, acq_merits=[miss_feature_merits[feature]]) for feature in rem_set}
            best_feature, new_quality = sorted(qualities.items(), key=operator.itemgetter(1), reverse=True)[0]
            
            if new_quality <= old_quality: break
            
            old_quality = new_quality
            inst[best_feature] = 0
            acq_set[best_feature] = miss_feature_merits[best_feature]
            rem_set.remove(best_feature)
        
        acq_cost = sum([smr.acquisition_costs[feature] for feature in acq_set])
        self.set_cost_total += acq_cost
        return [(key, merit) for key, merit in acq_set.items()], acq_cost
        
    def get_name(self):
        return "KQGSMRFSS"

class KBudgetAwareQGainSMRFSS(AbstractSMRFeatureSetSelection):
    def __init__(self, k):
        """
        selects a set of feature acquisitions based on a greedy search of the highest quality gain per iterative acquisition of a feature until k features are selected or no gain is had
        :param k: how many acquisitions to add to the set at maximum
        """
        super().__init__(k=k)
    
    def get_acquisition_feature_set(self, smr, miss_feature_merits):
        """
        """
        inst = {}
        for feature in smr.merits:
            inst[feature] = nan if feature in miss_feature_merits else 0
        rem_set = list(miss_feature_merits.keys())
        acq_set = {}
        
        old_quality = smr._get_quality(inst, merits=smr.merits, acq_merits=[])
        max_iter = min(self.k, len(rem_set))
        for i in range(max_iter):
            qualities = {feature:smr._get_quality(inst, merits=smr.merits, acq_merits=[miss_feature_merits[feature]]) for feature in rem_set}
            best_feature, new_quality = sorted(qualities.items(), key=operator.itemgetter(1), reverse=True)[0]
            
            if new_quality <= old_quality * min(1, smr.budget_manager.used_budget()): break
            
            old_quality = new_quality
            inst[best_feature] = 0
            acq_set[best_feature] = miss_feature_merits[best_feature]
            rem_set.remove(best_feature)
        
        acq_cost = sum([smr.acquisition_costs[feature] for feature in acq_set])
        self.set_cost_total += acq_cost
        return [(key, merit) for key, merit in acq_set.items()], acq_cost
        
    def get_name(self):
        return "KBAQGSMRFSS"