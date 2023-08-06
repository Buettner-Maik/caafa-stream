from abc import ABC, abstractmethod
from multiprocessing.sharedctypes import Value
import pandas as pd
import numpy as np
import osm.data_streams.constants as const

from osm.data_streams.active_feature_acquisition.supervised_merit_ranking.supervised_merit_ranking import AbstractSMR
from osm.data_streams.active_feature_acquisition.abstract_feature_acquisition import AbstractActiveFeatureAcquisitionStrategy
from osm.data_streams.active_feature_acquisition.supervised_merit_ranking.smr_feature_set_selection import AbstractSMRFeatureSetSelection

epsilon = 1 / (2 ** 20)

class PoolClassifierFeatureImportanceAFA(AbstractSMR):
    def __init__(self, window, fi_classifier, pipeline, target_col_name, budget_manager, 
        feature_selection, acq_set_size, dynamic_budget_threshold=False, 
        categories=None, acquisition_costs={}, budget_option=[('inst', 1)], debug=True):
        """
        Performs Supervised Merit Ranking AFA using a pool classifier's provided
        feature importances as ranking values
        Ranking values are updated per batch
        :param window: the framework window
        :param fi_classifier: a classifier that is trained to provide the feature importance values every batch
        :param pipeline: The sklearn.Pipeline which transforms the window data to train the classifier on
        :param feature_selection: the method by which the feature set for acquisition is selected
        :param dynamic_budget_threshold: whether the budget threshold should be dynamically adjusted once per batch
        :param categories: provide categories for all categorical features as dict 
        """
        #check for classifier
        if not "feature_importances_" in dir(fi_classifier):
            raise ValueError("The provided method has to support an internal feature_importance method.")
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
        self.fi_classifier = fi_classifier
        self.pipeline = pipeline

    def _initialize(self, data):
        """
        """
        super()._initialize(data)
        self.features = {}
        for i, col in enumerate(data):
            self.features[col] = i
        #self.features.remove(self.target_col_name)

    def _get_rank_values(self):
        """
        returns a dict containing all features and their importances
        """
        cols = {}
        fi = self.fi_classifier.feature_importances_

        for col in self.num_cols:
            cols[col] = fi[self.features[col]] + np.random.uniform() * epsilon
        # for col in self.cat_cols:
        #     cols[col] = fi[self.features[col]]
        return cols

    def _on_new_batch(self, data):
        #train on window
        dt = self.pipeline.fit_transform(data, data[self.target_col_name])
        self.fi_classifier.fit(dt, data[self.target_col_name])

    def _update_window(self, inst):
        pass

    def get_name(self):
        return "PCFI+" + self.feature_selection.get_name()