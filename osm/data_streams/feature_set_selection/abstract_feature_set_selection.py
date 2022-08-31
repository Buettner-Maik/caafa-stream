from abc import ABC, abstractmethod

from osm.data_streams.abstract_base_class import AbstractBaseClass



class AbstractFeatureSetSelection(AbstractBaseClass):
    def __init__(self):
        """
        """
        pass
    
    @abstractmethod
    def get_feature_set(self, context):
        """
        returns the feature set according to some method
        :param context: the relevant context, data or parent method, to which this selection method owes its feature set
        """
        pass
    
    def get_name(self):
        return "abstract_feature_set_selection"