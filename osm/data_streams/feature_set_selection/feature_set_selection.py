import random

class RandomFeatureSetSelection():
    def __init__(self, set_size):
        """
        gets a random feature set from 
        """
        super().__init__()
        self.set_size = set_size
    
    def get_feature_set(self, context):
        """
        returns the feature set according to some method
        :param context: the relevant context, data or parent method, to which this selection method owes its feature set
        """
        return random.sample(context, min(self.k, len(miss_feature_merits)))
    
    def get_name(self):
        return "random_feature_set_selection"