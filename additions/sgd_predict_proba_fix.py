from sklearn.linear_model import SGDClassifier

class SGDPredictFix(SGDClassifier):
    def __init__(self):
        super().__init__(loss='log')
    
    def predict_proba(self, X):
        #apparently properties aren't covered by hasattr()
        #so this just adds a function with the same name
        super().predict_proba(X)