import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

"""
Classifier for Running Example 1: inherits from sklearn's BaseEstimator
"""

class FClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param
        self.y_ = None

    def predict(self, X):
        closest = 1*(X[:,1]>(4*np.sin(X[:,0])))
        self.y_ = closest
        return self.y_

class SClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param
    def predict(self, X):
        closest = 1*(X[:,1]>np.sin(X[:,0])/X[:,0])
        self.y_ = closest
        return self.y_

class VerticalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param
    def predict(self, X):
        closest = 1*(X[:,0]>np.pi)
        self.y_ = closest
        return self.y_
    
class CornerToCornerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param
    def predict(self, X):
        closest = 1*(X[:,0]>np.pi/2)
        self.y_ = closest
        return self.y_

