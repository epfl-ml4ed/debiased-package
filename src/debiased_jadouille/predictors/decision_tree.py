import os
import copy
import pickle
import numpy as np

import tqdm
from typing import Tuple

from debiased_jadouille.predictors.predictor import Predictor
from sklearn.tree import DecisionTreeClassifier

class DTClassifier(Predictor):
    """This class implements the decision tree from scikit
    Args:
        Model (Model): inherits from the model class
    """
    
    def __init__(self, max_depth):
        super().__init__({'max_depth':max_depth})
        self._name = 'decision tree'
        self._notation = 'dt'
        self._max_depth = max_depth
        self._fold = -1


    def _format_final(self, x:list, y:list) -> Tuple[list, list]:
        return x, y
    
    def _format_features(self, x:list) -> list:
        return x
    
    def _init_model(self):
        self._set_seed(193)
        self.model = DecisionTreeClassifier(
            max_depth=self._max_depth
        )

    def init_model(self):
        self._init_model()


    def fit(self, x_train:list, y_train:list, x_val=[], y_val=[]):
        x_train, y_train = self._format_final(x_train, y_train)
        x_val, y_val = self._format_final(x_val, y_val)

        self._init_model()
        self.model.fit(x_train, y_train)
        

    def predict(self, x:list) -> list:
        test_x = self._format_features(x)
        predictions = self.model.predict(test_x)
        return predictions
    
    def predict_proba(self, x:list) -> list:
        test_x = self._format_features(x)
        predictions = self.model.predict_proba(test_x)
        return predictions
