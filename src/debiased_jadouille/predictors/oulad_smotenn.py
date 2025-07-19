import os
import copy
import pickle
import numpy as np

import tqdm
from typing import Tuple

from debiased_jadouille.predictors.predictor import Predictor

from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier

class SmoteENNRFBoostClassifier(Predictor):
    """This class implements the smoteENN (resampling for imbalanced problem) to random forest predictor as described in Hlosta, M., Zdrahal, Z., & Zendulka, J. (2018). Are we meeting a deadline? classification goal achievement in time in the presence of imbalanced data. Knowledge-Based Systems, 160, 278-295.
    This is implemented specifically for the oulad dataset

    Found through the original paper on the dataset
    Args:
        Model (Model): inherits from the model class
    """
    
    def __init__(self, n_estimators=500, max_depth=15):
        super().__init__({'n_estimators': n_estimators, 'max_depth': max_depth})
        self._n_estimators = n_estimators
        self._max_depth = max_depth

    def _format_final(self, x:list, y:list) -> Tuple[list, list]:
        return np.array(x), np.array(y)
    
    def _format_features(self, x:list) -> list:
        return np.array(x)
    
    def _init_model(self):
        self._set_seed(193)
        self._rebalancer = SMOTEENN()
        self.model = RandomForestClassifier(
            n_estimators = self._n_estimators,
            max_depth = self._max_depth
        )

    def init_model(self):
        self._init_model()

    def fit(self, x_train:list, y_train:list, x_val=[], y_val=[]):
        x_train, y_train = self._format_final(x_train, y_train)
        x_val, y_val = self._format_final(x_val, y_val)

        self._init_model()

        x_resampled, y_resampled = self._rebalancer.fit_resample(x_train, y_train)
        self.model.fit(x_resampled, y_resampled)
        

    def predict(self, x:list) -> list:
        test_x = self._format_features(x)
        predictions = self.model.predict(test_x)
        return predictions

    def predict_proba(self, x:list) -> list:
        test_x = self._format_features(x)
        predictions = self.model.predict_proba(test_x)
        return predictions
