import os
import copy
import pickle
import numpy as np
import pandas as pd

import tqdm
from typing import Tuple

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from debiased_jadouille.predictors.predictor import Predictor

from sklearn.linear_model import LogisticRegression

class LogisticRegressionClassifier(Predictor):
    """This class implements the logistic regression (to be trained with the "best features") as described in Gervet, T., Koedinger, K., Schneider, J., & Mitchell, T. (2020). When is deep learning the best approach to knowledge tracing?. Journal of Educational Data Mining, 12(3), 31-54 (LR) and Schmucker, R., Wang, J., Hu, S., & Mitchell, T. M. (2021). Assessing the performance of online students--new data, new approaches, improved accuracy. arXiv preprint arXiv:2109.01753 (EEDI)
    made for eedi
    Found through the paper citing the original paper

    Args:
        Model (Model): inherits from the model class
    """
    
    def __init__(self, penalty, C, solver):
        super().__init__({'penalty': penalty, 'C': C, 'solver':solver})
        self._penalty = penalty
        self._C = C
        self._solver = solver


    def _format_final(self, x:list, y:list) -> Tuple[list, list]:
        data = pd.DataFrame(x)
        return np.array(data), np.array(y)
    
    def _format_features(self, x:list) -> list:
        data = pd.DataFrame(x)
        return np.array(data)
    
    def _init_model(self):
        self._set_seed(193)
        self.model = LogisticRegression(
            penalty = self._penalty,
            C=self._C,
            solver=self._solver
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
