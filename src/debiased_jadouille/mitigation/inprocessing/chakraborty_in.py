from shutil import copytree

import os
import logging
import pickle
import numpy as np
import pandas as pd

from shutil import copytree, rmtree
from copy import deepcopy
from typing import Tuple

from debiased_jadouille.mitigation.inprocessing.chakraborty_repo.flash import flash_fair_LSR
from sklearn.linear_model import LogisticRegression
from debiased_jadouille.mitigation.inprocessing.inprocessor import InProcessor

class ChakrabortyInProcessor(InProcessor):
    """inprocessing

        References:
            Chakraborty, J., Majumder, S., Yu, Z., & Menzies, T. (2020, November). Fairway: a way to build fair ML software. In Proceedings of the 28th ACM joint meeting on European software engineering conference and symposium on the foundations of software engineering (pp. 654-665).
            https://github.com/joymallyac/Fairway/tree/master

    """
    
    def __init__(self, mitigating, discriminated, goals='ABCD'):
        super().__init__({'mitigating': mitigating, 'discriminated': discriminated, 'goals': goals})
        self._information = {}
        self._goals = goals

    def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
        data = pd.DataFrame(x)
        demographic_attributes = self.extract_demographics(demographics)
        demos = self.get_binary_protected_privileged(demographic_attributes)
        data['demographics'] = demos
        data['Probability'] = y
        data.columns = [str(col) for col in data.columns]
        return data
    
    def _format_features(self, x:list, demographics:list) -> list:
        return np.array(x)

    def _init_model(self):
        """Initiates a model with self._model
        """
        self.model = None

    def init_model(self):
        self._init_model()

    def fit(self, 
        x_train: list, y_train: list, demographics_train: list,
        x_val=[], y_val=[], demographics_val=[]
    ):
        """fits the model with the training data x, and labels y. 
        Warning: Init the model every time this function is called

        Args:
            x_train (list): training feature data 
            y_train (list): training label data
            x_val (list): validation feature data
            y_val (list): validation label data
        """
        self._init_model()
        demographic_attributes = self.extract_demographics(demographics_train)
        data = self._format_final(x_train, y_train, demographics_train)
        best_config = flash_fair_LSR(data, 'demographics', self._goals)
        p1 = best_config[0]
        if best_config[1] == 1:
            p2 = 'l1'
        else:
            p2 = 'l2'
        if best_config[2] == 1:
            p3 = 'liblinear'
        else:
            p3 = 'saga'
        p4 = best_config[3]
        self.model = LogisticRegression(C=p1, penalty=p2, solver=p3, max_iter=p4)
        self.model.fit(x_train, y_train)
    
    def predict(self, x: list, y, demographics: list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
            return x and y
        """
        return self.model.predict(x), y

    def predict_proba(self, x: list, demographics:list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
        """
        x = [list(xx) for xx in x]
        return self.model.predict_proba(x)

