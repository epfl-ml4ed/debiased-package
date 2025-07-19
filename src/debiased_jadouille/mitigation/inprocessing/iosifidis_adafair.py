from shutil import copytree

import os
import logging
import pickle
import numpy as np
import pandas as pd

# import torch.nn.functional as F
# import tensorflow as tf
# import torch
from shutil import copytree, rmtree
from copy import deepcopy
from typing import Tuple

from debiased_jadouille.mitigation.inprocessing.inprocessor import InProcessor
from debiased_jadouille.mitigation.inprocessing.iosifidis_adafair_repo.AdaFair import AdaFair

class IosifidisInProcessor(InProcessor):
    """inprocessing -> adaboost reweighting instances - half post processing

        References:
            Iosifidis, V., Roy, A., & Ntoutsi, E. (2022). Parity-based cumulative fairness-aware boosting. Knowledge and Information Systems, 64(10), 2737-2770.
            https://github.com/iosifidisvasileios/AdaFair/tree/master/data

    """
    
    def __init__(self, mitigating, discriminated, n_estimators=200,trade_off_c=0.1):
        super().__init__({'mitigating': mitigating, 'discriminated': discriminated, 'trade_off_c': trade_off_c})
        self._trade_off_c = trade_off_c
        self._n_estimators = n_estimators
        self._information = {}

    def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
        data = pd.DataFrame(x)
        demos = self.extract_demographics(demographics)
        demos = self.get_binary_protected_privileged(demos)
        data['demographics'] = demos

        yy = [-1 if y[i]==0 else 1 for i in range(len(y))]
        return np.array(data), np.array(yy)
    
    def _format_features(self, x:list, demographics:list) -> list:
        data = pd.DataFrame(x)
        demos = self.extract_demographics(demographics)
        demos = self.get_binary_protected_privileged(demos)
        data['demographics'] = demos
        return np.array(data)

    def _init_model(self):
        """Initiates a model with self._model
        """
        self.model = AdaFair(
            self._n_estimators, 
            saIndex=-1, saValue=1, trade_off_c=self._trade_off_c
        )

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
        data, labels = self._format_final(x_train, y_train, demographics_train)
        self.model.fit(data, labels)
    
    def predict(self, x: list, y, demographics: list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
            return x and y
        """
        data = self._format_features(x, demographics)
        return self.model.predict(data), y

    def predict_proba(self, x: list, demographics:list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
        """
        data = self._format_features(x, demographics)
        return self.model.predict_proba(data)

