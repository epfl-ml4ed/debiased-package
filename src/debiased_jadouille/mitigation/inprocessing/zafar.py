from shutil import copytree

import os
import logging
import pickle
import numpy as np
import pandas as pd
from collections import namedtuple

from sklearn import svm
from sklearn.model_selection import GridSearchCV

# import torch.nn.functional as F
# import tensorflow as tf
# import torch
from shutil import copytree, rmtree
from copy import deepcopy
from typing import Tuple

from debiased_jadouille.mitigation.inprocessing.donini_repo.linear_ferm import Linear_FERM
from debiased_jadouille.mitigation.inprocessing.inprocessor import InProcessor

class ZafarInProcessor(InProcessor):
    """inprocessing

        References:
            Zafar, M. B., Valera, I., Gomez-Rodriguez, M., & Gummadi, K. P. (2019). Fairness constraints: A flexible approach for fair classification. Journal of Machine Learning Research, 20(75), 1-42.
            https://github.com/jmikko/fair_ERM
    """
    
    def __init__(self, mitigating, discriminated, C=0.01, kernel='linear'):
        super().__init__({'mitigating': mitigating, 'discriminated': discriminated, 'C':C, 'kernel': kernel})
        self._C = C
        self._kernel = kernel
        self._information = {}

    def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
        data = pd.DataFrame(x)

        demographic_attributes = self.extract_demographics(demographics)
        demos = self.get_binary_protected_privileged(demographic_attributes)
        data['demographics'] = demos

        data = namedtuple('_', 'data, target')(np.array(data), y)
        return data
    
    def _format_features(self, x:list, demographics:list) -> list:
        data = pd.DataFrame(x)

        demographic_attributes = self.extract_demographics(demographics)
        demos = self.get_binary_protected_privileged(demographic_attributes)
        data['demographics'] = demos
        return np.array(data)

    def _init_model(self):
        """Initiates a model with self._model
        """
        self.svc = svm.SVC(C=self._C, kernel=self._kernel)
    
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
        data = self._format_final(x_train, y_train, demographics_train)

        self.model = Linear_FERM(
            data, self.svc, data.data[:, -1]
        )
        self.model.fit()
    
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
        predictions = self.model.predict(data)
        pred0 = 1 - np.array(predictions)
        probabilities = np.array([predictions, pred0]).reshape(2, len(predictions)).transpose()
        return probabilities
