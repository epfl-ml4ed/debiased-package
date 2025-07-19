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

from debiased_jadouille.mitigation.inprocessing.oneto_repo.GeneralFairERM import GeneralFairERM
from debiased_jadouille.mitigation.inprocessing.inprocessor import InProcessor

class OnetoInProcessor(InProcessor):
    """inprocessing

        References:
            Oneto, L., Donini, M., & Pontil, M. (2020, July). General fair empirical risk minimization. In 2020 International Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE.
            https://github.com/hyungrok-do/fair-glm-cvx/blob/main/models/oneto.py
    """
    
    def __init__(self, mitigating, discriminated):
        super().__init__({'mitigating': mitigating, 'discriminated': discriminated})
        self._information = {}

    def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
        data = pd.DataFrame(x)

        demographic_attributes = self.extract_demographics(demographics)
        demos = self.get_binary_protected_privileged(demographic_attributes)
        data['demographics'] = demos
        data = np.array(data)
        
        return data, np.array(y)
    
    def _format_features(self, x:list, demographics:list) -> list:
        data = pd.DataFrame(x)

        demographic_attributes = self.extract_demographics(demographics)
        demos = self.get_binary_protected_privileged(demographic_attributes)
        data['demographics'] = demos
        data = np.array(data)

    def _init_model(self):
        """Initiates a model with self._model
        """
        self.model = GeneralFairERM(
            sensitive_index=-1
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
        return self.model._predict(data), y

    def predict_proba(self, x: list, demographics:list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
        """
        data = self._format_features(x, demographics)
        predictions = self.model._predict(data)
        predictions = predictions.cpu().detach().numpy()
        pred0 = 1 - np.array(predictions)
        probabilities = np.array([predictions, pred0]).reshape(2, len(predictions)).transpose()
        return probabilities


        
