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

from debiased.mitigation.inprocessing.inprocessor import InProcessor

class InProcessor(InProcessor):
    """inprocessing

        References:


    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = ' et al.'
        self._notation = ''
        self._inprocessor_settings = self._settings['inprocessors']['']
        self._information = {}
        self._fold = -1
        

    def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
        demographic_attributes = self.extract_demographics(demographics)
        return np.array(x), np.array(y)
    
    def _format_features(self, x:list, demographics:list) -> list:
        demographic_attributes = self.extract_demographics(demographics)
        return np.array(x)

    def _init_model(self):
        """Initiates a model with self._model
        """
        self.model = None
        raise NotImplementedError

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
        raise NotImplementedError
    
    def predict(self, x: list, y, demographics: list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
            return x and y
        """
        raise NotImplementedError

    def predict_proba(self, x: list, demographics:list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
        """
        # predictions = predictions.cpu().detach().numpy()
        # pred0 = 1 - np.array(predictions)
        # probabilities = np.array([predictions, pred0]).reshape(2, len(predictions)).transpose()
        raise NotImplementedError

        
