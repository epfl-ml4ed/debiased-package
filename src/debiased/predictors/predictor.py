from shutil import copytree

import os
import logging
import pickle
import numpy as np
import pandas as pd

from shutil import copytree, rmtree
from copy import deepcopy
from typing import Tuple

class Predictor:
    """This implements the superclass which will be used in the machine learning pipeline
    """
    
    def __init__(self, settings: dict):
        self._settings = deepcopy(settings)
        self._random_seed = 193
        self._gs_fold = 0


    def _set_seed(self, seed):
        """Set the seed for the parameters initialisation or anything else
        """
        np.random.seed(seed)

    def _init_model(self):
        """Initiates a model with self._model
        """
        raise NotImplementedError

    def fit(self, x_train: list, y_train: list):
        """fits the model with the training data x, and labels y. 
        Warning: Init the model every time this function is called

        Args:
            x_train (list): training feature data 
            y_train (list): training label data
            x_val (list): validation feature data
            y_val (list): validation label data
        """
        raise NotImplementedError
    
    def predict(self, x: list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
        """
        raise NotImplementedError

    def predict_proba(self, x: list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
        """
        raise NotImplementedError

        
