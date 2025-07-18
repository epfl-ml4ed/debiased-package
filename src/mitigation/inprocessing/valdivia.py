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

from mitigation.inprocessing.inprocessor import InProcessor

class ValdiviaInProcessor(InProcessor):
    """inprocessing

        References:
            Valdivia, A., Sánchez‐Monedero, J., & Casillas, J. (2021). How fair can we go in machine learning? Assessing the boundaries of accuracy and fairness. International Journal of Intelligent Systems, 36(4), 1619-1643.
            https://github.com/anavaldi/fairness_nsga/tree/master/nsga2
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = ' et al.'
        self._notation = ''
        self._inprocessor_settings = self._settings['inprocessors']['valdivia']
        self._information = {}
        self._fold = -1

    def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
        return np.array(x), np.array(y)
    
    def _format_features(self, x:list, demographics:list) -> list:
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
        x_val:list, y_val:list, demographics_val: list
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
        raise NotImplementedError

    def save(self, extension='') -> str:
        """Saving the model in the following path:
        '../experiments/run_year_month_day/models/model_name_fx.pkl

        Returns:
            String: Path
        """
        path = '{}/models/'.format(self._settings['experiment']['name'])
        os.makedirs(path, exist_ok=True)
        with open('{}{}_{}.pkl'.format(path, self._notation, extension), 'wb') as fp:
            pickle.dump(self._information, fp)
        return '{}{}_{}'.format(path, self._notation, extension)

    def save_fold(self, fold: int) -> str:
        return self.save(extension='fold_{}'.format(fold))

    def save_fold_early(self, fold: int) -> str:
        return self.save(extension='fold_{}_len{}'.format(
            fold, self._maxlen
        ))
    

        
