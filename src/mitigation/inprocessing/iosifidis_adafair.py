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
from mitigation.inprocessing.iosifidis_adafair_repo.AdaFair import AdaFair

class IosifidisInProcessor(InProcessor):
    """inprocessing -> adaboost reweighting instances - half post processing

        References:
            Iosifidis, V., Roy, A., & Ntoutsi, E. (2022). Parity-based cumulative fairness-aware boosting. Knowledge and Information Systems, 64(10), 2737-2770.
            https://github.com/iosifidisvasileios/AdaFair/tree/master/data

    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'iosifidis et al.'
        self._notation = 'iosifidis3'
        self._inprocessor_settings = self._settings['inprocessors']['iosifidis']
        self._information = {}
        self._fold = -1

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
            self._inprocessor_settings['n_estimators'], 
            saIndex=-1, saValue=1, trade_off_c=self._inprocessor_settings['trade_off_c']
        )

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
    

        
