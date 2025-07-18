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

from mitigation.inprocessing.fish_repo import sdp
from mitigation.inprocessing.inprocessor import InProcessor

class FishInProcessor(InProcessor):
    """inprocessing

        References:


    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'fish et al.'
        self._notation = 'fish'
        self._inprocessor_settings = self._settings['inprocessors']['fish']
        self._information = {}
        self._fold = -1

    def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
        data = pd.DataFrame(x)
        demos = self.extract_demographics(demographics)
        demos = self.get_binary_protected_privileged(demos)
        data['demographics'] = demos
        data = np.array(data)
        return zip(data, y), len(data[0]) - 1
    
    def _format_features(self, x:list, demographics:list) -> list:
        return np.array(x)

    def _init_model(self):
        """Initiates a model with self._model
        """
        if self._inprocessor_settings['model'] == 'lr':
            self.model = sdp.lrLearner
        if self._inprocessor_settings['model'] == 'boosting':
            self.model = sdp.boostingLearner
        if self._inprocessor_settings['model'] == 'svm':
            self.model = sdp.svmLearner
        if self._inprocessor_settings['model'] == 'svm_linear':
            self.model = sdp.svmLinearLearner

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
        data, label = self._format_final(x_train, y_train, demographics_train)
        self.new_classifier = self.model(data, label, 1)
        preds = self.new_classifier(x_train)
        print('preds')
        print(preds)
    
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
    

        
