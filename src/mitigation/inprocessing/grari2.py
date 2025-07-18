from shutil import copytree

import os
import logging
import pickle
import numpy as np
import pandas as pd

from shutil import copytree, rmtree
from copy import deepcopy
from typing import Tuple

from sklearn.ensemble import GradientBoostingClassifier
from mitigation.inprocessing.grari_repo.functions import *
from mitigation.inprocessing.inprocessor import InProcessor

class Grari2InProcessor(InProcessor):
    """inprocessing

        References:


    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'grari2 et al.'
        self._notation = 'grari2'
        self._inprocessor_settings = self._settings['inprocessors']['grari2']
        self._information = {}
        self._fold = -1

    def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
        demographic_attributes = self.extract_demographics(demographics)
        demos = self.get_binary_protected_privileged(demographic_attributes)
        return np.array(x), np.array(y), np.array(demos)
    
    def _format_features(self, x:list, demographics:list) -> list:
        demographic_attributes = self.extract_demographics(demographics)
        demos = self.get_binary_protected_privileged(demographic_attributes)
        return np.array(x), np.array(demos)

    def _init_model(self):
        """Initiates a model with self._model
        """
        # self.model = GradientBoostingClassifier(
        #     n_estimators=self._inprocessor_settings['n_estimators'], learning_rate=self._inprocessor_settings['learning_rate'],
        #     max_depth=self._inprocessor_settings['max_depth'], max_features=int(self._inprocessor_settings['max_features'] * self._input_size),
        #     random_state=self._settings['seeds']['inprocessor'], min_impurity_decrease=self._inprocessor_settings['min_impurity_decrease'],
        #     min_samples_leaf=self._inprocessor_settings['min_samples_leaf'],
        #     min_samples_split=self._inprocessor_settings['min_samples_split'], min_weight_fraction_leaf=self._inprocessor_settings['min_weight_fraction_leaf']
        # )
        self.model = FAGTB(
            n_estimators=self._inprocessor_settings['n_estimators'], learning_rate=self._inprocessor_settings['learning_rate'], 
            max_depth=self._inprocessor_settings['max_depth'], min_samples_split=self._inprocessor_settings['min_samples_split'], 
            min_impurity=False, max_features=int(self._inprocessor_settings['max_features'] * self._input_size), 
            regression=1
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
        self._input_size = len(x_train[0])
        self._init_model()

        x, y, demos = self._format_final(x_train, y_train, demographics_train)
        y_pred = self.model.fit(
            x, y, demos, LAMBDA=self._inprocessor_settings['lambda'], Xtest=x, yt=y, sensitivet=demos
        )
    
    def predict(self, x: list, y, demographics: list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
            return x and y
        """
        x, _ = self._format_features(x, demographics)
        predictions = self.model.predict(x)
        predictions = [int(p >= 0.5) for p in predictions]
        return predictions, y

    def predict_proba(self, x: list, demographics:list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
        """
        x, _ = self._format_features(x, demographics)
        probabilities = self.model.predict(x)
        pred0 = 1 - np.array(probabilities)
        probabilities = np.array([probabilities, pred0]).reshape(2, len(probabilities)).transpose()
        return probabilities

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
    

        
