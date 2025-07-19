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
from debiased_jadouille.mitigation.inprocessing.grari_repo.functions import *
from debiased_jadouille.mitigation.inprocessing.inprocessor import InProcessor

class Grari2InProcessor(InProcessor):
    """inprocessing

        References:


    """
    
    def __init__(self, mitigating, discriminated, 
        n_estimators=100, learning_rate=0.05, max_depth=3, max_features=0.9,
        min_impurity_decrease=0, min_impurity_split=None, min_samples_leaf=2,
        min_samples_split=2, min_weight_fraction_leaf=0, presort=0, lambdaa=0.1
    ):
        super().__init__({
            'mitigating': mitigating, 'discriminated': discriminated, 
            'n_estimators':n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth,
            'max_features': max_features, 'min_impurity_decrease': min_impurity_decrease, 'min_impurity_split': min_impurity_split,
            'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split, 'min_weight_fraction_leaf': min_weight_fraction_leaf,
            'presort': presort, 'lambda': lambdaa

        })
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self._max_features = max_features
        self._min_impurity_decrease = min_impurity_decrease
        self._min_impurity_split = min_impurity_split
        self._min_samples_leaf = min_samples_leaf
        self._min_samples_split = min_samples_split
        self._min_weight_fraction_leaf = min_weight_fraction_leaf
        self._presort = presort
        self._lambda = lambdaa
        self._information = {}

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
        self.model = FAGTB(
            n_estimators=self._n_estimators, learning_rate=self._learning_rate, 
            max_depth=self._max_depth, min_samples_split=self._min_samples_split, 
            min_impurity=False, max_features=int(self._max_features * self._input_size), 
            regression=1
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
        self._input_size = len(x_train[0])
        self._init_model()

        x, y, demos = self._format_final(x_train, y_train, demographics_train)
        y_pred = self.model.fit(
            x, y, demos, LAMBDA=self._lambda, Xtest=x, yt=y, sensitivet=demos
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

