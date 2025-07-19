from shutil import copytree

import os
import logging
import pickle
import numpy as np
import pandas as pd

from shutil import copytree, rmtree
from copy import deepcopy
from typing import Tuple

from debiased_jadouille.mitigation.inprocessing.kilbertus_repo.fair_logistic_regression import FairLogisticRegression
from debiased_jadouille.mitigation.inprocessing.inprocessor import InProcessor

class KilbertusInProcessor(InProcessor):
    """inprocessing

        References:
            Kilbertus, N., Gascón, A., Kusner, M., Veale, M., Gummadi, K., & Weller, A. (2018, July). Blind justice: Fairness with encrypted sensitive attributes. In International Conference on Machine Learning (pp. 2630-2639). PMLR.
            https://github.com/nikikilbertus/blind-justice/tree/master/python/src

    """
    
    def __init__(self, mitigating, discriminated, optimiser='unconstrained', epochs=10, batchsize=64):
        super().__init__({'mitigating': mitigating, 'discriminated': discriminated, 'optimiser': optimiser, 'epochs': epochs, 'batchsize': batchsize})
        self._optimiser = optimiser
        self._epochs = epochs
        self._batchsize = batchsize
        self._information = {}

    def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
        return np.array(x), np.array(y)
    
    def _format_features(self, x:list, demographics:list) -> list:
        return np.array(x)

    def _init_model(self):
        """Initiates a model with self._model
        """
        self.model = FairLogisticRegression(
            opt=self._optimiser,
            random_state=193
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
        demographic_attributes = self.extract_demographics(demographics_train)
        demos = self.get_binary_protected_privileged(demographic_attributes)
        demos = [[dd] for dd in demos]
        n_epochs = int(
            self._epochs * self._batchsize / len(x_train)
        )
        n = len(x_train)
        d = len(x_train[0])
        p = 1

        print(
            'x', np.array(x_train).shape,
            'y', np.array(y_train).shape,
            'z', np.array(demos).shape
        )
        
        self.model.fit(
            np.array(x_train), np.array(y_train), np.array(demos)
        )
    
    def predict(self, x: list, y, demographics: list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
            return x and y
        """
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
        predictions = self.model.predict_proba(x)
        pred0 = 1 - np.array(predictions)
        probabilities = np.array([predictions, pred0]).reshape(2, len(predictions)).transpose()
        return probabilities

