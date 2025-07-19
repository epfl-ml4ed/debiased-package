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

from debiased_jadouille.predictors.xuetangx_svc import StandardScalingSVCClassifier
from debiased_jadouille.predictors.logistic_regression import LogisticRegressionClassifier
from debiased_jadouille.predictors.oulad_smotenn import SmoteENNRFBoostClassifier
from debiased_jadouille.predictors.portugal_garf import GARFClassifier

from debiased_jadouille.mitigation.inprocessing.inprocessor import InProcessor
from debiased_jadouille.mitigation.inprocessing.schreuder_repo.post_process import TransformDPAbstantion

class SchreuderInProcessor(InProcessor):
    """inprocessing

       References:
        Schreuder, N., & Chzhen, E. (2021, December). Classification with abstention but without disparities. In Uncertainty in Artificial Intelligence (pp. 1227-1236). PMLR.
        https://github.com/evgchz/dpabst


    """
    
    def __init__(self, mitigating, discriminated, alpha=0.5, perc_unlab=0.2, shuffle=True, predictor='svc'):
        super().__init__({'mitigating': mitigating, 'discriminated': discriminated, 'alpha': alpha, 'perc_unlab': perc_unlab, 'shuffle':shuffle, 'predictor': predictor})
        self._alpha = alpha
        self._perc_unlab = perc_unlab
        self._shuffle = shuffle
        self._predictor = predictor
        self._information = {}

    def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
        xx = np.array(x)
        yy = np.array(y)
        dd = np.array(demographics)
        if self._shuffle:
            permutation = np.random.permutation(len(y))
            xx = xx[permutation]
            yy = yy[permutation]
            dd = dd[permutation]
        n_unlab = int(len(y) * self._perc_unlab)
        n_train = len(y) - n_unlab

        train_features = np.array(xx[:n_train])
        train_labels = np.array(yy[:n_train])
        train_demographics = np.array(dd[:n_train])
        train_data = pd.DataFrame(train_features)
        train_data['demographics'] = train_demographics 
        train_data = np.array(train_data)

        ul_features = np.array(xx[n_train:])
        ul_labels = np.array(yy[n_train:])
        ul_demographics = np.array(dd[n_train:])
        ul_data = pd.DataFrame(ul_features)
        ul_data['demographics'] = ul_demographics
        ul_data = np.array(ul_data)

        return train_data, train_labels, ul_data, ul_labels
    
    def _format_features(self, x:list, demographics:list) -> list:
        data = pd.DataFrame(x)
        demos = self.extract_demographics(demographics)
        demos = self.get_binary_protected_privileged(demos)
        data['demographics'] = demos
        return np.array(data)

    def _choose_predictor(self):
        if self._predictor == 'svc':
            self._model = StandardScalingSVCClassifier
        if self._predictor == 'lr':
            self._model = LogisticRegressionClassifier
        if self._predictor == 'smoteRF':
            self._model = SmoteENNRFBoostClassifier
        if self._predictor == 'gaRF':
            self._model = GARFClassifier

    def _init_model(self):
        """Initiates a model with self._model
        """
        self.model = self._model(self._settings)

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
        demographic_attributes = self.get_binary_protected_privileged(demographic_attributes)
        alphas_dict = {}
        for un in np.unique(demographic_attributes):
            alphas_dict[un] = self._inprocessor_settings['alpha']
        train_data, train_labels, ul_data, ul_labels = self._format_final(x_train, y_train, demographic_attributes)
        
        self.model.fit(train_data, train_labels, train_data, train_labels)
        self.transformer = TransformDPAbstantion(self.model.model, alphas_dict)
        self.transformer.fit(ul_data)
    
    def predict(self, x: list, y:list, demographics:list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
            return x and y
        """
        data = self._format_features(x, demographics)
        predictions = self.transformer.predict(data)
        prediction_indices = [pp for pp in range(len(predictions)) if predictions[pp] < 2]
        predictions = [predictions[pi] for pi in prediction_indices]
        new_y = [y[pi] for pi in prediction_indices]
        self.predictions = [pp for pp in predictions]
        return predictions, new_y

    def predict_proba(self, x: list, demographics:list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
        """

        preds = pd.DataFrame(self.predictions)
        preds[1] = np.abs(preds[0] - 1)
        return np.array(preds)

    

        
