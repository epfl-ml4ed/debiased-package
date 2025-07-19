from shutil import copytree

import os
import logging
import pickle
import numpy as np
import pandas as pd

# import torch.nn.functional as F
# import tensorflow as tf
import torch
from shutil import copytree, rmtree
from copy import deepcopy
from typing import Tuple

from debiased_jadouille.mitigation.inprocessing.inprocessor import InProcessor
from debiased_jadouille.mitigation.inprocessing.islam_repo.evaluations import evaluate_model, dp_distance, p_rule
from debiased_jadouille.mitigation.inprocessing.islam_repo.train_models import training_adversarial_debiasing, load_preTrainedModel


class IslamInProcessor(InProcessor):
    """inprocessing

        References:
            Islam, R., Pan, S., & Foulds, J. R. (2021, July). Can we obtain fairness for free?. In Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society (pp. 586-596).
            https://github.com/rashid-islam/F3_via_grid

    """
    
    def __init__(self, mitigating, discriminated, 
        hidden_layers=[64, 32, 16], mini_batch_size=128, learning_rate=0.001, drop_probs=0,
        activations='rectify', weight_decays=0, n_pretrain_epochs=1, n_epochs=1, a_func='leaky',
        lambdaa=1

    ):
        super().__init__({
            'mitigating': mitigating, 'discriminated': discriminated, 
            'hidden_layers':hidden_layers, 'mini_batch_size': mini_batch_size, 'learning_rate': learning_rate,
            'drop_probs': drop_probs, 'activations': activations, 'weight_decays': weight_decays,
            'n_pretrain_epochs': n_pretrain_epochs, 'n_epochs': n_epochs, 'a_func': a_func, 'lambda': lambdaa
        })
        self._hidden_layers = hidden_layers
        self._mini_batch_size = mini_batch_size
        self._learning_rate = learning_rate
        self._drop_probs = drop_probs
        self._activations = activations
        self._weight_decays = weight_decays
        self._n_pretrain_epochs = n_pretrain_epochs
        self._n_epochs = n_epochs
        self._a_func = a_func
        self._lambda = lambdaa
        self._information = {}

    def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
        x_data = torch.Tensor(x)
        demographic_attributes = self.extract_demographics(demographics)
        demos = self.get_binary_protected_privileged(demographic_attributes)
        demos = torch.Tensor(demos).unsqueeze(1)
        labels = torch.Tensor(y).unsqueeze(1)
        return x_data, labels, demos
    
    def _format_features(self, x:list, demographics:list) -> list:
        x_data = torch.Tensor(x)
        demographic_attributes = self.extract_demographics(demographics)
        demos = self.get_binary_protected_privileged(demographic_attributes)
        demos = torch.Tensor(demos).unsqueeze(1)
        return x_data, demos

    def _init_model(self):
        """Initiates a model with self._model
        """
        self.model = None

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
        data, labels, demos = self._format_final(x_train, y_train, demographics_train)
        self.model = training_adversarial_debiasing(
            len(x_train[0]), self._hidden_layers,
            self._a_func, self._drop_probs, self._learning_rate, 
            self._n_pretrain_epochs, self._n_epochs, 
            data, labels, self._mini_batch_size,
            self._weight_decays, demos, self._lambda
        )
    
    def predict(self, x: list, y, demographics: list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
            return x and y
        """
        data, _ = self._format_features(x, demographics)
        _, pred_label_train = evaluate_model(self.model, data)
        return pred_label_train, y

    def predict_proba(self, x: list, demographics:list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
        """
        data, _ = self._format_features(x, demographics)
        probabilities, _ = evaluate_model(self.model, data)
        pred0 = 1 - np.array(probabilities)
        probabilities = np.array([probabilities, pred0]).reshape(2, len(probabilities)).transpose()
        return probabilities


        
