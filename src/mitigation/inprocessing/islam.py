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

from mitigation.inprocessing.inprocessor import InProcessor
from mitigation.inprocessing.islam_repo.evaluations import evaluate_model, dp_distance, p_rule
from mitigation.inprocessing.islam_repo.train_models import training_adversarial_debiasing, load_preTrainedModel


class IslamInProcessor(InProcessor):
    """inprocessing

        References:
            Islam, R., Pan, S., & Foulds, J. R. (2021, July). Can we obtain fairness for free?. In Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society (pp. 586-596).
            https://github.com/rashid-islam/F3_via_grid

    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'islam et al.'
        self._notation = 'islam'
        self._inprocessor_settings = self._settings['inprocessors']['islam']
        self._information = {}
        self._fold = -1

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
        data, labels, demos = self._format_final(x_train, y_train, demographics_train)
        self.model = training_adversarial_debiasing(
            len(x_train[0]), self._inprocessor_settings['hidden_layers'],
            self._inprocessor_settings['a_func'], self._inprocessor_settings['drop_probs'], self._inprocessor_settings['learning_rate'], 
            self._inprocessor_settings['n_pretrain_epochs'], self._inprocessor_settings['n_epochs'], 
            data, labels, self._inprocessor_settings['mini_batch_size'],
            self._inprocessor_settings['weight_decays'], demos, self._inprocessor_settings['lambda']
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
    

        
