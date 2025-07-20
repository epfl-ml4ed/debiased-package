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

class InProcessor:
    """This implements the superclass which will be used in the machine learning pipeline
    """
    
    def __init__(self, settings: dict):
        self._settings = deepcopy(settings)
        self._random_seed = 193
        self._information = {}

    def get_name(self):
        """Returns the name of the model, useful when debugging
        """
        return self._name

    def get_notation(self):
        """Shorter name of the model. Especially used when saving files in patah which
        contain the name of the models.
        """
        return self._notation

    def _set_seed(self, seed):
        """Set the seed for the parameters initialisation or anything else
        """
        np.random.seed(seed)

    def get_settings(self):
        return dict(self._model_settings)

    def extract_demographics(self, demographics):
        attributes = self._settings['mitigating'].split('.')
        dems = ['' for _ in range(len(demographics))]
        for att in attributes:
            dems = ['{}_{}'.format(dems[i], demographics[i][att]) for i in range(len(demographics))]
        return dems

    def get_binary_protected_privileged(self, demographics_attributes):
        """Returns 1 if protected, 0 if privileged

        Args:
            demographics_attributes (_type_): as returned by the function self.extract_demographics(demographics)
        """
        discriminated_list = [str(att) for att in self._settings['discriminated'].split('.')]
        return [1 if demographics_attributes[i] in discriminated_list else 0 for i in range(len(demographics_attributes))] 

    def get_binary_privileged(self, demographics_attributes):
        """Returns 1 if protected, 0 if privileged

        Args:
            demographics_attributes (_type_): as returned by the function self.extract_demographics(demographics)
        """
        discriminated_list = [str(att) for att in self._settings['discriminated'].split('.')]
        return [0 if demographics_attributes[i] in discriminated_list else 1 for i in range(len(demographics_attributes))] 

    def get_protected_indices(self, demographics_attributes):
        """Returns the indices of the instances belonging to the protected (discriminated) group

        Args:
            demographics_attributes (_type_): as returned by the function self.extract_demographics(demographics)
        """
        discriminated_list = [str(att) for att in self._settings['discriminated'].split('.')]
        return [i for i in range(len(demographics_attributes)) if demographics_attributes[i] in discriminated_list]

    def get_privileged_indices(self, demographics_attributes):
        """Returns the indices of the instances belonging to the protected (discriminated) group

        Args:
            demographics_attributes (_type_): as returned by the function self.extract_demographics(demographics)
        """
        discriminated_list = [str(att) for att in self._settings['discriminated'].split('.')]
        return [i for i in range(len(demographics_attributes)) if demographics_attributes[i] not in discriminated_list]


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

    def save(self) -> str:
        """Saving the model in the following path:
        '../experiments/run_year_month_day/models/model_name_fx.pkl

        Returns:
            String: Path
        """
        raise NotImplementedError
    

        
