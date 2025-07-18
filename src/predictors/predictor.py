from shutil import copytree

import os
import logging
import pickle
import numpy as np
import pandas as pd

from shutil import copytree, rmtree
from copy import deepcopy
from typing import Tuple

class Predictor:
    """This implements the superclass which will be used in the machine learning pipeline
    """
    
    def __init__(self, settings: dict):
        self._settings = deepcopy(settings)
        self._experiment_name = settings['experiment']['name']
        self._random_seed = settings['seeds']['model']
        self._gs_fold = 0

    def get_name(self):
        """Returns the name of the model, useful when debugging
        """
        return self._name

    def get_notation(self):
        """Shorter name of the model. Especially used when saving files in patah which
        contain the name of the models.
        """
        return self._notation

    def _set_seed(self):
        """Set the seed for the parameters initialisation or anything else
        """
        np.random.seed(self._settings['seeds']['model'])

    def set_gridsearch_parameters(self, params, combinations):
        """When using a gridsearch, the model uses this function to update
        its arguments. 
        """
        logging.debug('Gridsearch params: {}'.format(params))
        logging.debug('Combinations: {}'.format(combinations))
        print('    ', params, combinations)
        for i, param in enumerate(params):
            logging.debug('  index: {}, param: {}'.format(i, param))
            self._model_settings[param] = combinations[i]

    def set_gridsearch_fold(self, fold:int):
        """Used to save the model under a specific name
        """
        self._gs_fold = fold

    def set_outer_fold(self, fold:int):
        self._outer_fold = fold
            
    def get_settings(self):
        return dict(self._model_settings)

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

    # def _load_weights_torch(self, path):
    #     self.model = deepcopy(torch.load(path, map_location=torch.device('cpu')))

    def save(self) -> str:
        """Saving the model in the following path:
        '../experiments/run_year_month_day/models/model_name_fx.pkl

        Returns:
            String: Path
        """
        raise NotImplementedError
    

        
