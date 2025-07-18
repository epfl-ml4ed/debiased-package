from shutil import copytree

import os
import logging
import pickle
import numpy as np
import pandas as pd

from shutil import copytree, rmtree
from copy import deepcopy
from typing import Tuple

from predictors.predictor import Predictor

class PostProcessor:
    """This implements the superclass which will be used in the machine learning pipeline
    """
    
    def __init__(self, settings: dict):
        self._settings = deepcopy(settings)
        self._experiment_name = settings['experiment']['name']
        self._random_seed = settings['seeds']['postprocessor']
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
        # torch.manual_seed(self._settings['seeds']['postprocessor'])
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
            self._postprocessor_settings[param] = combinations[i]

    def set_gridsearch_fold(self, fold:int):
        """Used to save the model under a specific name
        """
        self._gs_fold = fold

    def set_outer_fold(self, fold:int):
        self._outer_fold = fold
            
    def get_settings(self):
        return dict(self._model_settings)

    def extract_demographics(self, demographics):
        attributes = self._settings['pipeline']['attributes']['mitigating'].split('.')
        dems = ['' for _ in range(len(demographics))]
        for att in attributes:
            dems = ['{}_{}'.format(dems[i], demographics[i][att]) for i in range(len(demographics))]
        return dems

    def get_binary_protected_privileged(self, demographics_attributes):
        """Returns 1 if protected, 0 if privileged

        Args:
            demographics_attributes (_type_): as returned by the function self.extract_demographics(demographics)
        """
        discriminated_list = [str(att) for att in self._settings['pipeline']['attributes']['discriminated'].split('.')]
        return [1 if demographics_attributes[i] in discriminated_list else 0 for i in range(len(demographics_attributes))] 

    def get_binary_privileged(self, demographics_attributes):
        """Returns 1 if protected, 0 if privileged

        Args:
            demographics_attributes (_type_): as returned by the function self.extract_demographics(demographics)
        """
        discriminated_list = [str(att) for att in self._settings['pipeline']['attributes']['discriminated'].split('.')]
        return [0 if demographics_attributes[i] in discriminated_list else 1 for i in range(len(demographics_attributes))] 

    def get_protected_indices(self, demographics_attributes):
        """Returns the indices of the instances belonging to the protected (discriminated) group

        Args:
            demographics_attributes (_type_): as returned by the function self.extract_demographics(demographics)
        """
        discriminated_list = [str(att) for att in self._settings['pipeline']['attributes']['discriminated'].split('.')]
        return [i for i in range(len(demographics_attributes)) if demographics_attributes[i] in discriminated_list]

    def get_privileged_indices(self, demographics_attributes):
        """Returns the indices of the instances belonging to the protected (discriminated) group

        Args:
            demographics_attributes (_type_): as returned by the function self.extract_demographics(demographics)
        """
        discriminated_list = [str(att) for att in self._settings['pipeline']['attributes']['discriminated'].split('.')]
        return [i for i in range(len(demographics_attributes)) if demographics_attributes[i] not in discriminated_list]


    def _init_model(self):
        """Initiates a model with self._model
        """
        raise NotImplementedError

    def transform(
            self, model: Predictor, features:list, ground_truths: list, predictions: list,
            probabilties: list, demographics: list
        ):
        """transform the data given the initial training data x, and labels y. 
        Warning: Init the model every time this function is called

        Args:
        """
        raise NotImplementedError

    def fit_transform(
        self, model: Predictor, features:list, ground_truths: list, predictions: list,
        probabilties: list, demographics: list
    ):
        """trains the model and transform the data given the initial training data x, and labels y. 
        Warning: Init the model every time this function is called

        Args:
            x_train (list): training feature data 
            y_train (list): training label data
            x_val (list): validation feature data
            y_val (list): validation label data
        """
        raise NotImplementedError

    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information

    def save(self) -> str:
        """Saving the model in the following path:
        '../experiments/run_year_month_day/models/model_name_fx.pkl

        Returns:
            String: Path
        """
        raise NotImplementedError
    

        
