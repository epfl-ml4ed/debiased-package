from shutil import copytree

import os
import logging
import pickle
import numpy as np
import pandas as pd

# import tensorflow as tf
# import torch
from shutil import copytree, rmtree
from copy import deepcopy
from typing import Tuple

class PreProcessor:
    """This implements the superclass which will be used in the machine learning pipeline
    """
    
    def __init__(self, settings: dict):
        self._settings = deepcopy(settings)
        self._random_seed = 193
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
        np.random.seed(510)

    def get_settings(self):
        return dict(self._settings)

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


    def transform(self, 
        x_train: list, y_train: list, demo_train: list,
        ):
        """transform the data given the initial training data x, and labels y. 
        Warning: Does not init the model every time this function is called

        Args:
            x_train (list): training feature data 
            y_train (list): training label data
            demo_train(list): training demographics data
            x_val (list): validation feature data
            y_val (list): validation label data
            demo_val (list): validation demographics data
        """
        raise NotImplementedError

    def fit_transform(self, 
            x_train: list, y_train: list, demo_train: list,
            x_val=[], y_val=[], demo_val=[]
        ):
        """trains the model and transform the data given the initial training data x, and labels y. 
        Warning: Init the model every time this function is called

        Args:
            x_train (list): training feature data 
            y_train (list): training label data
            demo_train(list): training demographics data
            x_val (list): validation feature data
            y_val (list): validation label data
            demo_val (list): validation demographics data
        """
        raise NotImplementedError

    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
