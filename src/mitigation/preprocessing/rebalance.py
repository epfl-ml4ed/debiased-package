from shutil import copytree

import os
import logging
import pickle
import numpy as np
import pandas as pd
from collections import Counter

# import tensorflow as tf

from shutil import copytree, rmtree
from copy import deepcopy
from typing import Tuple

from mitigation.preprocessing.preprocessor import PreProcessor

class RebalancePreProcessor(PreProcessor):
    """This is a template to test the pipelines. Through oversampling with replacement, it rebalances all demographics
    Preprocessing: 
        description
    
    References:
        apa style
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'rebalancing preprocessor'
        self._notation = 'reb_pre'
        self._preprocessor_settings = self._settings['preprocessors']['rebalance']
        self._information = {}

    def transform(self, 
        x_train: list, y_train: list, demo_train: list,
        ):
        """
        Args:
            x_train (list): training feature data 
            y_train (list): training label data
            demo_train(list): training demographics data
            x_val (list): validation feature data
            y_val (list): validation label data
            demo_val (list): validation demographics data
        """
        return x_train, y_train, demo_train
        
    def fit_transform(self, 
            x_train: list, y_train: list, demo_train: list,
            x_val: list, y_val: list, demo_val: list
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
        demographic_attributes = self.extract_demographics(demo_train)
        demographic_count = Counter(demographic_attributes)
        self._balance_index = max([demographic_count[c] for c in demographic_count]) * self._preprocessor_settings['index']
        
        demographic_attributes = self.extract_demographics(demo_train)
        x_preprocessed, y_preprocessed, demo_preprocessed = [], [], []
        for demo in demographic_attributes:
            demo_indices = [i for i in range(len(demographic_attributes)) if demographic_attributes[i] == demo]
            rebalance_indices = np.random.choice(demo_indices, size=self._balance_index, replace=True)
            x_preprocessed = [*x_preprocessed, *[x_train[ridx] for ridx in rebalance_indices]]
            y_preprocessed = [*y_preprocessed, *[y_train[ridx] for ridx in rebalance_indices]]
            demo_preprocessed = [*demo_preprocessed, *[demo_train[ridx] for ridx in rebalance_indices]]

        self._information['balance_index'] = self._balance_index
        return x_preprocessed, y_preprocessed, demo_preprocessed

    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
