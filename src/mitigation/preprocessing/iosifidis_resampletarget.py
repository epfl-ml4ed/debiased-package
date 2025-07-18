from random import random
import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from mitigation.preprocessing.preprocessor import PreProcessor

class IosifidisResamplingTargetPreProcessor(PreProcessor):
    """Resampling pre-processing
    oversampling to balance targets

    References:
        Iosifidis, V., & Ntoutsi, E. (2018). Dealing with bias via data augmentation in supervised learning scenarios. Jo Bates Paul D. Clough Robert Jäschke, 24(11).
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'iosifidis et al. - resampling'
        self._notation = 'iosifidisres'
        self._preprocessor_settings = {}
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

        ros = RandomOverSampler(random_state=self._settings['seeds']['preprocessor'], sampling_strategy='minority')
        x_sampled, y_sampled = ros.fit_resample(x_train, y_train)
        return x_sampled, y_sampled, []
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
