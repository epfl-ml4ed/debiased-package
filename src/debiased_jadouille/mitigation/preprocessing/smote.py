import math
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from debiased_jadouille.mitigation.preprocessing.preprocessor import PreProcessor
from pipelines.crossvalidation_pipeline import CrossValMaker

class SmotePreProcessor(PreProcessor):
    """Sampling

    Summary:
        SMOTE oversampling on the labels

    References:
        Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. Journal of artificial intelligence research, 16, 321-357.
    """
    
    def __init__(self, mitigating, discriminated, settings: dict):
        super().__init__(settings)
        self._name = 'chawla et al.'
        self._notation = 'chawla'
        self._preprocessor_settings = self._settings['preprocessors']['chawla']
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
        smote = SMOTE(
            random_state=self._settings['seeds']['preprocessor'],
            sampling_strategy=self._preprocessor_settings['sampling_strategy']
        )
        x_resampled, y_resampled = smote.fit_resample(x_train, y_train)
        return x_resampled, y_resampled, demo_train
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
