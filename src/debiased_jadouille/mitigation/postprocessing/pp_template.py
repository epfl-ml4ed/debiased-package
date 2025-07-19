import numpy as np
import pandas as pd

from debiased_jadouille.predictors.predictor import Predictor
from debiased_jadouille.mitigation.postprocessing.postprocessor import PostProcessor

class PostProcessor(PostProcessor):
    """post-processing

    References:
        
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = ' et al.'
        self._notation = ''
        self._postprocessor_settings = self._settings['postprocessors']['']
        self._information = {}

    def transform(
            self, model: Predictor, features:list, ground_truths: list, predictions: list,
            probabilities: list, demographics: list
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
        raise NotImplementedError

    def fit_transform( 
            self, model: Predictor, features:list, ground_truths: list, predictions: list,
            probabilities: list, demographics: list
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
        demographic_attributes = self.extract_demographics(demographics)
        raise NotImplementedError #return multicalibrate_predictions, multicalibrate_probabilities
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
