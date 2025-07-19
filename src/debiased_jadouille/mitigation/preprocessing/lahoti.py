import math
import numpy as np
import pandas as pd

from debiased_jadouille.mitigation.preprocessing.iFair.iFair import iFair
from debiased_jadouille.mitigation.preprocessing.preprocessor import PreProcessor

class LahotiPreProcessor(PreProcessor):
    """Representation pre-processing

    References:
        Lahoti, P., Gummadi, K. P., & Weikum, G. (2019, April). ifair: Learning individually fair data representations for algorithmic decision making. In 2019 ieee 35th international conference on data engineering (icde) (pp. 1334-1345). IEEE.
    """
    
    def __init__(self, mitigating, discriminated, k=3, max_iter=1000):
        super().__init__({'mitigating': mitigating, 'discriminated': discriminated, 'k': k, 'max_iter': max_iter})
        self._k = k 
        self._max_iter = max_iter
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
        # Data Preparation
        demographic_attributes = self.extract_demographics(demo_train)
        protected_attributes = self.get_binary_protected_privileged(demographic_attributes)
        datasets = [
            x_train[i] + [protected_attributes[i]] for i in range(len(x_train))
        ]
        datasets = np.array(datasets)
        transformed_x = self.ifair_model.transform(datasets)
        return transformed_x, y_train, demo_train

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
        # ifair model
        self.ifair_model = iFair(
            self._k, max_iter=self._max_iter,
        )
        # Data Preparation
        demographic_attributes = self.extract_demographics(demo_train)
        protected_attributes = self.get_binary_protected_privileged(demographic_attributes)
        datasets = [
            x_train[i] + [protected_attributes[i]] for i in range(len(x_train))
        ]
        datasets = np.array(datasets)
        # transformation
        transformed_x = self.ifair_model.fit_transform(datasets)
        self._k = self._k
        return transformed_x, y_train, demo_train
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
