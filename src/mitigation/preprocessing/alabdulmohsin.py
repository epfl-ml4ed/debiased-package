import math
import numpy as np
import pandas as pd

from mitigation.preprocessing.google_unmodified.ml_debiaser import Reduce2Binary
from mitigation.preprocessing.preprocessor import PreProcessor
from pipelines.crossvalidation_pipeline import CrossValMaker

class AlabdulmohsinPreProcessor(PreProcessor):
    """Massage the data to take out disparate impact
    Preprocessing: 
        Works for any number of classes. It debiases the data without using the demographic attribute during inference time

    Optimising:
        Demographic Parity

    References:
        Alabdulmohsin, I. M., Schrouff, J., & Koyejo, S. (2022). A reduction to binary approach for debiasing multiclass datasets. Advances in Neural Information Processing Systems, 35, 2480-2493.
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'alabdulmohsin et al.'
        self._notation = 'alabdul'
        self._preprocessor_settings = self._settings['preprocessors']['alabdulmohsin']
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

    def _one_hot_attributes(self, attributes):
        unique_attributes = [att for att in np.unique(attributes)]
        attribute_map = {ua: i for i, ua in enumerate(unique_attributes)}
        # n_ua = len(unique_attributes)

        onehot = [attribute_map[att] for att in attributes]

        # onehot = [[0 for _ in range(n_ua)] for _ in range(len(attributes))]
        # for instance in attributes:
        #     onehot[attribute_map[instance]] = 1
        return onehot

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
        # Data Prep
        demographic_attributes = self.extract_demographics(demo_train)
        demoatt = self._one_hot_attributes(demographic_attributes)
        ytrain = np.zeros((len(x_train), self._settings['pipeline']['nclasses']))
        ytrain[np.arange(len(x_train)), y_train] = 1.

        # Massaging
        self._r2b = Reduce2Binary(num_classes=self._settings['pipeline']['nclasses'])
        massaged_ys = self._r2b.fit(
            ytrain, demoatt, sgd_steps=self._preprocessor_settings['sgd_steps'],
            full_gradient_epochs=self._preprocessor_settings['full_gradient_epochs'], 
            max_admm_iter=self._preprocessor_settings['max_admm_iter']
        )
        self._information.update(self._preprocessor_settings)

        # Processed ys
        massaged_ys = [np.argmax(my) for my in massaged_ys]

        n_massaged = np.sum([int(y_train[idx] == massaged_ys[idx]) for idx in range(len(y_train))])
        print('{} instances were massaged! ({}%)'.format(n_massaged, (n_massaged/len(y_train)*100)))
        self._information['n_massaged'] = n_massaged
        return x_train, massaged_ys, demo_train
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
