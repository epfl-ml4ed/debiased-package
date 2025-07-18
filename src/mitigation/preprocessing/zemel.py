import math
import numpy as np
import pandas as pd

from sklearn import preprocessing
import scipy.optimize as optim
from mitigation.preprocessing.fair_representation_learning.helpers import *
from mitigation.preprocessing.preprocessor import PreProcessor
from pipelines.crossvalidation_pipeline import CrossValMaker

class ZemelPreProcessor(PreProcessor):
    """Maps data to another space such that similar majority/minority points get similar outcome, all the while obfuscating the private data
    Preprocessing: 
        Learns a mapping through which sensitive data is obfuscated

    Optimising:
        statistical Parity

    References:
        Zemel, R., Wu, Y., Swersky, K., Pitassi, T., & Dwork, C. (2013, May). Learning fair representations. In International conference on machine learning (pp. 325-333). PMLR.
        Github Repo: https://github.com/zjelveh/learning-fair-representations/blob/master/lfr.py
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'zemel et al'
        self._notation = 'zemel'
        self._preprocessor_settings = self._settings['preprocessors']['zemel']
        self._information = {'k': self._preprocessor_settings['k']}

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
        protected_indices = self.get_protected_indices(demographic_attributes)
        privileged_indices = self.get_privileged_indices(demographic_attributes)
        assert len(x_train) == len(y_train) and len(y_train) == len(demo_train) and len(demo_train) == len(demographic_attributes)

        data = preprocessing.scale(x_train)
        sensitive_data = np.array([data[prot_idx] for prot_idx in protected_indices])
        nonsensitive_data = np.array([data[priv_idx] for priv_idx in privileged_indices])
        sensitive_y = np.array([y_train[prot_idx] for prot_idx in protected_indices])
        nonsensitive_y = np.array([y_train[priv_idx] for priv_idx in privileged_indices])

        representation = LFR(
            self.final_residuals[0], sensitive_data, nonsensitive_data, sensitive_y, nonsensitive_y,
            self._preprocessor_settings['k'], 1e-4, 0.1, 1000, 1
        )

        representation_y = [int(y>0.5) for y in representation[0]] + [int(y>0.5) for y in representation[1]]
        representation_x = [x for x in representation[2]] + [x for x in representation[3]]
        representation_demo = [demo_train[prot_idx] for prot_idx in protected_indices] + [demo_train[priv_idx] for priv_idx in privileged_indices]
        return representation_x, representation_y, representation_demo


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

        # Data Preparation
        demographic_attributes = self.extract_demographics(demo_train)
        protected_indices = self.get_protected_indices(demographic_attributes)
        privileged_indices = self.get_privileged_indices(demographic_attributes)
        assert len(x_train) == len(y_train) and len(y_train) == len(demo_train) and len(demo_train) == len(demographic_attributes)

        data = preprocessing.scale(x_train)
        sensitive_data = np.array([data[prot_idx] for prot_idx in protected_indices])
        nonsensitive_data = np.array([data[priv_idx] for priv_idx in privileged_indices])
        sensitive_y = np.array([y_train[prot_idx] for prot_idx in protected_indices])
        nonsensitive_y = np.array([y_train[priv_idx] for priv_idx in privileged_indices])

        residuals = np.random.uniform(
            size=data.shape[1] * 2 + self._preprocessor_settings['k'] + data.shape[1] * self._preprocessor_settings['k']
        )

        bnd = []
        for i, k2 in enumerate(residuals):
            if i < data.shape[1] * 2 or i >= data.shape[1] * 2 + self._preprocessor_settings['k']:
                bnd.append((None, None))
            else:
                bnd.append((0, 1))
        
        self.final_residuals = optim.fmin_l_bfgs_b(
            LFR, x0=residuals, epsilon=1e-5, 
            args=(
                sensitive_data, nonsensitive_data, sensitive_y, nonsensitive_y,
                self._preprocessor_settings['k'], 1e-4, 0.1, 1000, 0
            ), bounds=bnd, approx_grad=True, maxfun=150000, maxiter=150000
        )

        representation = LFR(
            self.final_residuals[0], sensitive_data, nonsensitive_data, sensitive_y, nonsensitive_y,
            self._preprocessor_settings['k'], 1e-4, 0.1, 1000, 1
        )

        representation_y = []
        threshold = 0.5
        while((representation_y==[] or len(np.unique(representation_y))==1) and threshold<=1):
            representation_y = [int(y>threshold) for y in representation[0]] + [int(y>1-threshold) for y in representation[1]]
            representation_x = [x for x in representation[2]] + [x for x in representation[3]]
            representation_demo = [demo_train[prot_idx] for prot_idx in protected_indices] + [demo_train[priv_idx] for priv_idx in privileged_indices]
        return representation_x, representation_y, representation_demo
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
