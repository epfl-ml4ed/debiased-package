import numpy as np
import pandas as pd

from predictors.predictor import Predictor
from mitigation.postprocessing.postprocessor import PostProcessor

from mitigation.postprocessing.pleiss_repo.multicalibration import multicalibrate

class PleissPostProcessor(PostProcessor):
    """post-processing

    References:
        Pleiss, G., Raghavan, M., Wu, F., Kleinberg, J., & Weinberger, K. Q. (2017). On fairness and calibration. Advances in neural information processing systems, 30.
        https://github.com/sanatonek/fairness-and-callibration/
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'pleiss et al.'
        self._notation = 'pleiss'
        self._postprocessor_settings = self._settings['postprocessors']['pleiss']
        self._information = {}

    def oracle_trained(self, set, v_hat, omega,):
        """original oracles
        """
        r=0
        if abs(self.ps-v_hat)<2*omega:
            r = 100
        if abs(self.ps-v_hat)>4*omega:
            r = np.random.uniform(0, 1)
        if r!=100:
            r = np.random.uniform(self.ps-omega, self.ps+omega)
        return r

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
        demographic_attributes = self.extract_demographics(demographics)
        probabilities = model.predict_proba(features)
        probabilities = np.array(probabilities)[:, 1]
        
        multicalibrate_probabilities = multicalibrate(
            data=np.array(features), labels=[], predictions=probabilities,
            sensitive_features=demographic_attributes, 
            alpha=self._postprocessor_settings['alpha'], lmbda=self._postprocessor_settings['lambda'],
            oracle=self.oracle_trained
        )

        multicalibrate_predictions = np.array(multicalibrate_probabilities)[:, 1]
        return multicalibrate_predictions, multicalibrate_probabilities

    def oracle(self, set, v_hat, omega, labels):
        """original oracles
        """
        self.ps = np.mean(labels[set])
        r=0
        if abs(ps-v_hat)<2*omega:
            r = 100
        if abs(ps-v_hat)>4*omega:
            r = np.random.uniform(0, 1)
        if r!=100:
            r = np.random.uniform(ps-omega, ps+omega)
        return r

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
        probabilities = model.predict_proba(features)
        probabilities = np.array(probabilities)[:, 1]
        
        multicalibrate_probabilities = multicalibrate(
            data=np.array(features), labels=np.array(ground_truths), predictions=probabilities,
            sensitive_features=demographic_attributes, 
            alpha=self._postprocessor_settings['alpha'], lmbda=self._postprocessor_settings['lambda'],
            oracle=self.oracle
        )

        multicalibrate_predictions = np.array(multicalibrate_probabilities)[:, 1]

        return multicalibrate_predictions, multicalibrate_probabilities

    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
