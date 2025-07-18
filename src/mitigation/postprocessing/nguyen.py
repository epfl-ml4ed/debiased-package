import numpy as np
import pandas as pd

import sklearn.gaussian_process as gp
from scipy.linalg import cholesky, cho_solve, solve_triangular

from predictors.predictor import Predictor
from mitigation.postprocessing.postprocessor import PostProcessor
from mitigation.postprocessing.nguyen_repo import fcgp

class NguyenPostProcessor(PostProcessor):
    """post-processing

    References:
        Nguyen, D., Gupta, S., Rana, S., Shilton, A., & Venkatesh, S. (2021). Fairness improvement for black-box classifiers with Gaussian process. Information Sciences, 576, 542-556.
        https://github.com/nphdang/FCGP/blob/main/individual_fairness.py
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'nguyen et al.'
        self._notation = 'nguyen'
        self._postprocessor_settings = self._settings['postprocessors']['nguyen']
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
        protected = self.get_binary_protected_privileged(demographic_attributes)
        kernel = 1.0 * gp.kernels.RBF(length_scale=1.0)
        model_gp_validation = gp.GaussianProcessRegressor(
            kernel=kernel,
            alpha=self._postprocessor_settings['noise_gp_lower'],
            n_restarts_optimizer=5,
            normalize_y=False
        )
        xp_func = np.array(features)
        yp_func = np.array([y[0] for y in probabilities])
        model_gp_validation.fit(xp_func, yp_func)

        # no of noise_gp to search
        num_noise_gp = self._postprocessor_settings['budget']

        # search optimal noise_gp on validation set such that it maximizes score
        # (i.e. it maximizes both accuracy and fairness)
        score_arr = np.zeros(num_noise_gp)
        noise_gp_arr = np.zeros_like(score_arr)
        relabel_bo = []
        cnt = 0
        # iterate through possible noise_gp
        for noise in np.linspace(self._postprocessor_settings['noise_gp_lower'], self._postprocessor_settings['noise_gp_upper'], num_noise_gp):
            print("cnt: {}".format(cnt))
            print("current noise_gp: {}".format(round(noise, 4)))
            # use current noise_gp to build GP and relabel samples in validation set
            score_arr[cnt], relabel_bo[cnt] = fcgp.objective_function(
                features, model_gp_validation, probabilities,
                ground_truths, protected,
                [noise], self._postprocessor_settings['noise_gp_lower']
            )
            noise_gp_arr[cnt] = noise
            cnt += 1
        bestx_idx = np.argmax(score_arr)
        gen_pred_validation_round = relabel_bo[bestx_idx]
        # reformat gen_pred_validation_round as same as y_valid
        gen_pred_validation_round = np.array(gen_pred_validation_round).reshape(-1, 1)
        print('predictions')
        print(gen_pred_validation_round)
        return gen_pred_validation_round, gen_pred_validation_round

    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
