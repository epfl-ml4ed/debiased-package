
import os
import math
import pickle
import numpy as np
import pandas as pd
import cvxpy as cp

from typing import Tuple
from copy import deepcopy
from shutil import copytree, rmtree

from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from mitigation.inprocessing.inprocessor import InProcessor

class ChaiInProcessor(InProcessor):
    """inprocessing

        References:
            Chai, J., & Wang, X. (2022, June). Fairness with adaptive weights. In International Conference on Machine Learning (pp. 2853-2866). PMLR.
            https://github.com/cjy24/fairness_with_adaptive_weights
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'Chai et al.'
        self._notation = 'chai'
        self._inprocessor_settings = self._settings['inprocessors']['chai']
        self._information = {}
        self._fold = -1

    def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
        data = pd.DataFrame(x)
        demos = self.extract_demographics(demographics)
        demos = self.get_binary_protected_privileged(demos)
        data['demographics'] = demos
        data['label'] = y

        privileged = data[data['demographics'] == 0]
        protected = data[data['demographics'] == 1]

        privileged_pos = privileged[privileged['label'] == 1]
        privileged_neg = privileged[privileged['label'] == 0]
        protected_pos = protected[protected['label'] == 1]
        protected_neg = protected[protected['label'] == 0]

        privileged_pos = privileged_pos.drop(['demographics', 'label'], axis=1)
        privileged_neg = privileged_neg.drop(['demographics', 'label'], axis=1)
        protected_pos = protected_pos.drop(['demographics', 'label'], axis=1)
        protected_neg = protected_neg.drop(['demographics', 'label'], axis=1)
        return privileged_pos, privileged_neg, protected_pos, protected_neg
    
    def _format_features(self, x:list, demographics:list) -> list:
        return np.array(x)

    def _init_model(self):
        """Initiates a model with self._model
        """
        self.model = None
        raise NotImplementedError

    def init_model(self):
        self._init_model()

    def optim(self, loss, a, c):
        A = loss
        x = cp.Variable(loss.shape[0])
        objective = cp.Maximize(-a*cp.sum_squares(x)+cp.sum(cp.multiply(A,x)))
        constraints = [0 <= x, cp.sum(x) == c]
        prob = cp.Problem(objective, constraints)   
        result = prob.solve()
        for i in range(x.value.shape[0]):
            if abs(x.value[i]) < 0.01 or x.value[i] < 0:
                x.value[i] = 0
        x.value = x.value
        return x.value

    def dif(self, a, b):
        sum = 0
        for i in range(len(a)):
            sum += (a[i] - b[i]) ** 2
        sum0 = sum ** 0.5
        return sum0

    def random_search(self, 
        x_train, y_train, demographics_train, log_reg
    ):
        privileged_pos, privileged_neg, protected_pos, protected_neg = self._format_final(x_train, y_train, demographics_train)
        wi0 =  np.ones((
            len(protected_pos) + len(protected_neg) + len(privileged_pos) + len(privileged_neg)
        ))
        wi1 = wi0 + 10000000000000

        for k in (range(
            self._inprocessor_settings['min_range'],
            self._inprocessor_settings['max_range'],
            self._inprocessor_settings['n_steps']
        )):
            loss_prot_pos = wi0_prot_pos = np.ones(protected_pos.shape[0])
            loss_prot_neg = wi0_prot_neg = np.ones(protected_neg.shape[0])
            loss_priv_pos = wi0_priv_pos = np.ones(privileged_pos.shape[0])
            loss_priv_neg = wi0_priv_neg = np.ones(privileged_neg.shape[0])
            while self.dif(wi0, wi1) > 0.0001:
                clf = LogisticRegression(
                    penalty='none', dual=False, tol=1e-4, fit_intercept=False,
                    max_iter=400, solver='newton-cg', warm_start=True
                )
                clf.fit(x_train, y_train, wi0)

                for i1 in range(protected_pos.shape[0]):
                    loss_prot_pos[i1] = -math.log(log_reg.predict_proba(protected_pos)[i1][1]+1e-100)

                for i2 in range(protected_neg.shape[0]):
                    loss_prot_neg[i2] = -math.log(log_reg.predict_proba(protected_neg)[i2][0]+1e-100)

                for i3 in range(privileged_pos.shape[0]):
                    loss_priv_pos[i3] = -math.log(log_reg.predict_proba(privileged_pos)[i3][1]+1e-100)

                for i4 in range(privileged_neg.shape[0]):
                    loss_priv_neg[i4] = -math.log(log_reg.predict_proba(privileged_neg)[i4][0]+1e-100)

                wi1 = wi0
                wi0_bp_1 = self.optim(loss_prot_pos, k, c)
                wi0_bn_1 = self.optim(loss_prot_neg, k, c)
                wi0_wp_1 = self.optim(loss_priv_pos, k, c)
                wi0_wn_1 = self.optim(loss_priv_neg, k, c)

                X_train = np.concatenate([protected_pos, protected_neg, privileged_pos, privileged_neg])
                Y_train = np.concatenate((
                    [1 for _ in range(len(protected_pos))],
                    [0 for _ in range(len(protected_neg))],
                    [1 for _ in range(len(privileged_pos))],
                    [0 for _ in range(len(privileged_neg))],
                ))
                wi0 = np.concatenate((wi0_bp_1, wi0_bn_1, wi0_wp_1, wi0_wn_1))
                iter = iter + 1
            Y_predicted = classifier.predict(X_test)
            testscore_folds.append(np.sum(Y_test == Y_predicted)/len(Y_test))

            Y_predicted = classifier.predict(X_train)
            trainscore_folds.append(np.sum(Y_train == Y_predicted)/len(Y_train))




    def fit(self, 
        x_train: list, y_train: list, demographics_train: list,
        x_val:list, y_val:list, demographics_val: list
    ):
        """fits the model with the training data x, and labels y. 
        Warning: Init the model every time this function is called

        Args:
            x_train (list): training feature data 
            y_train (list): training label data
            x_val (list): validation feature data
            y_val (list): validation label data
        """
        self._init_model()
        demographic_attributes = self.extract_demographics(demographics_train)
        log_reg = LogisticRegression(penalty = 'l2', dual = False, tol = 1e-4, fit_intercept = False, max_iter=400, solver='newton-cg', warm_start = True)
        log_reg.fit(x_train, y_train)
        self.random_search(x_train, y_train, demographic_attributes, log_reg)


        raise NotImplementedError
    
    def predict(self, x: list, y, demographics: list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
            return x and y
        """
        raise NotImplementedError

    def predict_proba(self, x: list, demographics:list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
        """
        raise NotImplementedError

    def save(self, extension='') -> str:
        """Saving the model in the following path:
        '../experiments/run_year_month_day/models/model_name_fx.pkl

        Returns:
            String: Path
        """
        path = '{}/models/'.format(self._settings['experiment']['name'])
        os.makedirs(path, exist_ok=True)
        with open('{}{}_{}.pkl'.format(path, self._notation, extension), 'wb') as fp:
            pickle.dump(self._information, fp)
        return '{}{}_{}'.format(path, self._notation, extension)

    def save_fold(self, fold: int) -> str:
        return self.save(extension='fold_{}'.format(fold))

    def save_fold_early(self, fold: int) -> str:
        return self.save(extension='fold_{}_len{}'.format(
            fold, self._maxlen
        ))
    

        
