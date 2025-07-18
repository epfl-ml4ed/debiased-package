from statistics import linear_regression
import numpy as np
import pandas as pd
from collections import Counter

from scipy import stats
from mitigation.preprocessing.preprocessor import PreProcessor

class LiPreProcessor(PreProcessor):
    """Debugging the data

    References:
        Li, Y., Meng, L., Chen, L., Yu, L., Wu, D., Zhou, Y., & Xu, B. (2022, May). Training data debugging for the fairness of machine learning software. In Proceedings of the 44th International Conference on Software Engineering (pp. 2215-2227).
        https://github.com/fairnesstest/LTDD/tree/main/LTDD
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'li et al.'
        self._notation = 'li'
        self._preprocessor_settings = self._settings['preprocessors']['calders']
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
        data = pd.DataFrame(x_train)
        demos = self.extract_demographics(demo_train)
        demos = self.get_binary_privileged(demos)
        data['demographics'] = demos

        for i_col, col in enumerate(self._column_u):
            data[col] = data[col] - self.linear_regression(data['demographics'], self._slopes[i_col], self._intercepts[i_col])
        data = data.drop('demographics', axis=1)
        new_x = [xx for xx in np.array(data)]
        return new_x, y_train, demo_train

    def linear_regression(self, x, slope, intercept):
        return x * slope + intercept

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
        data = pd.DataFrame(x_train)
        demos = self.extract_demographics(demo_train)
        demos = self.get_binary_privileged(demos)
        data['demographics'] = demos

        self._slopes = []
        self._intercepts = []
        r_values = []
        p_values = []
        self._column_u = []
        flag = 0
        ce = []
        times = 0
        for i in data.columns:
            flag += 1
            if i != 'demographics':
                slope, intercept, rvalue, pvalue, stderr = stats.linregress(data['demographics'], data[i])
                r_values.append(rvalue)
                p_values.append(pvalue)
                if pvalue < 0.05:
                    times = times + 1
                    self._column_u.append(i)
                    ce.append(flag)
                    self._slopes.append(slope)
                    self._intercepts.append(intercept)
                    data[i] = data[i] - self.linear_regression(data['demographics'], slope, intercept)
        data = data.drop('demographics', axis=1)
        new_x = [xx for xx in np.array(data)]
        return new_x, y_train, demo_train
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
