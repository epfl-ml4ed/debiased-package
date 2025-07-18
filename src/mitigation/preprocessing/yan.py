import numpy as np
import pandas as pd
from mitigation.preprocessing.notimplementable.fairbalance.fairBalance import fairBalance

from mitigation.preprocessing.preprocessor import PreProcessor

class YanPreProcessor(PreProcessor):
    """Resampling pre-processing
    SMOTE based

    References:
        Yan, S., Kao, H. T., & Ferrara, E. (2020, October). Fair class balancing: Enhancing model fairness without observing sensitive attributes. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management (pp. 1715-1724).
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'yan et al.'
        self._notation = 'yan'
        self._preprocessor_settings = self._settings['preprocessors']['yan']
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

    def _format_data(self, x, y, demographics):
        """Format the arrays as a pandas dataframe, as shown here:
        https://github.com/ShenYanUSC/Fair_Class_Balancing/blob/main/fairBalance.py
            x (_type_): features
            y (_type_): labels
            demographics (_type_): demographic attributes 
        """
        demographic_attributes = self.extract_demographics(demographics)
        sensitive_attribute = self.get_binary_protected_privileged(demographic_attributes)
        concatenate = [
            [*x[student], y[student], sensitive_attribute[student]] for student in range(len(x))
        ]
        data = pd.DataFrame(concatenate)
        feature_names = ['f_{}'.format(f_i) for f_i in range(len(x[0]))]
        columns = feature_names + ['label', 'demographic']
        data.columns = columns
        return data, feature_names

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
        data_train, feature_names = self._format_data(x_train, y_train, demo_train)

        fair_balance = fairBalance(
            data_train, feature_names, feature_names, ['demographic'], 'demographic', 'label',
            self._preprocessor_settings['clustering'], knn=self._preprocessor_settings['knn']
        )
        fair_balance.fit()
        x_sampled, y_sampled = fair_balance.generater()

        return x_sampled, y_sampled, []
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
