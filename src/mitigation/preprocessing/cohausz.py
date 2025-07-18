import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from mitigation.preprocessing.peleshapley import mod_SensDetec_hsic_shapley
from mitigation.preprocessing.preprocessor import PreProcessor

class CohauszPreProcessor(PreProcessor):
    """Resampling pre-processing

    References:
        Brooks, C., Thompson, C., & Teasley, S. (2015, March). Who you are or what you do: Comparing the predictive power of demographics vs. activity patterns in massive open online courses (MOOCs). In Proceedings of the second (2015) ACM conference on learning@ scale (pp. 245-248)
        Cohausz, L., Tschalzev, A., Bartelt, C., & Stuckenschmidt, H. (2023). Investigating the Importance of Demographic Features for EDM-Predictions. International Educational Data Mining Society
        Pan, C., & Zhang, Z. (2024). Examining the Algorithmic Fairness in Predicting High School Dropouts. In Proceedings of the 17th International Conference on Educational Data Mining (pp. 262-269).
        Yu, R., Lee, H., & Kizilcec, R. F. (2021, June). Should college dropout prediction models include protected attributes?. In Proceedings of the eighth ACM conference on learning@ scale (pp. 91-100).
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'Cohausz et al.'
        self._notation = 'cohausz'
        self._preprocessor_settings = {}
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
        return self.fit_transform(x_train, y_train, demo_train, x_train, y_train, demo_train)

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
        # data          
        data = pd.DataFrame(x_train)
        if len(self._settings['pipeline']['attributes']['included']) == 0:
            for attribute in self._settings['pipeline']['attributes']['mitigating'].split('.'):
                data[attribute] = [dt[attribute] for dt in demo_train]
        else:
            columns = [c for c in data.columns]
            for demo_idx in self._settings['pipeline']['attributes']['included']:
                data = data.drop(columns[demo_idx], axis=1)

        return np.array(data), y_train, []
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
