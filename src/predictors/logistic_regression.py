import os
import copy
import pickle
import numpy as np
import pandas as pd

import tqdm
from typing import Tuple

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from predictors.predictor import Predictor

from sklearn.linear_model import LogisticRegression

class LogisticRegressionClassifier(Predictor):
    """This class implements the logistic regression (to be trained with the "best features") as described in Gervet, T., Koedinger, K., Schneider, J., & Mitchell, T. (2020). When is deep learning the best approach to knowledge tracing?. Journal of Educational Data Mining, 12(3), 31-54 (LR) and Schmucker, R., Wang, J., Hu, S., & Mitchell, T. M. (2021). Assessing the performance of online students--new data, new approaches, improved accuracy. arXiv preprint arXiv:2109.01753 (EEDI)
    made for eedi
    Found through the paper citing the original paper

    Args:
        Model (Model): inherits from the model class
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'logistic regression'
        self._notation = 'logr'
        self._model_settings = settings['predictors']['lr']
        self._fold = -1

    def update_settings(self, settings):
        self._model_settings.update(settings)
        self._settings['predictors']['lr'].update(settings)
        self._choose_fit()

    def _format_final(self, x:list, y:list) -> Tuple[list, list]:
        # self.scaler = MinMaxScaler()
        data = pd.DataFrame(x)
        # self.scaler.fit(data)
        # data = self.scaler.transform(data)
        return np.array(data), np.array(y)
    
    def _format_features(self, x:list) -> list:
        data = pd.DataFrame(x)
        # data = self.scaler.transform(data)
        return np.array(data)
    
    def _init_model(self):
        self._set_seed()
        self.model = LogisticRegression(
            penalty = self._model_settings['penalty'],
            C=self._model_settings['C'],
            solver=self._model_settings['solver']
        )

    def init_model(self):
        self._init_model()

    def fit(self, x_train:list, y_train:list, x_val:list, y_val:list):
        

        #code to make a student-wise train/test split
        #user_ids = x_train[:, 0].toarray().flatten()
        #users_train = train_df["user_id"].unique()
        #users_test = test_df["user_id"].unique()
        #train = x_train[np.where(np.isin(user_ids, users_train))]
        #test = x_train[np.where(np.isin(user_ids, users_test))]
        
        x_train, y_train = self._format_final(x_train, y_train)
        x_val, y_val = self._format_final(x_val, y_val)

        self._init_model()
        self.model.fit(x_train, y_train)
        self._fold += 1

    def predict(self, x:list, y:list) -> list:
        test_x = self._format_features(x)
        predictions = self.model.predict(test_x)
        return predictions, y
    
    def predict_proba(self, x:list) -> list:
        test_x = self._format_features(x)
        predictions = self.model.predict_proba(test_x)
        return predictions

    def save(self, extension='') -> str:
        path = '{}/models/'.format(self._settings['experiment']['name'])
        os.makedirs(path, exist_ok=True)
        with open('{}lr_{}.pkl'.format(path, extension), 'wb') as fp:
            pickle.dump(self, fp)
        return '{}lr_{}'.format(path, extension)
    
    def get_model_path(self):
        return self.model_path

    def get_path(self, fold: int) -> str:
        return self.get_path(fold)
            
    def save_fold(self, fold: int) -> str:
        return self.save(extension='fold_{}'.format(fold))

    def save_fold_early(self, fold: int) -> str:
        return self.save(extension='fold_{}_len{}'.format(
            fold, self._maxlen
        ))