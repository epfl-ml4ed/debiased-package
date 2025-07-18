import os
import copy
import pickle
import numpy as np

import tqdm
from typing import Tuple

from predictors.predictor import Predictor

from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier

class SmoteENNRFBoostClassifier(Predictor):
    """This class implements the smoteENN (resampling for imbalanced problem) to random forest predictor as described in Hlosta, M., Zdrahal, Z., & Zendulka, J. (2018). Are we meeting a deadline? classification goal achievement in time in the presence of imbalanced data. Knowledge-Based Systems, 160, 278-295.
    This is implemented specifically for the oulad dataset

    Found through the original paper on the dataset
    Args:
        Model (Model): inherits from the model class
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'smote-enn-randomforest'
        self._notation = 'smotennrf'
        self._model_settings = settings['predictors']['smotennrf']
        self._fold = -1

    def update_settings(self, settings):
        self._model_settings.update(settings)
        self._settings['predictors']['smotennrf'].update(settings)
        self._choose_fit()

    def _format_final(self, x:list, y:list) -> Tuple[list, list]:
        return np.array(x), np.array(y)
    
    def _format_features(self, x:list) -> list:
        return np.array(x)
    
    def _init_model(self):
        self._set_seed()
        self._rebalancer = SMOTEENN()
        self.model = RandomForestClassifier(
            n_estimators = self._model_settings['n_estimators'],
            max_depth = self._model_settings['max_depth']
        )

    def init_model(self):
        self._init_model()


    def fit(self, x_train:list, y_train:list, x_val:list, y_val:list):
        x_train, y_train = self._format_final(x_train, y_train)
        x_val, y_val = self._format_final(x_val, y_val)

        self._init_model()

        x_resampled, y_resampled = self._rebalancer.fit_resample(x_train, y_train)
        self.model.fit(x_resampled, y_resampled)
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
        with open('{}smotenn_{}.pkl'.format(path, extension), 'wb') as fp:
            pickle.dump(self, fp)
        return '{}smotenn_{}'.format(path, extension)
    
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