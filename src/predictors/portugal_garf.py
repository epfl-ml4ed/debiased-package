import os
import copy
import pickle
import numpy as np

import tqdm
from typing import Tuple

from predictors.predictor import Predictor

from sklearn.ensemble import RandomForestClassifier
from sklearn_genetic import GAFeatureSelectionCV
from sklearn_genetic.plots import plot_fitness_evolution

class GARFClassifier(Predictor):
    """This class implements the Genetic algorithm (features selection) to Random Forest predictor as described in Farissi, A., & Dahlan, H. M. (2020, April). Genetic algorithm based feature selection with ensemble methods for student academic performance prediction. In journal of physics: Conference series (Vol. 1500, No. 1, p. 012110). IOP Publishing.
    This is implemented specifically for the math/portuguese dataset

    Found through google scholar search
    Args:
        Model (Model): inherits from the model class
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'genetic-algorithm-random-forest'
        self._notation = 'garf'
        self._model_settings = settings['predictors']['garf']
        self._fold = -1

    def update_settings(self, settings):
        self._model_settings.update(settings)
        self._settings['predictors']['garf'].update(settings)
        self._choose_fit()

    def _format_final(self, x:list, y:list) -> Tuple[list, list]:
        return np.array(x), np.array(y)
    
    def _format_features(self, x:list) -> list:
        return np.array(x)
    
    def _init_model(self):
        self._set_seed()
        self.classifier = RandomForestClassifier(
            n_estimators = self._model_settings['n_estimators'],
            max_depth = self._model_settings['max_depth']
        )
        self.model = GAFeatureSelectionCV(
                    estimator=self.classifier,
                    cv=10,
                    scoring='roc_auc',
                    population_size=self._model_settings['population_size'],
                    generations=self._model_settings['generations'],
                    n_jobs=-1,
                    keep_top_k=2,
                    elitism=True,
                    verbose=False
        )


    def init_model(self):
        self._init_model()


    def fit(self, x_train:list, y_train:list, x_val:list, y_val:list):
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
        with open('{}garf_{}.pkl'.format(path, extension), 'wb') as fp:
            pickle.dump(self, fp)
        return '{}garf_{}'.format(path, extension)
    
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