import os
import copy
import pickle
import numpy as np

import tqdm
from typing import Tuple

from debiased_jadouille.predictors.predictor import Predictor

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
    
    def __init__(self, n_estimators=5, max_depth=5, population_size=10, generations=10):
        super().__init__({
            'n_estimators': n_estimators, 'max_depth': max_depth,
            'population_size': population_size, 'generations': generations
        })
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._population_size = population_size
        self._generations = generations

    def _format_final(self, x:list, y:list) -> Tuple[list, list]:
        return np.array(x), np.array(y)
    
    def _format_features(self, x:list) -> list:
        return np.array(x)
    
    def _init_model(self):
        self._set_seed(193)
        self.classifier = RandomForestClassifier(
            n_estimators = self._n_estimators,
            max_depth = self._max_depth
        )
        self.model = GAFeatureSelectionCV(
                    estimator=self.classifier,
                    cv=10,
                    scoring='roc_auc',
                    population_size=self._population_size,
                    generations=self._generations,
                    n_jobs=-1,
                    keep_top_k=2,
                    elitism=True,
                    verbose=False
        )


    def init_model(self):
        self._init_model()

    def fit(self, x_train:list, y_train:list, x_val=[], y_val=[]):
        x_train, y_train = self._format_final(x_train, y_train)
        x_val, y_val = self._format_final(x_val, y_val)

        self._init_model()
        self.model.fit(x_train, y_train)

    def predict(self, x:list) -> list:
        test_x = self._format_features(x)
        predictions = self.model.predict(test_x)
        return predictions
    
    def predict_proba(self, x:list) -> list:
        test_x = self._format_features(x)
        predictions = self.model.predict_proba(test_x)
        return predictions
