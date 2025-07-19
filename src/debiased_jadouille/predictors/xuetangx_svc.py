import os
import copy
import pickle
import numpy as np

import tqdm
from typing import Tuple

from debiased_jadouille.predictors.predictor import Predictor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class StandardScalingSVCClassifier(Predictor):
    """This class implements the SVC with standard scaled features as described in Basnet, R. B., Johnson, C., & Doleck, T. (2022). Dropout prediction in Moocs using deep learning and machine learning. Education and Information Technologies, 27(8), 11499-11513.
    This is implemented specifically for the xuetangx dataset

    Found through the paper citing the original paper
    Args:
        Model (Model): inherits from the model class
    """
    
    def __init__(self, kernel, C):
        super().__init__({'kernel':kernel, 'C': C})
        self._kernel = kernel
        self._C = C

    def _format_final(self, x:list, y:list) -> Tuple[list, list]:
        return np.array(x), np.array(y)
    
    def _format_features(self, x:list) -> list:
        return np.array(x)
    
    def _init_model(self):
        self._set_seed(193)
        self.model = make_pipeline(
            StandardScaler(),
            SVC(kernel=self._kernel, C=self._C, probability=True)
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
