import os
import yaml
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Tuple

from sklearn.model_selection import StratifiedKFold

from predictors.predictor import Predictor
from crossvalidation.splitters.splitter import Splitter
from crossvalidation.crossvalidator import CrossValidator
from crossvalidation.scorers.scorer import Scorer
from crossvalidation.gridsearches.gridsearch import GridSearch

from copy import deepcopy


class NonNestedXVal(CrossValidator):
    """Implements nested cross validation: 
            For each fold, get train and test set:
                split the train set into a train and validation set
                perform gridsearch on the chosen model, and choose the best model according to the validation set
                Predict the test set on the best model according to the gridsearch
            => Outer loop computes the performances on the test set
            => Inner loop selects the best model for that fold

    Args:
        XValidator (XValidators): Inherits from the model class
    """
    
    def __init__(
        self, settings:dict, gridsearch:GridSearch, gridsearch_splitter: Splitter, 
        outer_splitter: Splitter, model:Predictor, scorer:Scorer
    ):
        super().__init__(settings, model, scorer)
        print('HEYA')
        self._name = 'transfernonnested cross validator'
        self._notation = 'trnonnested_xval'

        
        self._gs_splitter = gridsearch_splitter # To create the folds within the gridsearch from the train set 
        self._outer_splitter = outer_splitter(settings) # to create the folds between development and test
        
        self._scorer = scorer(settings)
        self._gridsearch = gridsearch
        
        #debug
        self._model = model
        
    def crossval(self, x:list, y:list, demographics:list) -> dict:
        
        results = {}
        results['dataset'] = deepcopy(self._settings['pipeline'])
        results['settings'] = self._settings
        results['optim_scoring'] = self._xval_settings['nested_xval']['optim_scoring'] #debug
        for f, (train_index, test_index) in enumerate(self._outer_splitter.split(x, y, demographics)):
            results[f] = {}
            results[f]['train_index'] = train_index
            results[f]['test_index'] = test_index
            
            # division train / test
            x_train = [x[xx] for xx in train_index]
            y_train = [y[yy] for yy in train_index]
            dem_train = [demographics[dd] for dd in train_index]
            x_test = [x[xx] for xx in test_index]
            y_test = [y[yy] for yy in test_index]
            dem_test = [demographics[dd] for dd in test_index]
            
            # Inner loop
            model = self._model(self._settings)
            model.set_outer_fold(f)
            results[f]['loss_history'] = model.fit(x_train, y_train, x_val=x_train, y_val=y_train)
            results[f]['best_estimator'] = model.save_fold(f)

            # Predict
            y_pred, y_test = model.predict(x_test, y_test)
            y_proba = model.predict_proba(x_test)
            test_results = self._scorer.get_scores(y_test, y_pred, y_proba, dem_test)
            results[f]['y_pred'] = y_pred
            results[f]['y_proba'] = y_proba
            results[f]['y_test'] = y_test
            results[f].update(test_results)
            
            print(' Best Results on outer fold: {}'.format(test_results))
            self._model_notation = model.get_notation()
            self.save_results(results)
            if self._settings['pipeline']['dataset'] in ['xuetangx', 'eedi', 'eedi2']:
                break
        return results
    
    def save_results(self, results):
        path = '{}/results/'.format(
            self._settings['experiment']['name']
        )
        os.makedirs(path, exist_ok=True)

        path += '{}_m{}_modelseeds{}_all_folds.pkl'.format(
            self._notation, self._model_notation,
            self._settings['seeds']['model']
        )
        with open(path, 'wb') as fp:
            pickle.dump(results, fp)
            
            