import os
import yaml
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Tuple

from sklearn.model_selection import StratifiedKFold

from predictors.predictor import Predictor
from mitigation.preprocessing.preprocessor import PreProcessor
from crossvalidation.splitters.splitter import Splitter
from crossvalidation.crossvalidator import CrossValidator
from crossvalidation.scorers.scorer import Scorer
from crossvalidation.gridsearches.gridsearch import GridSearch

from copy import deepcopy


class NonNestedPreProcessingXVal(CrossValidator):
    """Implements non nested cross validation with pre-processing: 
            For each fold, get train and test set:
                split the train set into a train and validation set
                for each fold, fit-transform the train data, and feed it to the model

    Args:
        XValidator (XValidators): Inherits from the model class
    """
    
    def __init__(
        self, settings:dict, gridsearch:GridSearch, gridsearch_splitter: Splitter, 
        outer_splitter: Splitter, preprocessor: PreProcessor, model:Predictor, scorer:Scorer
    ):
        super().__init__(settings, model, scorer)
        self._name = 'transfernonnested cross validator'
        self._notation = 'trnonnested_xval'
        self._outer_splitter = outer_splitter(settings)

        
        self._scorer = scorer(settings)
        self._gridsearch = gridsearch
        
        self._preprocessor = preprocessor
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
            print('test')
            print(np.unique(y_test))
            print(np.unique(y_train))

            # Preprocessing
            preprocessor = self._preprocessor(self._settings)
            preprocessor.set_outer_fold(f)
            x_preprocessed, y_preprocessed, dem_preprocessed = preprocessor.fit_transform(
                x_train, y_train, dem_train, x_test, y_test, dem_test
            )
            results[f]['preprocessor_info'] = preprocessor.get_information()
            print(np.unique(y_preprocessed))
            # Model Training
            model = self._model(self._settings)
            model.set_outer_fold(f)
            results[f]['loss_history'] = model.fit(x_preprocessed, y_preprocessed, x_val=x_train, y_val=y_train)
            results[f]['best_estimator'] = model.save_fold(f)

            # Predict
            preprocessed_testdata, _, _ = preprocessor.transform(x_test, y_test, dem_test)
            y_pred, y_test = model.predict(preprocessed_testdata, y_test)
            y_proba = model.predict_proba(preprocessed_testdata)
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

        path += 'preproc_{}_m{}_modelseeds{}_all_folds.pkl'.format(
            self._notation, self._model_notation,
            self._settings['seeds']['model']
        )
        with open(path, 'wb') as fp:
            pickle.dump(results, fp)
            
            