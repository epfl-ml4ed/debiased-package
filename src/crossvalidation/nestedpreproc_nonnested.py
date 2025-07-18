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
from crossvalidation.scorers.preprocessing_scorer import PreProcessingScorer
from crossvalidation.gridsearches.gridsearch import GridSearch

from copy import deepcopy
# from utils.config_handler import ConfigHandler

class NestedPreProcXVal(CrossValidator):
    """Implements nested cross validation for the preprocessor solely, optimising for the preprocessor: 
            For each fold, get train and test set:
                split the train set into a train and validation set
                perform gridsearch on the chosen preprocessor, and choose the best model according to the validation set
            => Outer loop computes the performances on the test set
            => Inner loop selects the best preprocessor for that fold

    Args:
        XValidator (XValidators): Inherits from the model class
    """
    
    def __init__(
        self, settings:dict, gridsearch:GridSearch, gridsearch_splitter: Splitter, 
        outer_splitter: Splitter, preprocessor: PreProcessor, model:Predictor, scorer:Scorer
    ):
        super().__init__(settings, model, scorer)
        self._name = 'nested preprocessing cross validator'
        self._notation = 'nestedpreproc_xval'
        self._gs_splitter = gridsearch_splitter # To create the folds within the gridsearch from the train set 
        self._outer_splitter = outer_splitter(settings) # to create the folds between development and test
        
        self._scorer = scorer(settings)
        self._gridsearch = gridsearch
        
        #debug
        self._preprocessor = preprocessor
        self._model = model
        
    def _init_gs(self, fold):
        self._gridsearch_scorer = PreProcessingScorer(self._settings)
        self._gridsearch_scorer.set_optimiser_function(self._xval_settings['nestedpreproc_xval']['optim_scoring'])
        self._gs = self._gridsearch(
            preprocessor=self._preprocessor,
            grid=self._xval_settings['nestedpreproc_xval']['paramgrid'],
            scorer=self._gridsearch_scorer,
            splitter = self._gs_splitter,
            settings=self._settings,
            outer_fold=fold,
        )

        
    def crossval(self, x:list, y:list, demographics:list) -> dict:
        results = {}
        results['dataset'] = deepcopy(self._settings['pipeline'])
        logging.debug('x:{}, y:{}'.format(x, y))
        results['optim_scoring'] = self._xval_settings['nestedpreproc_xval']['optim_scoring'] #debug
        for f, (train_index, test_index) in enumerate(self._outer_splitter.split(x, y, demographics)):
            logging.debug('outer fold, length train: {}, length test: {}'.format(len(train_index), len(test_index)))
            logging.debug('outer fold: {}'.format(f))
            logging.info('- ' * 30)
            logging.info(' Fold {}'.format(f))
            logging.debug('    train indices: {}'.format(train_index))
            logging.debug('    test indices: {}'.format(test_index))

            results[f] = {}
            results[f]['train_index'] = train_index
            results[f]['test_index'] = test_index
            
            # division train / test
            x_train = [x[xx] for xx in train_index]
            y_train = [y[yy] for yy in train_index]
            demographics_train = [demographics[dd] for dd in train_index]
            x_test = [x[xx] for xx in test_index]
            y_test = [y[yy] for yy in test_index]
            demographics_test = [demographics[dd] for dd in test_index]

            # Pre-Processing
            self._init_gs(f)
            self._gs.fit(x_train, y_train, demographics_train, f)
            x_preprocessed, y_preprocessed, demo_preprocessed = self._gs.fit_transform(x_train, y_train, demographics_train)
            
            # Classification
            model = self._model(self._settings)
            results[f]['loss_history'] = model.fit(x_preprocessed, y_preprocessed, x_val=x_train, y_val=y_train)

            preprocessed_testdata, _, _ = self._gs.transform(x_test, y_test, demographics_test)
            y_pred, y_testt = model.predict(preprocessed_testdata, y_test)
            y_proba = model.predict_proba(preprocessed_testdata)
            test_results = self._scorer.get_scores(y_testt, y_pred, y_proba, demographics_test)
            logging.debug('    predictions: {}'.format(y_pred))
            logging.debug('    probability predictions: {}'.format(y_proba))
            
            results[f]['y_pred'] = y_pred
            results[f]['y_proba'] = y_proba
            results[f].update(test_results)
            
            results[f]['best_params'] = self._gs.get_best_model_settings()
            best_estimator = self._gs.get_best_model()
            # results[f]['best_estimator'] = best_estimator.save_fold(f)
            results[f]['gridsearch_object'] = self._gs.get_path(f)
            logging.info(' best parameters: {}'.format(results[f]['best_params']))
            # logging.info(' estimator path: {}'.format(results[f]['best_estimator']))
            logging.info(' gridsearch path: {}'.format(results[f]['gridsearch_object']))
            
            print('Best Results on outer fold: {}'.format(test_results))
            logging.info('Best Results on outer fold: {}'.format(test_results))
            self._model_notation = best_estimator.get_notation()
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
            
            