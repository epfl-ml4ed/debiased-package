import os
import yaml
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Tuple

from sklearn.model_selection import StratifiedKFold

from predictors.predictor import Predictor
from mitigation.postprocessing.postprocessor import PostProcessor
from crossvalidation.splitters.splitter import Splitter
from crossvalidation.crossvalidator import CrossValidator
from crossvalidation.scorers.scorer import Scorer
from crossvalidation.gridsearches.gridsearch import GridSearch

from copy import deepcopy
# from utils.config_handler import ConfigHandler

class SyncPostNestedXVal(CrossValidator):
    """Implements nested cross validation for the classification model and the post-processing: 
            For each fold, get train and test set:
                split the train set into a train and validation set
                perform gridsearch on the chosen post-processor, and choose the best model according to the validation set
            => Outer loop computes the performances on the test set, predictions and models have been pre-computed in the baselines
            => Inner loop selects the best postprocessor for that fold

    Args:
        XValidator (XValidators): Inherits from the model class
    """
    
    def __init__(
        self, settings:dict, gridsearch:GridSearch, gridsearch_splitter: Splitter, postprocessor: PostProcessor,  scorer:Scorer
    ):
        super().__init__(settings, postprocessor, scorer)
        self._name = 'sync-nested postprocessing cross validator'
        self._notation = 'syncnestedpostproc_xval'
        self._gs_splitter = gridsearch_splitter # To create the folds within the gridsearch from the train set 
        
        self._scorer = scorer(settings)
        self._gridsearch = gridsearch
        
        self._postprocessor = postprocessor
        
    def _init_gs(self, fold):
        self._scorer.set_optimiser_function(self._xval_settings['post_cval']['optim_scoring'])
        self._gs = self._gridsearch(
            postprocessor = self._postprocessor,
            grid=self._xval_settings['post_cval']['paramgrid'],
            scorer=self._scorer,
            splitter = self._gs_splitter,
            settings=self._settings,
            outer_fold=fold,
        )

        
    def crossval(
        self, models: list, 
        train_features:list, train_ground_truths: list, 
        test_features: list, test_ground_truths: list,
        train_demographics: list, test_demographics: list
    ) -> dict:
        """Does the crossvalidation for the post processing.
        Since it is assuming that post-processing methods are applied on *already trained* models, 
        each in put arguments is a list of 10 objects, one per fold. The folds are the same as 
        the ones used to train the *already trained model*

        Args:
            model (Predictor): list of the models used, one per fold
            train_features (list): list of the training data for each fold
            train_ground_truths (list): list of the ground truths for each fold
            test_features (list): list of the test features for each fold
            test_ground_truths (list): list of the test ground truths for each fold
            train_demographics(list): list of the training demographics for each fold
            test_demographics (list): list of the test demographics for each fold 

        Returns:
            dict: _description_
        """
        results = {}
        results['dataset'] = deepcopy(self._settings['pipeline'])
        results['optim_scoring'] = self._xval_settings['post_cval']['optim_scoring'] #debug
        for f in range(self._settings['crossvalidation']['nfolds']):
            results[f] = {}

            # division train / test
            x_train = [tf for tf in train_features[f]]
            y_train = [tgt for tgt in train_ground_truths[f]]
            demographics_train = [td for td in train_demographics[f]]
            train_predictions = models[f].predict(x_train)
            train_probas = models[f].predict_proba(x_train)

            x_test = [ttf for ttf in test_features[f]]
            y_test = [ttgt for ttgt in test_ground_truths[f]]
            demographics_test = [ttd for ttd in test_demographics[f]]
            test_predictions = models[f].predict(x_test)
            test_probas = models[f].predict_proba(x_test)

            # Gridsearch
            self._init_gs(f)
            self._gs.fit(
                models[f], x_train, y_train, train_predictions, train_probas, demographics_train, f
            )
            # Predict
            y_pred, y_proba = self._gs.transform(
                models[f], x_test, y_test, test_predictions, test_probas, demographics_test
            )
            test_results = self._scorer.get_scores(y_test, y_pred, y_proba, demographics_test)
            logging.debug('    predictions: {}'.format(y_pred))
            logging.debug('    probability predictions: {}'.format(y_proba))
            
            results[f]['y_pred'] = y_pred
            results[f]['y_proba'] = y_proba
            results[f].update(test_results)
            
            results[f]['best_params'] = self._gs.get_best_model_settings()
            best_estimator = self._gs.get_best_model()
            results[f]['gridsearch_object'] = self._gs.get_path(f)
            logging.info(' best parameters: {}'.format(results[f]['best_params']))
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
            
            