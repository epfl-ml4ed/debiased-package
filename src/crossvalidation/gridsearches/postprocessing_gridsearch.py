import os
import copy
import pickle
import logging
import itertools

import numpy as np
import pandas as pd
from six import b

from predictors.predictor import Predictor
from mitigation.postprocessing.postprocessor import PostProcessor
from crossvalidation.splitters.splitter import Splitter
from crossvalidation.scorers.scorer import Scorer
from crossvalidation.gridsearches.gridsearch import GridSearch

class PostProcessingGridSearch(GridSearch):
    """
    Supervised gridsearch where:
    for each combination:
        x_train -> model (trained) -> post processing (gridsearch) -> corrected predictions
    retrieve best parameters

    """
    def __init__(self, postprocessor: PostProcessor, grid:dict, scorer:Scorer, splitter:Splitter, settings:dict, outer_fold:int):
        super().__init__(postprocessor, grid, scorer, splitter, settings, outer_fold)
        """
        Args:
            postprocessor: pre-processing model type (not instantiated)
            grid: hyperparameter grid
            scorer: what object to use to compute the scores
            splitter: how to split the data
            settings: all parameters
            outer_fold: what fold of the nested cross validation we are at (for saving purposes)
        """
        self._name = 'postprocessor gridsearch'
        self._notation = 'postsupgs'
        self._folds = {}
        
    def fit(
            self, predictor: Predictor, features:list, ground_truths: list, predictions: list,
            probabilities: list, demographics: list, fold:int
        ):
        self.loss_histories = {}
        for _, combination in enumerate(self._combinations):
            self.loss_histories['_'.join([str(c) for c in combination])] = {}
            logging.info('  Testing parameters: {}'.format(combination))
            folds = []
            fold_indices = {}
            splitter = self._splitter(self._settings)
            for f, (train_index, validation_index) in enumerate(splitter.split(features, ground_truths, demographics)):
                self.loss_histories['_'.join([str(c) for c in combination])][f] = {'postprocessing': [], 'classification': []}
                logging.debug('    inner fold, train length: {}, test length: {}'.format(len(train_index), len(validation_index)))
                
                # Split
                xx_train = [features[xx] for xx in train_index]
                yy_train = [ground_truths[yy] for yy in train_index]
                demo_train = [demographics[ti] for ti in train_index]
                prediction_train = [predictions[pt] for pt in train_index]
                probabilities_train = [probabilities[pb] for pb in train_index]

                x_val = [features[xx] for xx in validation_index]
                y_val = [ground_truths[yy] for yy in validation_index]
                demo_val = [demographics[di] for di in validation_index]
                prediction_val = [predictions[pt] for pt in validation_index]
                probabilities_val = [probabilities[pb] for pb in validation_index]

                # Post Processing Training
                postprocessor = self._model(self._settings)
                postprocessor.set_outer_fold(self._outer_fold)
                postprocessor.set_gridsearch_parameters(self._parameters, combination)
                postprocessor.set_gridsearch_fold(f)
                ytrain_predict_postprocessed, ytrain_proba_postprocessed = postprocessor.fit_transform(
                    predictor, xx_train, yy_train, prediction_train, probabilities_train, demo_train
                )
                
                # Classification Evaluation
                ytest_predict_postprocessed, ytest_proba_postprocessed = postprocessor.transform(
                    predictor, x_val, y_val, prediction_val, probabilities_val, demo_val
                )
                score = self._scoring_function(
                    y_val, ytest_predict_postprocessed, ytest_proba_postprocessed, demo_val
                )
                logging.info('    Score for fold {}: {} {}'.format(f, score, self._scoring_name))
                folds.append(score)
                fold_indices[f] = {
                    'train': train_index,
                    'validation': validation_index
                }
                if self._settings['pipeline']['dataset'] in ['xuetangx', 'eedi', 'eedi2']:
                    break

            self._add_score(combination, folds, fold_indices)
            self.save(fold)
            
        best_parameters = self.get_best_model_settings()
        combinations = []
        for param in self._parameters:
            combinations.append(best_parameters[param])
            
        config = copy.deepcopy(self._settings)
        postprocessor = self._model(config)
        postprocessor.set_outer_fold(self._outer_fold)
        postprocessor.set_gridsearch_parameters(self._parameters, combinations)
        postprocessor.fit_transform(predictor, xx_train, yy_train, prediction_train, probabilities_train, demo_train)
        self._best_model = postprocessor
        
            
    def transform(self, 
        model: Predictor, x_val:list, y_val:list, prediction_val:list, probabilities_val:list, demo_val:list
    ) -> list:
        """ picks the best model from the gridsearch, and predicts x_test
        
        Returns x and y (useful when the format of the label is different to compute the score, and to train the model)
        """
        return self._best_model.transform(model, x_val, y_val, prediction_val, probabilities_val, demo_val)
        
        
    def fit_transform(self, 
        model: Predictor, x_val:list, y_val:list, prediction_val:list, probabilities_val:list, demo_val:list
    ) -> list:
        return self._best_model.fit_transform(model, x_val, y_val, prediction_val, probabilities_val, demo_val)
        