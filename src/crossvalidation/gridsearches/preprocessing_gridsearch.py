import os
import copy
import pickle
import logging
import itertools

import numpy as np
import pandas as pd
from six import b

from predictors.predictor import Predictor
from mitigation.preprocessing.preprocessor import PreProcessor
from crossvalidation.splitters.splitter import Splitter
from crossvalidation.scorers.preprocessing_scorer import PreProcessingScorer
from crossvalidation.gridsearches.gridsearch import GridSearch

class PreProcessingGridSearch(GridSearch):
    """
    Supervised gridsearch where:
    for each combination:
        x_train -> model -> predictions
    retrieve best parameters

    """
    def __init__(self, preprocessor:PreProcessor, grid:dict, scorer:PreProcessingScorer, splitter:Splitter, settings:dict, outer_fold:int):
        super().__init__(preprocessor, grid, scorer, splitter, settings, outer_fold)
        """
        Args:
            preprocessor: pre-processing model
            grid: hyperparameter grid
            scorer: what object to use to compute the scores
            splitter: how to split the data
            settings: all parameters
            outer_fold: what fold of the nested cross validation we are at (for saving purposes)
        """
        self._name = 'preprocessor gridsearch'
        self._notation = 'supgs'
        self._folds = {}
        
    def fit(self, x_train:list, y_train:list, demographics: list, fold:int):
        self.loss_histories = {}
        for _, combination in enumerate(self._combinations):
            self.loss_histories['_'.join([str(c) for c in combination])] = {}
            logging.info('  Testing parameters: {}'.format(combination))
            folds = []
            fold_indices = {}
            splitter = self._splitter(self._settings)
            for f, (train_index, validation_index) in enumerate(splitter.split(x_train, y_train, demographics)):
                self.loss_histories['_'.join([str(c) for c in combination])][f] = []
                logging.debug('    inner fold, train length: {}, test length: {}'.format(len(train_index), len(validation_index)))
                
                # Split
                xx_train = [x_train[xx] for xx in train_index]
                yy_train = [y_train[yy] for yy in train_index]
                demo_train = [demographics[ti] for ti in train_index]
                x_val = [x_train[xx] for xx in validation_index]
                y_val = [y_train[yy] for yy in validation_index]
                demo_val = [demographics[di] for di in validation_index]
                
                # Pre-Processing Training
                preprocessor = self._model(self._settings)
                preprocessor.set_outer_fold(self._outer_fold)
                preprocessor.set_gridsearch_parameters(self._parameters, combination)
                preprocessor.set_gridsearch_fold(f)
                x_preprocessed, y_preprocessed, dem_preprocessed = preprocessor.fit_transform(
                    xx_train, yy_train, demo_train, x_val, y_val, demo_val
                )
                self.loss_histories['_'.join([str(c) for c in combination])][f].append(preprocessor.get_information())
                
                # Pre-Processing Evaluation
                x_pred, y_pred, dem_pred = preprocessor.transform(
                    x_val, y_val, demo_val
                )
                score = self._scoring_function(
                    x_val, y_val, demo_val, x_pred, y_pred, dem_pred
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
        preprocessor = self._model(config)
        preprocessor.set_outer_fold(self._outer_fold)
        preprocessor.set_gridsearch_parameters(self._parameters, combinations)
        print('Warning '*50)
        print('no validation in optimal inner fold')
        _ = preprocessor.fit_transform(x_train, y_train, demographics, x_train, y_train, demographics)
        # model.save(extension='best_model_f{}'.format(fold))
        self._best_model = preprocessor
        
            
    def fit_transform(self, x_val: list, y_val: list, demo_val: list):
        """ picks the best model from the gridsearch, and predicts x_test
        
        Returns x and y (useful when the format of the label is different to compute the score, and to train the model)
        """
        return self._best_model.fit_transform(x_val, y_val, demo_val, x_val, y_val, demo_val)

    def transform(self, x_val: list, y_val: list, demo_val: list):
        """ picks the best model from the gridsearch, and predicts x_test
        
        Returns x and y (useful when the format of the label is different to compute the score, and to train the model)
        """
        return self._best_model.transform(x_val, y_val, demo_val)
        