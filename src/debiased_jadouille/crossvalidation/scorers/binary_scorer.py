import numpy as np
import pandas as pd
import logging
from typing import Tuple

from debiased_jadouille.crossvalidation.scorers.scorer import Scorer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score
from collections import Counter


class BinaryClfScorer(Scorer):
    """This class is used to create a scorer object tailored towards binary classification

    Args:
        Scorer (Scorer): Inherits from scorer
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'binary classification scorer'
        self._notation = '2clfscorer'
        self._score_dictionary = {
            'accuracy': self._get_accuracy,
            'balanced_accuracy': self._get_balanced_accuracy,
            'precision': self._get_precision,
            'recall': self._get_recall,
            'roc': self._get_roc,
            'tp': self._get_tp_rate,
            'fp': self._get_fp_rate,
            'fn': self._get_fn_rate,
            'rmse': self._get_rmse
        }
        
        self._croissant = {
            'accuracy': True,
            'balanced_accuracy': True,
            'precision': True,
            'recall': True,
            'roc': True,
            'tp': False,
            'fp': False,
            'fn': False
        }
        
        self._get_score_functions(settings)

    def get_score(self, score_name, indices, ytrue, ypred, yprobs, demos):
        return self._score_dictionary[score_name](
            [ytrue[idx] for idx in indices], 
            [ypred[idx] for idx in indices], 
            [yprobs[idx] for idx in indices], 
            []
           )
        
# Performance Scorer
    def _get_accuracy(self, y_true: list, y_pred: list, yprobs: list, demographics:list) -> float:
        return accuracy_score(y_true, y_pred)
    
    def _get_balanced_accuracy(self, y_true: list, y_pred: list, yprobs: list, demographics:list) -> float:
        return balanced_accuracy_score(y_true, y_pred)
    
    def _get_precision(self, y_true: list, y_pred: list, yprobs: list, demographics:list) -> float:
        return precision_score(y_true, y_pred)
    
    def _get_recall(self, y_true: list, y_pred: list, yprobs: list, demographics:list) -> float:
        return recall_score(y_true, y_pred)
    
    def _get_roc(self, y_true: list, y_pred: list, y_probs: list, demographics:list) -> float:
        if len(np.unique(y_true)) == 1:
            return -1
        if len(np.unique(y_true)) > 2: 
            return roc_auc_score(y_true, np.array(y_probs), average='macro', multi_class='ovo')
        return roc_auc_score(y_true, np.array(y_probs)[:, 1])

    def _get_rmse(self, y_true: list, y_pred: list, y_probs: list, demographics:list) -> float:
        return np.sum(
            np.abs(np.array(y_true) - np.array([yp[1] for yp in y_probs]))
        ) / len(y_true)

    def _get_tp_rate(self, y_true:list, y_pred:list, y_probs:list, demographics:list) -> float:
        try:
            positive = [i for i in range(len(y_true)) if y_true[i] == 1]
            yt = np.array([y_true[i] for i in positive])
            # print(y_pred)
            yp = np.array([y_pred[i] for i in positive])
            s = sum(yt == yp) / len(positive)
        except ZeroDivisionError:
            s = -1
        return s

    def _get_fp_rate(self, y_true:list, y_pred:list, y_probs:list, demographics:list) -> float:
        try:
            negatives = [i for i in range(len(y_true)) if y_true[i] == 0]
            yf = np.array([y_true[i] for i in negatives])
            yp = np.array([y_pred[i] for i in negatives])
            s = sum(yf != yp) / len(negatives)
        except ZeroDivisionError:
            s = -1
        return s

    def _get_fn_rate(self, y_true:list, y_pred:list, y_probs:list, demographics:list) -> float: 
        try:
            pos_idx = [i for i in range(len(y_true)) if y_true[i] == 1]
            ps = len(pos_idx)
            tps = len([y_pred[idx] for idx in pos_idx if y_pred[idx] == 1])
            fns = ps - tps
            s = fns / (fns + tps)
        except ZeroDivisionError:
            s = -1
        return s

    
    def get_scores(self, y_true: list, y_pred: list, y_probs: list, demographics:list) -> dict:
        scores = {}
        for score in self._scorers:
            scores[score] = self._scorers[score](y_true, y_pred, y_probs, demographics)
            
        return scores

