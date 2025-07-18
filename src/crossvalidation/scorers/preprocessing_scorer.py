import numpy as np
import pandas as pd
import logging
from typing import Tuple

from crossvalidation.scorers.scorer import Scorer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score
from collections import Counter


class PreProcessingScorer(Scorer):
    """This class is used to create a scorer object tailored towards binary classification

    Args:
        Scorer (Scorer): Inherits from scorer
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'preprocessing scorer'
        self._notation = 'preprocscorer'
        self._score_dictionary = {
            'roc': self._get_roc,
            # 'yourownmetric': self._
        }
        
        self._croissant = { # ascending
            'roc': True,
            'fn': False,
            'yourownmetric': True
        }
        
        self._get_score_functions(settings)
        
# Performance Scorer
    def _get_roc(self, x_true:list, y_true:list, demo_true:list, x_pred:list, y_pred:list, demo_pred:list) -> float:
        if len(np.unique(y_true)) == 1:
            return -1
        # print(y_probs)
        return roc_auc_score(y_true, np.array(y_pred))


