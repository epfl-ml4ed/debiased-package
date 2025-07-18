import numpy as np
import pandas as pd

from crossvalidation.scorers.fairness_binary_scorer import BinaryFairnessScorer
from predictors.predictor import Predictor
from mitigation.postprocessing.postprocessor import PostProcessor

class KamiranPostProcessor(PostProcessor):
    """post-processing

    References:
        Kamiran, F., Karim, A., & Zhang, X. (2012, December). Decision theory for discrimination-aware classification. In 2012 IEEE 12th international conference on data mining (pp. 924-929). IEEE.
        https://github.com/Trusted-AI/AIF360/blob/main/aif360/algorithms/postprocessing/reject_option_classification.py
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'kamiran post et al.'
        self._notation = 'kamiranpost'
        self._postprocessor_settings = self._settings['postprocessors']['kamiran']
        self._information = {}

        self._scorer = BinaryFairnessScorer(settings)

    def transform(
            self, model: Predictor, features:list, ground_truths: list, predictions: list,
            probabilities: list, demographics: list
        ):
        """
        Args:
            x_train (list): training feature data 
            y_train (list): training label data
            demo_train(list): training demographics data
            x_val (list): validation feature data
            y_val (list): validation label data
            demo_val (list): validation demographics data
        """
        demographic_attributes = self.extract_demographics(demographics)
        best_predictions = self.predict(probabilities, demographic_attributes, self.best_classification_threshold, self.best_ROC_margin)
        return best_predictions, probabilities

    def fit_transform( 
            self, model: Predictor, features:list, ground_truths: list, predictions: list,
            probabilities: list, demographics: list
        ):
        """trains the model and transform the data given the initial training data x, and labels y. 
        Warning: Init the model every time this function is called

        Args:
            x_train (list): training feature data 
            y_train (list): training label data
            demo_train(list): training demographics data
            x_val (list): validation feature data
            y_val (list): validation label data
            demo_val (list): validation demographics data
        """
        demographic_attributes = self.extract_demographics(demographics)
        
        fair_metric_arr = np.zeros(self._postprocessor_settings['num_class_thresholds']*self._postprocessor_settings['num_ROC_margin'])
        balanced_acc_arr = np.zeros_like(fair_metric_arr)
        ROC_margin_arr = np.zeros_like(fair_metric_arr)
        class_thresh_arr = np.zeros_like(fair_metric_arr)

        cnt = 0
        # Iterate through class thresholds
        for class_thresh in np.linspace(self._postprocessor_settings['low_threshold'],
                                        self._postprocessor_settings['high_threshold'],
                                        self._postprocessor_settings['num_class_thresholds']):
            if class_thresh <= 0.5:
                low_ROC_margin = 0.0
                high_ROC_margin = class_thresh
            else:
                low_ROC_margin = 0.0
                high_ROC_margin = (1.0-class_thresh)

            # Iterate through ROC margins
            for ROC_margin in np.linspace(
                                low_ROC_margin,
                                high_ROC_margin,
                                self._postprocessor_settings['num_ROC_margin']):
        
                # Predict using the current threshold and margin
                ROC_margin_arr[cnt] = ROC_margin
                dataset_transf_pred = self.predict(probabilities, demographic_attributes, class_thresh, ROC_margin)
                balanced_acc_arr[cnt] = 0.5 * (self._scorer._compute_single_tp(ground_truths, dataset_transf_pred, []) + self._scorer._compute_single_tn(ground_truths, dataset_transf_pred, []))
                fair_metric_arr[cnt] = self._scorer._equalised_odds(ground_truths, dataset_transf_pred, probabilities, demographic_attributes)['max_diff']['tp']

                cnt += 1

        rel_inds = np.logical_and(fair_metric_arr >= self._postprocessor_settings['metric_lb'],
                                  fair_metric_arr <= self._postprocessor_settings['metric_ub'])
        if any(rel_inds):
            best_ind = np.where(balanced_acc_arr[rel_inds]
                                == np.max(balanced_acc_arr[rel_inds]))[0][0]
        else:
            print("Unable to satisy fairness constraints")
            rel_inds = np.ones(len(fair_metric_arr), dtype=bool)
            best_ind = np.where(fair_metric_arr[rel_inds]
                                == np.min(fair_metric_arr[rel_inds]))[0][0]

        self.best_ROC_margin = ROC_margin_arr[rel_inds][best_ind]
        self.best_classification_threshold = class_thresh_arr[rel_inds][best_ind]

        best_predictions = self.predict(probabilities, demographic_attributes, self.best_classification_threshold, self.best_ROC_margin)
        return best_predictions, probabilities

    def predict(self, probabilities, demographic_attributes, classification_threshold, roc_margin):
        """Obtain fair predictions using the ROC method.

        Args:
            dataset (BinaryLabelDataset): Dataset containing scores that will
                be used to compute predicted labels.

        Returns:
            dataset_pred (BinaryLabelDataset): Output dataset with potentially
            fair predictions obtain using the ROC method.
        """
        fav_pred_inds = (np.array(probabilities) > classification_threshold)
        unfav_pred_inds = ~fav_pred_inds

        y_pred = np.zeros(np.array(probabilities).shape)
        y_pred[fav_pred_inds] = 1
        y_pred[unfav_pred_inds] = 0

        # Indices of critical region around the classification boundary
        crit_region_inds = np.logical_and(
                np.array(probabilities) <= classification_threshold+roc_margin,
                np.array(probabilities) > classification_threshold-roc_margin)

        # Indices of privileged and unprivileged groups
        cond_priv = np.array(self.get_binary_privileged(demographic_attributes))
        cond_unpriv = np.array(self.get_binary_protected_privileged(demographic_attributes))

        # New, fairer labels
        new_labels = y_pred
        new_labels[np.logical_and(crit_region_inds, cond_priv.reshape(-1,1))] = 0
        new_labels[np.logical_and(crit_region_inds, cond_unpriv.reshape(-1,1))] = 1
        # print(np.sum(new_labels))
        return new_labels[:, 1]

        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
