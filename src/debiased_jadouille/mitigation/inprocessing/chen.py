from shutil import copytree

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from aif360.datasets import BinaryLabelDataset
from debiased_jadouille.mitigation.inprocessing.chen_repo.WAE import data_dis
import os
import logging
import pickle
import numpy as np
import pandas as pd

# import torch.nn.functional as F
# import tensorflow as tf
# import torch
from shutil import copytree, rmtree
from copy import deepcopy
from typing import Tuple

from debiased_jadouille.mitigation.inprocessing.inprocessor import InProcessor

class ChenInProcessor(InProcessor):
    """inprocessing

        References:
            Chen, Z., Zhang, J. M., Sarro, F., & Harman, M. (2022, November). MAAT: a novel ensemble approach to addressing fairness and performance bugs for machine learning software. In Proceedings of the 30th ACM joint european software engineering conference and symposium on the foundations of software engineering (pp. 1122-1134).
            https://github.com/chenzhenpeng18/FSE22-MAAT/tree/main/MAAT

    """
    
    def __init__(self, mitigating, discriminated, model_name='lr'):
        super().__init__({'mitigating': mitigating, 'discriminated': discriminated, 'model_name': model_name})
        self._model_name = model_name
        self._information = {}

    def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
        self.scaler = MinMaxScaler()
        data = pd.DataFrame(x)
        self.scaler.fit(data)
        data = pd.DataFrame(self.scaler.transform(data), columns=data.columns)

        demographic_attributes = self.extract_demographics(demographics)
        demos = self.get_binary_privileged(demographic_attributes)
        data['demographics'] = demos
        privileged_groups = [{'demographics':1}]
        protected_groups = [{'demographics':0}]

        data['Probability'] = y
        return data, privileged_groups, protected_groups
    
    def _format_features(self, x:list, demographics:list) -> list:
        data = pd.DataFrame(x)
        data = pd.DataFrame(self.scaler.transform(data), columns=data.columns)

        demographic_attributes = self.extract_demographics(demographics)
        demos = self.get_binary_privileged(demographic_attributes)
        data['demographics'] = demos
        privileged_groups = [{'demographics':1}]
        protected_groups = [{'demographics':0}]

        data['Probability'] = 0
        return data, privileged_groups, protected_groups

    def _init_model(self):
        """Initiates a model with self._model
        """
        if self._model_name == "lr":
            self.model1 = LogisticRegression()
            self.model2 = LogisticRegression()
        elif self._model_name == "svm":
            self.model1 = CalibratedClassifierCV(base_estimator=LinearSVC())
            self.model2 = CalibratedClassifierCV(base_estimator=LinearSVC())
        elif self._model_name == "rf":
            self.model1 = RandomForestClassifier()
            self.model2 = RandomForestClassifier()

    def init_model(self):
        self._init_model()

    def fit(self, 
        x_train: list, y_train: list, demographics_train: list,
        x_val=[], y_val=[], demographics_val=[]
    ):
        """fits the model with the training data x, and labels y. 
        Warning: Init the model every time this function is called

        Args:
            x_train (list): training feature data 
            y_train (list): training label data
            x_val (list): validation feature data
            y_val (list): validation label data
        """
        self._init_model()

        data, priv_groups, prot_groups = self._format_final(x_train, y_train, demographics_train)
        dataset = BinaryLabelDataset(
            favorable_label=1, unfavorable_label=0, df=data, 
            label_names=['Probability'], protected_attribute_names=['demographics']
        )

        data_bis = data_dis(data, 'demographics', 'education')
        dataset_bis = BinaryLabelDataset(
            favorable_label=1, unfavorable_label=0, df=data_bis, 
            label_names=['Probability'], protected_attribute_names=['demographics']
        )

        self.model1.fit(dataset.features, dataset.labels)
        self.model2.fit(dataset_bis.features, dataset_bis.labels)
    
    def predict(self, x: list, y, demographics: list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
            return x and y
        """
        data, priv_groups, prot_groups = self._format_features(x, demographics)
        dataset = BinaryLabelDataset(
            favorable_label=1, unfavorable_label=0, df=data, 
            label_names=['Probability'], protected_attribute_names=['demographics']
        )

        data_bis = data_dis(data, 'demographics', 'education')
        dataset_bis = BinaryLabelDataset(
            favorable_label=1, unfavorable_label=0, df=data_bis, 
            label_names=['Probability'], protected_attribute_names=['demographics']
        )

        pred_de1 = self.model1.predict_proba(dataset.features)
        pred_de2 = self.model2.predict_proba(dataset_bis.features)

        res = []
        for i in range(len(pred_de1)):
            prob_t = (pred_de1[i][1]+pred_de2[i][1])/2
            if prob_t >= 0.5:
                res.append(1)
            else:
                res.append(0)
        return res, y


    def predict_proba(self, x: list, demographics:list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
        """
        data, priv_groups, prot_groups = self._format_features(x, demographics)
        dataset = BinaryLabelDataset(
            favorable_label=1, unfavorable_label=0, df=data, 
            label_names=['Probability'], protected_attribute_names=['demographics']
        )

        data_bis = data_dis(data, 'demographics', 'education')
        dataset_bis = BinaryLabelDataset(
            favorable_label=1, unfavorable_label=0, df=data_bis, 
            label_names=['Probability'], protected_attribute_names=['demographics']
        )

        pred_de1 = self.model1.predict_proba(dataset.features)
        pred_de2 = self.model2.predict_proba(dataset_bis.features)


        res = []
        for i in range(len(pred_de1)):
            proba = pred_de1[i][1]+pred_de2[i][1]/2
            res.append([1-proba, proba])
        return res

