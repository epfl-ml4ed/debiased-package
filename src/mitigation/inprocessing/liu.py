from shutil import copytree

import os
import logging
import pickle
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from shutil import copytree, rmtree
from copy import deepcopy
from typing import Tuple

from mitigation.inprocessing.inprocessor import InProcessor

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        # Define a linear layer with input dimension to 1 output (binary classification)
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # Apply the sigmoid activation function to the linear layer's output
        outputs = torch.sigmoid(self.linear(x))
        return outputs

class LiuInProcessor(InProcessor):
    """inprocessing

        References:
            Liu, Z., Jiao, X., Li, C., & Xing, W. (2024). Fair Prediction of Students' Summative Performance Changes Using Online Learning Behavior Data. In Proceedings of the 17th International Conference on Educational Data Mining (pp. 686-691).
            https://github.com/ZifengLiu98/logistic-regression-with-treatment-equality-constrain/
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'liu et al.'
        self._notation = 'liu'
        self._inprocessor_settings = self._settings['inprocessors']['liu']
        self._information = {}
        self._fold = -1

    def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
        data = pd.DataFrame(x)

        demographic_attributes = self.extract_demographics(demographics)
        demos = self.get_binary_protected_privileged(demographic_attributes)
        data['demographics'] = demos
        return torch.Tensor(np.array(data)), torch.Tensor(y)
    
    def _format_features(self, x:list, demographics:list) -> list:
        data = pd.DataFrame(x)

        demographic_attributes = self.extract_demographics(demographics)
        demos = self.get_binary_protected_privileged(demographic_attributes)
        data['demographics'] = demos
        return  torch.Tensor(np.array(data))

    def _init_model(self):
        """Initiates a model with self._model
        """
        self.model = LogisticRegressionModel(self.input_size+1)
        self.optimiser = optim.Adam(self.model.parameters(), lr=0.001)

    def init_model(self):
        self._init_model()

    def calculate_group_fairness_metric(self, outputs, labels, sensitive_feature):
        epsilon = 1e-6  # A small constant to prevent division by zero

        # Split data into two groups based on sensitive feature
        group_0 = (sensitive_feature == 0)
        group_1 = (sensitive_feature == 1)

        # Compute False Positive Rate (FPR) and False Negative Rate (FNR) for each group
        fpr_0 = torch.mean(outputs[group_0 & (labels == 0)]) if group_0.any() and (labels[group_0] == 0).any() else torch.tensor(0.0)
        fnr_0 = torch.mean(1 - outputs[group_0 & (labels == 1)]) if group_0.any() and (labels[group_0] == 1).any() else torch.tensor(0.0)
        fpr_1 = torch.mean(outputs[group_1 & (labels == 0)]) if group_1.any() and (labels[group_1] == 0).any() else torch.tensor(0.0)
        fnr_1 = torch.mean(1 - outputs[group_1 & (labels == 1)]) if group_1.any() and (labels[group_1] == 1).any() else torch.tensor(0.0)

        # Calculate a fairness metric based on FPR and FNR
        return torch.abs(fnr_0 * fpr_1/(fpr_0 * fnr_1 + epsilon)-1)

    def custom_loss(self, outputs, labels, inputs, lambda_fairness=0.01):
        # Basic loss: binary cross-entropy
        criterion = nn.BCELoss()
        loss = criterion(outputs, labels.view(-1, 1))

        # Add fairness loss
        fairness_loss = 0
        # Calculate fairness metric for each sensitive feature index
        fairness_loss += self.calculate_group_fairness_metric(outputs, labels, inputs[:, -1])
        # Total loss includes both basic and fairness-induced losses
        total_loss = loss + lambda_fairness * fairness_loss

        return total_loss

    def fit(self, 
        x_train: list, y_train: list, demographics_train: list,
        x_val:list, y_val:list, demographics_val: list
    ):
        """fits the model with the training data x, and labels y. 
        Warning: Init the model every time this function is called

        Args:
            x_train (list): training feature data 
            y_train (list): training label data
            x_val (list): validation feature data
            y_val (list): validation label data
        """
        self.input_size = len(x_train[0])
        self._init_model()

        data_tensor, label_tensor = self._format_final(x_train, y_train, demographics_train)
        train_loader = DataLoader(
            TensorDataset(data_tensor, label_tensor), shuffle=True,
            batch_size=self._inprocessor_settings['batch_size']
        )

        for epoch in range(self._inprocessor_settings['epochs']):
            print('epoch: {}'.format(epoch))
            self.model.train()
            for x_batch, y_batch in train_loader:
                # zero the parameter gradients
                self.optimiser.zero_grad()

                outputs = self.model(x_batch)
                loss = self.custom_loss(outputs, y_batch, x_batch)
                loss.backward()
                self.optimiser.step()

    
    def predict(self, x: list, y, demographics: list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
            return x and y
        """
        data = self._format_features(x, demographics)
        test_loader = DataLoader(
            data, shuffle=True,
            batch_size=self._inprocessor_settings['batch_size']
        )
        
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_x in test_loader:
                outputs = self.model(batch_x)
                predicted = (outputs.data > 0.5).float()
                predictions += predicted
        return predictions, y

    def predict_proba(self, x: list, demographics:list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
        """
        data = self._format_features(x, demographics)
        test_loader = DataLoader(
            data, shuffle=True,
            batch_size=self._inprocessor_settings['batch_size']
        )
        
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_x in test_loader:
                outputs = self.model(batch_x)
                predicted = (outputs.data).float()
                predictions += predicted
        pred0 = 1 - np.array(predictions)
        probabilities = np.array([predictions, pred0]).reshape(2, len(predictions)).transpose()
        return probabilities

    def save(self, extension='') -> str:
        """Saving the model in the following path:
        '../experiments/run_year_month_day/models/model_name_fx.pkl

        Returns:
            String: Path
        """
        path = '{}/models/'.format(self._settings['experiment']['name'])
        os.makedirs(path, exist_ok=True)
        with open('{}{}_{}.pkl'.format(path, self._notation, extension), 'wb') as fp:
            pickle.dump(self._information, fp)
        return '{}{}_{}'.format(path, self._notation, extension)

    def save_fold(self, fold: int) -> str:
        return self.save(extension='fold_{}'.format(fold))

    def save_fold_early(self, fold: int) -> str:
        return self.save(extension='fold_{}_len{}'.format(
            fold, self._maxlen
        ))
    

        
