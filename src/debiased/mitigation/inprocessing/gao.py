from shutil import copytree

import os
import logging
import pickle
import numpy as np
import pandas as pd

# import torch.nn.functional as F
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from shutil import copytree, rmtree
from sklearn import preprocessing
from copy import deepcopy
from typing import Tuple

from torch.utils.data import TensorDataset, DataLoader
from debiased.mitigation.inprocessing.gao_repo.models import Net
from debiased.mitigation.inprocessing.inprocessor import InProcessor

class GaoInProcessor(InProcessor):
    """inprocessing

        References:
            Gao, X., Zhai, J., Ma, S., Shen, C., Chen, Y., & Wang, Q. (2022, May). FairNeuron: improving deep neural network fairness with adversary games on selective neurons. In Proceedings of the 44th International Conference on Software Engineering (pp. 921-933).
            https://github.com/Antimony5292/FairNeuron/tree/main/FN
    """
    
    def __init__(self, mitigating, discriminated, batch_size=128, epochs=10, grl_lambda=50, device='cpu'):
        super().__init__({'mitigating': mitigating, 'discriminated': discriminated, 'batch_size':batch_size, 'epochs': epochs, 'grl_lambda': grl_lambda})
        self._batch_size = batch_size
        self._epochs = epochs
        self._grl_lambda = grl_lambda
        self._information = {}
        self.device = device

    def _format_final(self, x:list, y:list, demographics:list) -> Tuple[list, list]:
        x_tensor = torch.tensor(pd.DataFrame(x).to_numpy().astype(np.float32))
        y_tensor = torch.tensor(np.array(y).reshape(-1, 1).astype(np.float32))
        demos = self.extract_demographics(demographics)
        demos = self.get_binary_protected_privileged(demos)
        demographic_tensor = torch.tensor(preprocessing.OneHotEncoder().fit_transform(np.array(demos).reshape(-1, 1)).toarray())
        return x_tensor, y_tensor, demographic_tensor
    
    def _format_features(self, x:list, demographics:list) -> list:
        x_tensor = torch.tensor(pd.DataFrame(x).to_numpy().astype(np.float32))
        demos = self.extract_demographics(demographics)
        demos = self.get_binary_protected_privileged(demos)
        demographic_tensor = torch.tensor(preprocessing.OneHotEncoder().fit_transform(np.array(demos).reshape(-1, 1)).toarray())
        return x_tensor, demographic_tensor

    def _init_model(self):
        """Initiates a model with self._model
        """
        self.model = Net(
            input_shape=self.input_shape, grl_lambda=self._grl_lambda
        ).to(self.device)

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
        x_tensor, y_tensor, demographic_tensor = self._format_final(x_train, y_train, demographics_train)
        train_loader = DataLoader(
            TensorDataset(x_tensor, y_tensor, demographic_tensor), shuffle=True,
            batch_size=self._batch_size
        )
        self.input_shape = len(x_train[0])
        self._init_model()

        model = self.model.to(self.device)
        criterion = nn.MSELoss().to(self.device)
        criterion_bias = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        torch.manual_seed(193)

        for epoch in range(self._epochs):
            print('epoch: {}'.format(epoch))
            self.model.train()
            batch_losses = []
            for x_batch, y_batch, s_batch in train_loader:
                # zero the parameter gradients
                optimizer.zero_grad()
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                s_batch = s_batch.to(self.device)

                # forward + backward + optimize
                if self._grl_lambda is not None and self._grl_lambda != 0:
                    outputs, outputs_protected = model(x_batch)
                    loss = criterion(outputs, y_batch) + criterion_bias(outputs_protected, s_batch.argmax(dim=1))
                else:
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())

            training_loss = np.mean(batch_losses)

    
    def predict(self, x: list, y, demographics: list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
            return x and y
        """
        x_tensor, demographic_tensor = self._format_features(x, demographics)
        test_loader = DataLoader(
            TensorDataset(x_tensor, demographic_tensor), shuffle=True,
            batch_size=self._batch_size
        )
        x_tensor, demographic_tensor = x_tensor.to(self.device), demographic_tensor.to(self.device)
        
        self.model.eval()
        predictions = torch.Tensor().to(self.device)
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(self.device)
                scores, _ = self.model(batch_x)
                probas = scores.cpu().detach().numpy()
                probas = [np.argmax(yp) for yp in probas]
                probas = torch.Tensor(probas).to(self.device)
                predictions = torch.cat((predictions, probas))
        predictions = predictions.cpu().detach().numpy()
        return predictions, y

    def predict_proba(self, x: list, demographics:list) -> list:
        """Predict the labels of x

        Args:
            x (list): features
            
        Returns:
            list: list of raw predictions for each data point
        """
        x_tensor, demographic_tensor = self._format_features(x, demographics)
        test_loader = DataLoader(
            TensorDataset(x_tensor, demographic_tensor), shuffle=True,
            batch_size=self._batch_size
        )
        x_tensor, demographic_tensor = x_tensor.to(self.device), demographic_tensor.to(self.device)
        
        self.model.eval()
        predictions = torch.Tensor().to(self.device)
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(self.device)
                scores, _ = self.model(batch_x)
                predictions = torch.cat((predictions, scores))
        predictions = predictions.cpu().detach().numpy()
        pred0 = 1 - np.array(predictions)
        probabilities = np.array([predictions, pred0]).reshape(2, len(predictions)).transpose()
        return probabilities

        
