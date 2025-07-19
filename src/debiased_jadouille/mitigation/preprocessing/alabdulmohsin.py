import math
import numpy as np
import pandas as pd

from debiased_jadouille.mitigation.preprocessing.google_unmodified.ml_debiaser import Reduce2Binary
from debiased_jadouille.mitigation.preprocessing.preprocessor import PreProcessor

class AlabdulmohsinPreProcessor(PreProcessor):
    """Massage the data to take out disparate impact
    Preprocessing: 
        Works for any number of classes. It debiases the data without using the demographic attribute during inference time

    Optimising:
        Demographic Parity

    References:
        Alabdulmohsin, I. M., Schrouff, J., & Koyejo, S. (2022). A reduction to binary approach for debiasing multiclass datasets. Advances in Neural Information Processing Systems, 35, 2480-2493.
    """
    
    def __init__(self, mitigating, discriminated, sgd_steps=10, full_gradient_epochs=1, max_admm_iter=1, n_classes=2):
        super().__init__({'mitigating': mitigating, 'discriminated': discriminated, 'sgd_steps':sgd_steps, 'full_gradient_epochs': full_gradient_epochs, 'max_admm_iter': max_admm_iter})
        self._sgd_steps = sgd_steps
        self._full_gradient_epochs = full_gradient_epochs
        self._max_admm_iter = max_admm_iter
        self._n_classes = n_classes
        self._information = {}

    def transform(self, 
        x_train: list, y_train: list, demo_train: list,
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
        return x_train, y_train, demo_train

    def _one_hot_attributes(self, attributes):
        unique_attributes = [att for att in np.unique(attributes)]
        attribute_map = {ua: i for i, ua in enumerate(unique_attributes)}
        # n_ua = len(unique_attributes)

        onehot = [attribute_map[att] for att in attributes]

        # onehot = [[0 for _ in range(n_ua)] for _ in range(len(attributes))]
        # for instance in attributes:
        #     onehot[attribute_map[instance]] = 1
        return onehot

    def fit_transform(self, 
            x_train: list, y_train: list, demo_train: list,
            x_val=[], y_val=[], demo_val=[]
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
        # Data Prep
        demographic_attributes = self.extract_demographics(demo_train)
        demoatt = self._one_hot_attributes(demographic_attributes)
        ytrain = np.zeros((len(x_train), self._n_classes))
        ytrain[np.arange(len(x_train)), y_train] = 1.

        # Massaging
        self._r2b = Reduce2Binary(num_classes=self._n_classes)
        massaged_ys = self._r2b.fit(
            ytrain, demoatt, sgd_steps=self._sgd_steps,
            full_gradient_epochs=self._full_gradient_epochs, 
            max_admm_iter=self._max_admm_iter
        )

        # Processed ys
        massaged_ys = [np.argmax(my) for my in massaged_ys]

        n_massaged = np.sum([int(y_train[idx] == massaged_ys[idx]) for idx in range(len(y_train))])
        print('{} instances were massaged! ({}%)'.format(n_massaged, (n_massaged/len(y_train)*100)))
        return x_train, massaged_ys, demo_train
        
