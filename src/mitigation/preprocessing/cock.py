from multiprocessing.sharedctypes import Value
from random import random
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, SpectralClustering, HDBSCAN
from mitigation.preprocessing.preprocessor import PreProcessor
from imblearn.over_sampling import RandomOverSampler

class CockPreProcessor(PreProcessor):
    """Resampling pre-processing
    Blind Resampling

    References:
        
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'Cock et al.'
        self._notation = 'cock'
        self._preprocessor_settings = self._settings['preprocessors']['cock']
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

    def _init_clustering(self, x_train, parameter):
        if self._preprocessor_settings['clustering'] == 'spectral':
            cluster_algo = SpectralClustering(
                n_clusters=parameter
            )

        elif self._preprocessor_settings['clustering'] == 'kmeans':
            cluster_algo = KMeans(n_clusters=parameter)

        elif self._preprocessor_settings['clustering'] == 'hdbscan':
            cluster_algo = HDBSCAN(min_cluster_size=int(len(x_train)/parameter))

        return cluster_algo

    def _cluster_optimising(self, x_train):
        best_silhouette_score = -3
        best_param = 0
        for param in self._preprocessor_settings['combinations']:
            clustering = self._init_clustering(x_train, param)
            labels = clustering.fit_predict(x_train)
            try:
                silscore = silhouette_score(x_train, labels)
                if silscore > best_silhouette_score:
                    best_silhouette_score = silscore
                    best_param = param
            except ValueError:
                continue

        best_clustering = self._init_clustering(x_train, best_param)
        return best_clustering.fit_predict(x_train)

    def fit_transform(self, 
            x_train: list, y_train: list, demo_train: list,
            x_val: list, y_val: list, demo_val: list
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
        # Cluster
        resampling_attributes = self._cluster_optimising(x_train)

        # Compute resampling
        resampling_counter = Counter(resampling_attributes)
        n_samples = resampling_counter.most_common()[0][1]
        # Resampling
        resampling_strategy = {ra: n_samples for ra in np.unique(resampling_attributes)}
        ros = RandomOverSampler(random_state=self._settings['seeds']['preprocessor'], sampling_strategy=resampling_strategy)
        
        x_sampled, _ = ros.fit_resample(x_train, resampling_attributes)
        indices_sampled = ros.sample_indices_
        y_sampled = [y_train[iss] for iss in indices_sampled]
        demo_sampled = [demo_train[iss] for iss in indices_sampled]
        return x_sampled, y_sampled, demo_sampled
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
