import math
import numpy as np
import pandas as pd

from mitigation.preprocessing.preprocessor import PreProcessor
from pipelines.crossvalidation_pipeline import CrossValMaker

class LuongPreProcessor(PreProcessor):
    """Massage the data to take out disparate impact
    Preprocessing: 
        Looks into the neighbours. Grossly said, if it appears that the privileged closest neighbours have a positive outcome compared to the
        discriminated sample, then flip the label

    Optimising:
        Demographic Parity

    References:
        Luong, B. T., Ruggieri, S., & Turini, F. (2011, August). k-NN as an implementation of situation testing for discrimination discovery and prevention. In Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 502-510).
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'luong et al.'
        self._notation = 'luong'
        self._preprocessor_settings = {}
        self._information = {'k': self._settings['preprocessors']['luong']['k']}

    def _z(self, x, m, s):
        return (x-m) / s

    def _features_indices(self, features, feature_category):
        """Returns the features that are of features category (must be continuous, discrete or categorical)
        """
        feature_indices = self._settings['preprocessors']['luong'][feature_category]
        new_features = [
            [student[fi] for fi in feature_indices]for student in features
        ]
        return new_features

    def _distance_dataframe(self, distances):
        d = pd.DataFrame(distances)
        sorted_columns = [cc for cc in d.columns]
        sorted_columns.sort()
        d = d.sort_index()
        d = d[sorted_columns]
        return d


    def continuous_distances(self, full_features):
        features = self._features_indices(full_features, 'continuous')
        means = np.mean(features, axis=0)
        stds = np.mean(features, axis=0)

        zs = np.array(features) - means
        zs = zs / stds

        distances = {i: {i: 0} for i in range(len(features))}
        for x_1 in range(len(features)):
            for x_2 in range(x_1+1, len(features)):
                d = np.sum(np.abs(zs[x_1] - zs[x_2]))
                distances[x_1][x_2] = d
                distances[x_2][x_1] = d
        return self._distance_dataframe(distances)

    def discrete_distances(self, full_features):
        features = self._features_indices(full_features, 'discrete')
        maxima = np.max(features, axis=0)
        maxima = maxima - 1
        ms = np.array(features) - 1
        ms = ms / maxima

        distances = {i: {i: 0} for i in range(len(features))}
        for x_1 in range(len(features)):
            for x_2 in range(x_1+1, len(features)):
                d = np.sum(np.abs(ms[x_1] - ms[x_2]))
                distances[x_1][x_2] = d
                distances[x_2][x_1] = d
        return self._distance_dataframe(distances)

    def categorical_distances(self, full_features):
        features = self._features_indices(full_features, 'categorical')
        distances = {i: {i: 0} for i in range(len(features))}
        for x_1 in range(len(features)):
            for x_2 in range(x_1+1, len(features)):
                d = np.sum([int(dd!=0) for dd in np.array(features[x_1]) - np.array(features[x_2])])
                distances[x_1][x_2] = d
                distances[x_2][x_1] = d
        return self._distance_dataframe(distances)


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

    def k_nearest_neighbours(self, distances, indices):
        sorted_indices = np.argsort(distances)
        neighbours =  [indices[si] for si in sorted_indices]
        return neighbours[:self._information['k']]

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
        # Compute distance matrix
        continuous_distances = self.continuous_distances(x_train)
        discrete_distances = self.discrete_distances(x_train)
        categorical_distances = self.categorical_distances(x_train)
        full_distances = continuous_distances + discrete_distances + categorical_distances
        full_distances /= len(x_train[0])

        # Data Preparation
        demographic_attributes = self.extract_demographics(demo_train)
        protected_indices = self.get_protected_indices(demographic_attributes)
        privileged_indices = self.get_privileged_indices(demographic_attributes)
        assert len(x_train) == len(y_train) and len(y_train) == len(demo_train) and len(demo_train) == len(demographic_attributes)
        protected_distances = full_distances[protected_indices]
        privileged_distances = full_distances[privileged_indices]

        massaged_ys = [y for y in y_train]
        massage_count = 0
        for protected in protected_indices:
            if y_train[protected] == 0:
                k_1 =  self.k_nearest_neighbours(protected_distances.loc[protected].values, protected_indices) #protected k nearest neighbours
                k_2 = self.k_nearest_neighbours(privileged_distances.loc[protected].values, privileged_indices) # privileged k nearest neighbours

                label_protected = y_train[protected]
                labels_k1 = [y_train[kk1] for kk1 in k_1]
                labels_k2 = [y_train[kk2] for kk2 in k_2]
                p_1 = np.sum([int(lk1==label_protected) for lk1 in labels_k1]) / self._information['k']
                p_2 = np.sum([int(lk2==label_protected) for lk2 in labels_k2]) / self._information['k']
                diff = p_1 - p_2

                if diff >= 0:
                    massaged_ys[protected] = 1
                    massage_count += 1

        print('Corrected {} instances!'.format(massage_count))
        self._information['massage_count'] = massage_count
        return x_train, massaged_ys, demo_train
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
