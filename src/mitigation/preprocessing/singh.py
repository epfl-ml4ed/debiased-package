import numpy as np
import pandas as pd
from collections import Counter

from mitigation.preprocessing.preprocessor import PreProcessor

class SinghSamplePreProcessor(PreProcessor):
    """Resampling pre-processing

    References:
        Singh, A., Singh, J., Khan, A., & Gupta, A. (2022). Developing a novel fair-loan classifier through a multi-sensitive debiasing pipeline: Dualfair. Machine Learning and Knowledge Extraction, 4(1), 240-253.
        https://github.com/ariba-k/fair-loan-predictor
        note: self-implementation though the repository exists
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'singsample et al.'
        self._notation = 'sisa'
        self._preprocessor_settings = {} #self._settings['preprocessors']['']
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

    def _get_medians(self, y_train, demographic_attributes):
        counts = {}
        for da in np.unique(demographic_attributes):
            da_indices = [d for d in range(len(demographic_attributes)) if demographic_attributes[d]==da]
            da_ys = [y_train[daidx] for daidx in da_indices]
            da_counter = Counter(da_ys)
            counts[da] = {
                'positive': da_counter[1],
                'negative': da_counter[0]
            }
        positive_medians = np.median([counts[cd]['positive'] for cd in counts])
        negative_medians = np.median([counts[cd]['negative'] for cd in counts])
        return positive_medians, negative_medians

    def _resample(self, group, median):
        if len(group) >= median:
            replace = False
        else:
            replace = True
        
        return np.random.choice(group, size=int(median), replace=replace)

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
        demographic_attributes = self.extract_demographics(demo_train)
        positive_medians, negative_medians = self._get_medians(y_train, demographic_attributes)
        medians = {0: negative_medians, 1: positive_medians}

        x_sampled, y_sampled, demo_sampled = [], [], []
        for da in np.unique(demographic_attributes):
            # demographic subgroup
            da_indices = [d for d in range(len(demographic_attributes)) if demographic_attributes[d]==da]
            da_xs = [x_train[daidx] for daidx in da_indices]
            da_ys = [y_train[daidx] for daidx in da_indices]
            da_demos = [demographic_attributes[daidx] for daidx in da_indices]

            # sampling per label
            for label in [0, 1]:
                da_label_indices = [i for i in range(len(da_ys)) if da_ys[i] == label]
                da_label_xs = [da_xs[dlidx] for dlidx in da_label_indices]
                da_label_demos = [da_demos[dlidx] for dlidx in da_label_indices]
                new_indices = self._resample([i for i in range(len(da_label_indices))], medians[label])

                x_sampled = x_sampled + [da_label_xs[ni] for ni in new_indices]
                y_sampled = y_sampled + [label for _ in range(len(new_indices))]
                demo_sampled = demo_sampled + [da_label_demos[ni] for ni in new_indices]

        # checkup_pre = ['{}_{}'.format(y_train[student], (demographic_attributes[student])) for student in range(len(x_train))]
        # checkup_post = ['{}_{}'.format(y_sampled[student], (demo_sampled[student])) for student in range(len(x_sampled))]
        # print('pre')
        # print(Counter(checkup_pre))
        # print('post')
        # print(Counter(checkup_post))
        return x_sampled, y_sampled, demo_sampled
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
