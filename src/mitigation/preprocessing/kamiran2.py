import math
import numpy as np

from mitigation.preprocessing.preprocessor import PreProcessor
from pipelines.crossvalidation_pipeline import CrossValMaker

class Kamiran2PreProcessor(PreProcessor):
    """Remove/Duplicate border examples

    Optimising:
        Demographic Parity
    
    References:
        Kamiran, F., & Calders, T. (2010, May). Classification with no discrimination by preferential sampling. In Proc. 19th Machine Learning Conf. Belgium and The Netherlands (Vol. 1, No. 6). Citeseer.
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'kamiran2 et al.'
        self._notation = 'kami2'
        self._preprocessor_settings = {}
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
        # Train Ranker
        ml_pipeline = CrossValMaker(self._settings)
        ranker = ml_pipeline.get_model()
        ranker = ranker(self._settings)
        ranker.fit(x_train, y_train, x_val, y_val)
        self._ranker = ranker

        ranks = self._ranker.predict_proba(x_train)
        ranks = [r[1] for r in ranks]

        # Data Preparation
        demographic_attributes = self.extract_demographics(demo_train)
        protected_indices = self.get_protected_indices(demographic_attributes)
        privileged_indices = self.get_privileged_indices(demographic_attributes)
        assert len(x_train) == len(y_train) and len(y_train) == len(demo_train) and len(demo_train) == len(demographic_attributes)
        assert len(protected_indices) + len(privileged_indices) == len(x_train)

        # Create subgroups
        protected_positive = [
            prot_i for prot_i in protected_indices if y_train[prot_i] == 1
        ]
        propo_ranks = [ranks[propo] for propo in protected_positive]
        propo_sortedranks = [propos for propos in np.argsort(propo_ranks)]
        propo_sortedranks = [protected_positive[propo_sortedranks[i]] for i in range(len(propo_sortedranks))]
        propo_size = (len(protected_indices) * np.sum(y_train)) / len(y_train)

        protected_negative = [
            prot_i for prot_i in protected_indices if y_train[prot_i] == 0
        ]
        prone_ranks = [ranks[prone] for prone in protected_negative]
        prone_sorted_ranks = [prone for prone in np.flip(np.argsort(prone_ranks))]
        prone_sorted_ranks = [protected_negative[prone_sorted_ranks[i]] for i in range(len(prone_sorted_ranks))]
        prone_size = (len(protected_indices) * (len(y_train) - np.sum(y_train))) / len(y_train)

        privileged_positive = [
            priv_i for priv_i in privileged_indices if y_train[priv_i] == 1
        ]
        pripo_ranks = [ranks[pripo] for pripo in privileged_positive]
        pripo_sortedranks = [pripos for pripos in np.argsort(pripo_ranks)]
        pripo_sortedranks = [privileged_positive[pripo_sortedranks[i]] for i in range(len(pripo_sortedranks))]
        pripo_size = (len(privileged_indices) * np.sum(y_train)) / len(y_train)

        privileged_negative = [
            priv_i for priv_i in privileged_indices if y_train[priv_i] == 0
        ]
        prine_ranks = [ranks[prine] for prine in privileged_negative]
        prine_sortedranks = [prine for prine in np.flip(np.argsort(prine_ranks))]
        prine_sortedranks = [privileged_negative[prine_sortedranks[i]] for i in range(len(prine_sortedranks))]
        prine_size = (len(privileged_indices) * (len(y_train) - np.sum(y_train))) / len(y_train)

        groups = {
            'propo': {'sorted': propo_sortedranks, 'size': propo_size},
            'prone': {'sorted': prone_sorted_ranks, 'size': prone_size},
            'pripo': {'sorted': pripo_sortedranks, 'size': pripo_size},
            'prine': {'sorted': prine_sortedranks, 'size': prine_size}
        }

        x_sampled = []
        y_sampled = []
        demo_sampled = []
        for subgroup in groups:
            n_edits = int(groups[subgroup]['size'] - len(groups[subgroup]['sorted']))

            if n_edits < 0: # remove!
                ranks = groups[subgroup]['sorted'][n_edits:]
                x_sampled = x_sampled + [x_train[r] for r in ranks]
                y_sampled = y_sampled + [y_train[r] for r in ranks]
                demo_sampled = demo_sampled + [demo_train[r] for r in ranks]

            elif n_edits > 0:
                ranks = groups[subgroup]['sorted'] + groups[subgroup]['sorted'][:n_edits]
                x_sampled = x_sampled + [x_train[r] for r in ranks]
                y_sampled = y_sampled + [y_train[r] for r in ranks]
                demo_sampled = demo_sampled + [demo_train[r] for r in ranks]

            else:
                x_sampled = x_sampled + [x_train[gsg] for gsg in groups[subgroup]['sorted']]
                y_sampled = y_sampled + [y_train[gsg] for gsg in groups[subgroup]['sorted']]
                demo_sampled = demo_sampled + [demo_train[gsg] for gsg in groups[subgroup]['sorted']]

        return x_sampled, y_sampled, demo_sampled

        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
