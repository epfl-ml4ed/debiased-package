import math
import numpy as np

from mitigation.preprocessing.preprocessor import PreProcessor
from pipelines.crossvalidation_pipeline import CrossValMaker

class KamiranPreProcessor(PreProcessor):
    """Massage the data to take out disparate impact
    Preprocessing: 
        Create a list of promotion candidates (historically discriminated and assigned negative outcome),
        and demotion candidates (historically privileged and assigned positive outcome) based on biased ranker
        (classifier)

    Optimising:
        Demographic Parity
    
    References:
        Kamiran, F., & Calders, T. (2009, February). Classifying without discriminating. In 2009 2nd international conference on computer, control and communication (pp. 1-6). IEEE.
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'kamiran et al.'
        self._notation = 'kami'
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

        # Create Promotion and Demotion List
        promotion_list = [
            prot_i for prot_i in protected_indices if y_train[prot_i] == 0
        ]
        promotion_ranks = [ranks[pl] for pl in promotion_list]
        promotion_sorted = [ps for ps in np.flip(np.argsort(promotion_ranks))]
        promotion_priorities = [promotion_list[promotion_sorted[i]] for i in range(len(promotion_sorted))]

        demotion_list = [
            demot_i for demot_i in privileged_indices if y_train[demot_i] == 1
        ]
        demotion_ranks = [ranks[dl] for dl in demotion_list]
        demotion_sorted = [ds for ds in np.argsort(demotion_ranks)]
        demotion_priorities = [demotion_list[demotion_sorted[i]] for i in range(len(demotion_sorted))]


        # Computer number of demotions and promotions
        n_protected = len(protected_indices)
        n_privileged = len(privileged_indices)
        n_protected_positive = len([pit for pit in protected_indices if y_train[pit] == 1])
        n_privileged_positive = len(demotion_list)

        m_num = (n_protected * n_privileged_positive) - (n_privileged * n_protected_positive)
        m_den = n_protected + n_privileged
        m = int(math.ceil(m_num / m_den))

        # Promote and Demote
        massaged_ys = [y for y in y_train]
        for i_m in range(m):
            assert massaged_ys[promotion_priorities[i_m]] == 0 and massaged_ys[demotion_priorities[i_m]] == 1
            massaged_ys[promotion_priorities[i_m]] = 1
            massaged_ys[demotion_priorities[i_m]] = 0

        self._information['m'] = m
        print('De- and Pro-moted {} instances!'.format(m))

        return x_train, massaged_ys, demo_train
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
