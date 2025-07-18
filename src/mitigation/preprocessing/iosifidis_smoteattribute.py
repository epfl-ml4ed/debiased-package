from collections import Counter

from imblearn.over_sampling import SMOTE
from mitigation.preprocessing.preprocessor import PreProcessor

class IosifidisSmoteAttributePreProcessor(PreProcessor):
    """Resampling pre-processing
    smote to balance attribute

    References:
        Iosifidis, V., & Ntoutsi, E. (2018). Dealing with bias via data augmentation in supervised learning scenarios. Jo Bates Paul D. Clough Robert Jäschke, 24(11).
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'iosifidis et al. - smoteatt'
        self._notation = 'iosifidissmoteatt'
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
        demographic_attributes = self.extract_demographics(demo_train)
        sensitive = self.get_binary_protected_privileged(demographic_attributes)

        samples = [
            [*x_train[student], y_train[student]] for student in range(len(x_train))
        ]
        smote = SMOTE(random_state=self._settings['seeds']['preprocessor'], sampling_strategy='minority')
        samples_sampled, sensitive_sampled = smote.fit_resample(samples, sensitive)
        x_sampled = [samples_sampled[student][:-1] for student in range(len(samples_sampled))]
        y_sampled = [samples_sampled[student][-1] for student in range(len(samples_sampled))]
        y_sampled = [int(ys >= 0.5) for ys in y_sampled]
        return x_sampled, y_sampled, sensitive_sampled
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
