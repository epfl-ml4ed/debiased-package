import numpy as np
import pandas as pd

from mitigation.preprocessing.FAWOS.TrainingSet import Training
from mitigation.preprocessing.preprocessor import PreProcessor


class SalazarPreProcessor(PreProcessor):
    """Resampling pre-processing
    FAWOS

    References:
        Salazar, T., Santos, M. S., Araújo, H., & Abreu, P. H. (2021). Fawos: Fairness-aware oversampling algorithm based on distributions of sensitive attributes. IEEE Access, 9, 81370-81379
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'salazar et al.'
        self._notation = 'salazar'
        self._preprocessor_settings = self._settings['preprocessors']['salazar']
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

        train_set = Training(self._settings, self._preprocessor_settings, x_train, y_train, demographic_attributes)
        train_dataset = pd.DataFrame(x_train)
        feature_columns = ['f{}'.format(f) for f in train_dataset.columns]
        train_dataset.columns = feature_columns
        train_dataset['demographic'] = sensitive
        train_dataset['target'] = y_train
        train_set.reset_encoding_mapping()

        for feature in train_set.features:
            feature_name = feature.name
            feature_values_raw = train_dataset[feature_name]
            feature_values = feature.feature_type.encode(train_set, feature_name, feature_values_raw)
            train_dataset[feature_name] = feature_values

        x_sampled = np.array(train_dataset[feature_columns])
        y_sampled = np.array(train_dataset['target'])
        demo_sampled = np.array(train_dataset['demographic'])
        return x_sampled, y_sampled, demo_sampled


    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
