import numpy as np
from collections import Counter

from debiased_jadouille.mitigation.preprocessing.preprocessor import PreProcessor

class CaldersPreProcessor(PreProcessor):
    """Resampling pre-processing
    - Reweighting the classes

    References:
        Calders, T., Kamiran, F., & Pechenizkiy, M. (2009, December). Building classifiers with independency constraints. In 2009 IEEE international conference on data mining workshops (pp. 13-18). IEEE
    """
    
    def __init__(self, mitigating, discriminated, sampling_proportions= 1):
        super().__init__({'mitigating': mitigating, 'discriminated': discriminated, 'sampling_proportions':sampling_proportions})
        self._sampling_proportions = sampling_proportions
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
        demographic_attributes = self.extract_demographics(demo_train)

        proportions = {'demographics': {}, 'labels': {}}
        for demo in np.unique(demographic_attributes):
            proportions['demographics'][demo] = len([i for i in range(len(demographic_attributes)) if demographic_attributes[i] == demo]) / len(demographic_attributes)
        for label in np.unique(y_train):
            proportions['labels'][label] = len([i for i in range(len(y_train)) if y_train[i] == label]) / len(y_train)

        weights = {}
        for demo in np.unique(demographic_attributes):
            weights[demo] = {}
            for label in np.unique(y_train):
                try:
                    conjunction = len(
                        [i for i in range(len(y_train)) if y_train[i] == label and demographic_attributes[i] == demo]
                    ) / len(y_train)
                    weights[demo][label] = (proportions['demographics'][demo] * proportions['labels'][label]) / conjunction
                except ZeroDivisionError:
                    weights[demo][label] = 1
        weights_train = [weights[demographic_attributes[student]][y_train[student]] for student in range(len(x_train))]
        weights_train = np.array(weights_train) / np.sum(weights_train)

        sampled_students = np.random.choice(
            [i for i in range(len(x_train))],
            size=int(len(x_train)*self._sampling_proportions),
            replace=True,
            p=weights_train
        )
        x_sampled = [x_train[sast] for sast in sampled_students]
        y_sampled = [y_train[sast] for sast in sampled_students]
        demo_sampled = [demo_train[sast] for sast in sampled_students]

        self._information['weights'] = demo_sampled
        return x_sampled, y_sampled, demo_sampled
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
