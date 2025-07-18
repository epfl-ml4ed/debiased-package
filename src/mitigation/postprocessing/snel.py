import numpy as np
import pandas as pd

from predictors.predictor import Predictor
from mitigation.postprocessing.postprocessor import PostProcessor

class SnelPostProcessor(PostProcessor):
    """post-processing

    References:
        Snel, P., & van Otterloo, S. (2022). Practical bias correction in neural networks: a credit default prediction case study. Computers and Society Research Journal, 3.
        https://github.com/pietsnel/Practical-bias-correction-in-neural-networks/blob/main/notebook.ipynb
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'Snel et al.'
        self._notation = 'snel'
        self._postprocessor_settings = self._settings['postprocessors']['snel']
        self._information = {}

    def transform(
            self, model: Predictor, features:list, ground_truths: list, predictions: list,
            probabilities: list, demographics: list
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
        demographic_attributes = self.extract_demographics(demographics)
        data = pd.DataFrame(features)
        data['probabilities'] = [pb for pb in probabilities]
        data['demographics'] = demographic_attributes
        data['sorting'] = [i for i in range(len(data))]

        all_groups = pd.DataFrame()
        for group, group_df in data.groupby('demographics'):
            if group in self.thresholds:
                group_df['new_probabilities'] = self.classify_to_corrected(group_df, self.thresholds[group])
                all_groups = pd.concat([all_groups, group_df])
            else:
                group_df['new_probabilities'] = group_df['probabilities']
        all_groups = all_groups.sort_values('sorting')
        return np.array(all_groups['new_probabilities']), np.array(probabilities)

    def classify_to_corrected(self, df, treshold):
        return df['probabilities'].apply(lambda x: 1 if x[1] >= treshold else 0 )

    def calculate_group_treshold(self, group_df):
        #calculate overall dataset probabilities of default
        actual_probability = group_df['label'].mean()
        predicted_probability = group_df['predictions'].mean()

        #calculated the relative increase and apply to adversely impacted group
        relative_delta = predicted_probability / actual_probability
        target_probability = group_df['label'].mean() * relative_delta

        
        group_classifications = {}
        
        #loop through tresholds and find optimum to meet target treshold
        for i in np.linspace(0,1,1001):
            loop_df = self.classify_to_corrected(group_df, i)
            target_per = loop_df.value_counts(1)
            try: 
                group_classifications[i] = abs(target_per[1] - target_probability)
            except KeyError:
                break    

        optimum = min(group_classifications.values())
        return [key for key in group_classifications if group_classifications[key] == optimum][0]

    def fit_transform( 
            self, model: Predictor, features:list, ground_truths: list, predictions: list,
            probabilities: list, demographics: list
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
        demographic_attributes = self.extract_demographics(demographics)
        data = pd.DataFrame(features)
        data['demographics'] = demographic_attributes
        data['label'] = ground_truths
        data['predictions'] = predictions
        data['probabilities'] = probabilities
        data['sorting'] = [i for i in range(len(data))]

        all_groups = pd.DataFrame()
        self.thresholds = {}
        for group, group_df in data.groupby('demographics'):
            adjusted_threshold = self.calculate_group_treshold(group_df)
            self.thresholds[group] = adjusted_threshold
            group_df['new_probabilities'] = self.classify_to_corrected(group_df, adjusted_threshold)
            all_groups = pd.concat([all_groups, group_df])
        all_groups = all_groups.sort_values('sorting')
        return np.array(all_groups['new_probabilities']), np.array(probabilities)
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
