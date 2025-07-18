import numpy as np
import pandas as pd

from mitigation.preprocessing.preprocessor import PreProcessor

class ZelayaOverPreProcessor(PreProcessor):
    """Resampling pre-processing
    fair correct over resampling

    References:
        https://github.com/vladoxNCL/fairCorrect/blob/master/fairCorrect.ipynb
        Zelaya, V., Missier, P., & Prangle, D. (2019). Parametrised data sampling for fairness optimisation. KDD XAI.
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'zelayaover et al.'
        self._notation = 'zelayaover'
        self._preprocessor_settings = self._settings['preprocessors']['zelaya']
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

    def fairCorrectOver(self, df, pa, fav, d=1):
        """Correct the proportion of positive cases for favoured and unfavoured subgroups through
        oversampling the unfavoured positive and favoured negative classes. Parameter d should be
        a number between -1 and 1 for this to work properly."""
        
        # import math for floor function
        import math
        
        # subset favoured positive, favoured negative, unfavoured positive, unfavoured negative
        fav_pos = df[(df[pa] == fav) & (df.label == 1)]
        fav_neg = df[(df[pa] == fav) & (df.label == 0)]
        unfav_pos = df[(df[pa] != fav) & (df.label == 1)]
        unfav_neg = df[(df[pa] != fav) & (df.label == 0)]
        
        # get favoured and unfavoured number of rows
        fav_size = fav_pos.shape[0] + fav_neg.shape[0]
        unfav_size = unfav_pos.shape[0] + unfav_neg.shape[0]

        # get positive ratios for favoured and unfavoured
        fav_pr = fav_pos.shape[0] / fav_size
        unfav_pr = unfav_pos.shape[0] / unfav_size
        pr = df[df['label'] == 1].shape[0] / df.shape[0]

        # coefficients for fitting quad function
        a = ((fav_pr + unfav_pr) / 2) - pr
        b = (fav_pr - unfav_pr) / 2
        c = pr

        # corrected ratios
        corr_fpr = (a * (d ** 2)) + (b * d) + c
        corr_upr = (a * (d ** 2)) - (b * d) + c
        
        # correcting constants
        fav_k = (1 - corr_fpr) / corr_fpr
        unfav_k = corr_upr / (1 - corr_upr)
        
        # sample sizes for unfav_pos and fav_neg
        unfav_pos_size = math.floor(unfav_neg.shape[0] * unfav_k)
        fav_neg_size = math.floor(fav_pos.shape[0] * fav_k)
        
        # samples from fav_pos and unfav_neg to correct proportions
        corr_unfav_pos = unfav_pos.sample(unfav_pos_size, replace=True)
        corr_fav_neg = fav_neg.sample(fav_neg_size, replace=True)
        
        # concatenate df's
        corr_dfs = [corr_unfav_pos, fav_pos, unfav_neg, corr_fav_neg]
        corr_df = pd.concat(corr_dfs)
        
        return corr_df

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
        sensitive_attribute = self.get_binary_protected_privileged(demographic_attributes)
        full_data = pd.DataFrame(x_train)
        full_data['demographic'] = sensitive_attribute
        full_data['label'] = y_train
        corrected_df = self.fairCorrectOver(full_data, 'demographic', 1, d=self._preprocessor_settings['d'])
        
        y_sampled = corrected_df['label']
        demo_sampled = corrected_df['demographic']
        x_sampled = corrected_df.drop('label', axis=1)
        x_sampled = x_sampled.drop('demographic', axis=1)
        x_sampled = np.array(x_sampled)
        return x_sampled, y_sampled, demo_sampled
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
