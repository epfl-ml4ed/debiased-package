import math
import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB as nb
from mitigation.preprocessing.preprocessor import PreProcessor

class ZelayaPSPPreProcessor(PreProcessor):
    """Resampling pre-processing
    fair correct psp

    References:
        https://github.com/vladoxNCL/fairCorrect/blob/master/fairCorrect.ipynb
        Zelaya, V., Missier, P., & Prangle, D. (2019). Parametrised data sampling for fairness optimisation. KDD XAI.
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'zelaya psp et al.'
        self._notation = 'zelayapsp'
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

    def fairCorrectPSP(self, df, pa, fav, d=1):
        df2 = df.copy()
        
        # Train NB model and get prediction confidence
        Xnb = df.drop(['label'], axis=1).values
        ynb = df['label'].values
        
        model = nb()
        model.fit(Xnb, ynb)
        logProbs = model.predict_log_proba(Xnb)
        df2['pPos'] = [logProbs[i][1] for i in range(len(logProbs))]
        
        # subset favoured positive, favoured negative, unfavoured positive, unfavoured negative
        fav_pos = (df2[(df2[pa] == fav) & (df2['label'] == 1)]
                .sort_values('pPos', ascending=True))
        fav_neg = (df2[(df2[pa] == fav) & (df2['label'] == 0)]
                .sort_values('pPos', ascending=False))
        unfav_pos = (df2[(df2[pa] != fav) & (df2['label'] == 1)]
                    .sort_values('pPos', ascending=True))
        unfav_neg = (df2[(df2[pa] != fav) & (df2['label'] == 0)]
                    .sort_values('pPos', ascending=False))
        
        # drop aux columns
        fav_pos = fav_pos.drop(['pPos'], axis=1)
        fav_neg = fav_neg.drop(['pPos'], axis=1)
        unfav_pos = unfav_pos.drop(['pPos'], axis=1)
        unfav_neg = unfav_neg.drop(['pPos'], axis=1)

        print('DDDDDD')
        print(fav_pos)
        print('FAV NEG', fav_neg)
        print(unfav_pos)
        print(unfav_neg)
        
        if d < 1:
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

            # number of elements to remove or add
            fav_move = math.floor((fav_pr - corr_fpr) * fav_size)
            unfav_move = math.floor((corr_upr - unfav_pr) * unfav_size)

            # elements to add
            print('DEBUG')
            print(fav_neg)
            print([fav_neg] * math.ceil(fav_move / fav_neg.shape[0]))
            print([unfav_pos] * math.ceil(unfav_move / unfav_pos.shape[0]))
            ext_fn = pd.concat([fav_neg] * math.ceil(fav_move / fav_neg.shape[0]), ignore_index=True)
            ext_up = pd.concat([unfav_pos] * math.ceil(unfav_move / unfav_pos.shape[0]), ignore_index=True)
        
            # remove from fp and un
            fav_pos = fav_pos.tail(fav_pos.shape[0] - fav_move)
            unfav_neg = unfav_neg.tail(unfav_neg.shape[0] - unfav_move)
        
            # add to fn and up
            fav_neg = pd.concat([fav_neg, ext_fn.head(fav_move)], ignore_index=True) 
            unfav_pos = pd.concat([unfav_pos, ext_up.head(unfav_move)], ignore_index=True)
        
        # concatenate df's
        corr_dfs = [fav_pos, fav_neg, unfav_pos, unfav_neg]
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
        corrected_df = self.fairCorrectPSP(full_data, 'demographic', 1, d=self._preprocessor_settings['d'])
        
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
    
        
