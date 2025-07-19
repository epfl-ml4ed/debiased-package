import math
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from debiased_jadouille.mitigation.preprocessing.preprocessor import PreProcessor

class ZelayaSMOTEPreProcessor(PreProcessor):
    """Resampling pre-processing
    fair correct under resampling

    References:
        https://github.com/vladoxNCL/fairCorrect/blob/master/fairCorrect.ipynb
        Zelaya, V., Missier, P., & Prangle, D. (2019). Parametrised data sampling for fairness optimisation. KDD XAI.
    """
    
    def __init__(self, mitigating, discriminated, d=-0.3):
        super().__init__({'mitigating': mitigating, 'discriminated': discriminated, 'd':d})
        self._d = d
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

    def fairCorrectSMOTE(self, df, pa, fav, d=1):
        """Correct the proportion of positive cases for favoured and unfavoured subgroups through 
        oversampling the unfavoured positive and favoured negative classes. Parameter d should be
        a number between -1 and 1 for this to work properly."""
        
        # Put label last for easier manipulation
        df = df[[c for c in df if c not in ['label']] + ['label']]
        
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
        
        # SMOTE oversample UP & FN to correct constant
        
        sm = SMOTE()
        auxfav = fav_pos.sample(math.floor(fav_pos.shape[0] * fav_k),
                                replace=True)
        auxunfav = unfav_neg.sample(math.floor(unfav_neg.shape[0] * unfav_k),
                                    replace=True)
        
        dff_pre = pd.concat([auxfav, fav_neg])
        dfu_pre = pd.concat([unfav_pos, auxunfav])
        
        # Split in X & y for SMOTE to work
        Xf_pre = dff_pre.drop(['label'], axis=1).values
        Xu_pre = dfu_pre.drop(['label'], axis=1).values
        yf_pre = dff_pre.label.values
        yu_pre = dfu_pre.label.values

        # Apply SMOTE
        X_smf, y_smf = sm.fit_resample(Xf_pre, yf_pre)
        X_smu, y_smu = sm.fit_resample(Xu_pre, yu_pre)
        
        # Put back together into dataframes
        y_smf = y_smf.reshape((y_smf.shape[0], 1))
        X_smf = np.append(X_smf, y_smf, axis=1)
        y_smu = y_smu.reshape((y_smu.shape[0], 1))
        X_smu = np.append(X_smu, y_smu, axis=1)
        smfDF = pd.DataFrame(X_smf, columns=df.columns.tolist())
        smuDF = pd.DataFrame(X_smu, columns=df.columns.tolist())
        
        # concatenate df's
        corr_dfs = [smfDF[smfDF.label == 0], smuDF[smuDF.label == 1], fav_pos, unfav_neg]
        corr_df = pd.concat(corr_dfs)
        
        return corr_df

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
        sensitive_attribute = self.get_binary_protected_privileged(demographic_attributes)
        full_data = pd.DataFrame(x_train)
        full_data['demographic'] = sensitive_attribute
        full_data['label'] = y_train
        corrected_df = self.fairCorrectSMOTE(full_data, 'demographic', 1, d=self._d)
        
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
    
        
