import numpy as np
import pandas as pd
from collections import Counter
import debiased_jadouille.mitigation.preprocessing.fair_over_sampling.Fair_OS as fair_sampler
from debiased_jadouille.mitigation.preprocessing.preprocessor import PreProcessor

from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.datasets import StructuredDataset, BinaryLabelDataset

class DablainPreProcessor(PreProcessor):
    """Resampling pre-processing

    References:
        Dablain, D., Krawczyk, B., & Chawla, N. (2022). Towards a holistic view of bias in machine learning: Bridging algorithmic fairness and imbalanced learning. arXiv preprint arXiv:2207.06084.
        https://github.com/dd1github/Fair-Over-Sampling
    """
    
    def __init__(self, mitigating, discriminated, proportion=1, k=5):
        super().__init__({'mitigating': mitigating, 'discriminated': discriminated, 'proportion':proportion, 'k': k})
        self._proportion = proportion
        self._k = k
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

    def get_majority_class(self, y_train):
        classes = Counter(y_train)
        return classes.most_common()[0][0], classes.most_common()[0][1]

    def get_all_ns(self, dataset, reweighter):
        (priv_cond, unpriv_cond, fav_cond, unfav_cond,
        cond_p_fav, cond_p_unfav, cond_up_fav, cond_up_unfav) =\
                reweighter._obtain_conditionings(dataset)

        n = np.sum(dataset.instance_weights, dtype=np.float64)
        n_p = np.sum(dataset.instance_weights[priv_cond], dtype=np.float64)
        n_up = np.sum(dataset.instance_weights[unpriv_cond], dtype=np.float64)
        n_fav = np.sum(dataset.instance_weights[fav_cond], dtype=np.float64)
        n_unfav = np.sum(dataset.instance_weights[unfav_cond], dtype=np.float64)

        n_p_fav = np.sum(dataset.instance_weights[cond_p_fav], dtype=np.float64)
        n_p_unfav = np.sum(dataset.instance_weights[cond_p_unfav],
                           dtype=np.float64)
        n_up_fav = np.sum(dataset.instance_weights[cond_up_fav],
                          dtype=np.float64)
        n_up_unfav = np.sum(dataset.instance_weights[cond_up_unfav],
                            dtype=np.float64)

        return n_fav, n_unfav, n_p_fav, n_up_fav, n_p_unfav, n_up_unfav

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
        # Prepare data
        demographic_attributes = self.extract_demographics(demo_train)
        sensitive_attributes = self.get_binary_protected_privileged(demographic_attributes)
        protected_groups = [{'demographic': 1}]
        privileged_groups = [{'demographic': 0}]
        full_data = pd.DataFrame(x_train)
        full_data['demographic'] = self.get_binary_protected_privileged(demographic_attributes)
        full_data['label'] = y_train
        dataset = BinaryLabelDataset(
            favorable_label=1, unfavorable_label=0,
            df=full_data, label_names=['label'], protected_attribute_names=['demographic'],
            unprivileged_protected_attributes=1,
            privileged_protected_attributes=0
        )
        
        # Reweighting, round 1
        rw = Reweighing(unprivileged_groups=protected_groups,
                privileged_groups=privileged_groups)
        rw.fit(dataset)
        n_fav, n_unfav, n_p_fav, n_up_fav, n_p_unfav, n_up_unfav = self.get_all_ns(dataset, rw)

        pv_max = np.max(sensitive_attributes)
        pv_min = np.min(sensitive_attributes)
        pv_mid = (pv_max + abs(pv_min)) / 2
        pv_mid_pt = pv_max - pv_mid
        if n_unfav > n_fav:
            majority = 0 
        else:
            majority = 1 
        if n_p_fav < n_p_unfav:
            nsamp1 = int(n_p_unfav - n_p_fav)
            prot_grp1 = 1 
            if majority == 1: 
                cls_trk1 = 1 
            else:  
                cls_trk1 = 0
        if  n_p_unfav <= n_p_fav:
            nsamp1 = int(n_p_fav - n_p_unfav)
            prot_grp1 = 1 
            if majority == 1: 
                cls_trk1 = 0 
            else:  
                cls_trk1 = 1
        if n_up_fav < n_up_unfav:
            nsamp2 = int(n_up_unfav - n_up_fav)
            prot_grp2 = 0 
            if majority == 1: 
                cls_trk2 = 1 
            else:  
                cls_trk2 = 0
        if  n_up_unfav <= n_up_fav:
            nsamp2 = int(n_up_fav - n_up_unfav)
            prot_grp2 = 0 
            if majority == 1: 
                cls_trk2 = 0 
            else:  
                cls_trk2 = 1
        if nsamp1 < nsamp2:
            nsamp = nsamp1
            cls_trk = cls_trk1
            prot_grp = prot_grp1
        else:
            nsamp = nsamp2
            cls_trk = cls_trk2
            prot_grp = prot_grp2

        # Get protected/majority
        pv_max = np.max(sensitive_attributes)
        pv_min = np.min(sensitive_attributes)
        pv_mid = (pv_max + abs(pv_min)) / 2
        pv_mid_pt = pv_max - pv_mid
            
        oversampler = fair_sampler.FOS_1(
            proportion=self._proportion, n_neighbors=self._k
        ) 
        maj_min, nsamp = self.get_majority_class(y_train)
        full_data = pd.DataFrame(x_train) 
        full_data['demographic'] = self.get_binary_protected_privileged(demographic_attributes)
        full_data = np.array(full_data)
        X_samp, y_samp = oversampler.sample(full_data, y_train, -1, pv_mid_pt,
                            1, maj_min, nsamp,pv_max,pv_min)
            
        ######################
        if nsamp1 < nsamp2:
            nsamp = nsamp2
            cls_trk = cls_trk2
            prot_grp = prot_grp2
        else:
            nsamp = nsamp1
            cls_trk = cls_trk1
            prot_grp = prot_grp1
        maj_min = cls_trk 
        oversampler= fair_sampler.FOS_2(
            proportion=self._proportion, n_neighbors=self._k
        ) 
        x_sampled, y_sampled = oversampler.sample(X_samp, y_samp, -1, pv_mid_pt,
                            prot_grp, maj_min, nsamp)

        x_sampled = np.array(x_sampled)[:, :-1]

        return x_sampled, y_sampled, []
        
    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
