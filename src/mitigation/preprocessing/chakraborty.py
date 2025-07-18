import random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neighbors import NearestNeighbors as NN
from mitigation.preprocessing.preprocessor import PreProcessor

class ChakrabortyPreProcessor(PreProcessor):
    """Resampling fair smote
    use smote close to original instances, rebalance everything

    References:
        Chakraborty, J., Majumder, S., & Menzies, T. (2021, August). Bias in machine learning software: Why? how? what to do?. In Proceedings of the 29th ACM joint meeting on european software engineering conference and symposium on the foundations of software engineering (pp. 429-440).
        https://github.com/joymallyac/Fair-SMOTE/blob/master/Generate_Samples.py
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'chakraborty et al.'
        self._notation = 'chakraborty'
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

    def _get_ngbr(self, df, knn):
        rand_sample_idx = random.randint(0, df.shape[0] - 1)
        parent_candidate = df.iloc[rand_sample_idx]
        ngbr = knn.kneighbors(parent_candidate.values.reshape(1,-1),3,return_distance=False)
        candidate_1 = df.iloc[ngbr[0][0]]
        candidate_2 = df.iloc[ngbr[0][1]]
        candidate_3 = df.iloc[ngbr[0][2]]
        return parent_candidate,candidate_2,candidate_3

    def _generate_samples(self, no_of_samples,df):
        total_data = df.values.tolist()
        knn = NN(n_neighbors=5,algorithm='auto').fit(df)
        
        for _ in range(no_of_samples):
            cr = 0.8
            f = 0.8
            parent_candidate, child_candidate_1, child_candidate_2 = self._get_ngbr(df, knn)
            new_candidate = []
            for key,value in parent_candidate.items():
                if isinstance(parent_candidate[key], bool):
                    new_candidate.append(parent_candidate[key] if cr < random.random() else not parent_candidate[key])
                elif isinstance(parent_candidate[key], str):
                    new_candidate.append(random.choice([parent_candidate[key],child_candidate_1[key],child_candidate_2[key]]))
                elif isinstance(parent_candidate[key], list):
                    temp_lst = []
                    for i, each in enumerate(parent_candidate[key]):
                        temp_lst.append(parent_candidate[key][i] if cr < random.random() else
                                        int(parent_candidate[key][i] +
                                            f * (child_candidate_1[key][i] - child_candidate_2[key][i])))
                    new_candidate.append(temp_lst)
                else:
                    new_candidate.append(abs(parent_candidate[key] + f * (child_candidate_1[key] - child_candidate_2[key])))        
            total_data.append(new_candidate)
        
            final_df = pd.DataFrame(total_data)
            return final_df

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
        subgroups = ['{}_{}'.format(
                demographic_attributes[student], y_train[student]
            ) for student in range(len(demographic_attributes))
        ]
        counter_subgroups = Counter(subgroups)
        max_prop = np.max([v for v in counter_subgroups.values()])

        x_sampled = []
        y_sampled = []
        demo_sampled = []
        for demo in np.unique(demographic_attributes):
            for label in np.unique(y_train):
                dem_idx = [
                    i for i in range(len(y_train)) if demographic_attributes[i] == demo and y_train[i] == label
                ]
                n_samples = max_prop - len(dem_idx)

                if n_samples > 0:
                    sg_df = pd.DataFrame([x_train[didx] for didx in dem_idx])
                    new_samples = self._generate_samples(n_samples, sg_df)
                    x_sampled = x_sampled + [list(ns.values) for i, ns in new_samples.iterrows()]
                    y_sampled = y_sampled + [label for _ in range(len(new_samples))]
                    demo_sampled = demo_sampled + [demo for _ in range(len(new_samples))]
                else:
                    x_sampled = x_sampled + [x_train[didx] for didx in dem_idx]
                    y_sampled = y_sampled + [label for _ in range(len(dem_idx))]
                    demo_sampled = demo_sampled + [demo for _ in range(len(dem_idx))]

        self._information['counter'] = Counter(demo_sampled)
        return x_sampled, y_sampled, demo_sampled

    def get_information(self):
        """For each pre-processor, returns information worth saving for future results
        """
        return self._information
    
        
