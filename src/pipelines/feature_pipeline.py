import pickle
import numpy as np
import pandas as pd

class FeaturePipeline:

    def __init__(self, settings):
        self._settings = dict(settings)

    def _get_demographics(self, features, available_demographics):
        return [
            {a_demo: features[idx][a_demo] for a_demo in available_demographics} for idx in features
        ]

    def _select_dataset(self):
        ####### Previous Encodings
        if self._settings['pipeline']['dataset'] == 'fh2t':
            # path = '../data/oulad/data_dictionary.pkl'
            # label = 'binary_label'
            # self._settings['pipeline']['nclasses'] = 2
            self._settings['pipeline']['attributes'] = {
                'mitigating': 'gender',
                'discriminated': '', # female
                'included': []
            }
            raise NotImplementedError

        if self._settings['pipeline']['dataset'] == 'xuetangx':
            path = '{}/xuetangx/data_dictionary.pkl'.format(self._settings['paths']['data'])
            label = 'binary_label'
            self._settings['pipeline']['nclasses'] = 2
            self._settings['pipeline']['attributes'] = {
                'mitigating': 'gender',
                'discriminated': '_2',
                'included': [-1, -2]
            }
            

        if self._settings['pipeline']['dataset'] == 'eedi':
            path = '{}/eedi/data_dictionary.pkl'.format(self._settings['paths']['data'])
            label = 'binary_label'
            self._settings['pipeline']['nclasses'] = 4
            self._settings['crossvalidation']['scorer']['scoring_metrics'] = ['roc']
            self._settings['crossvalidation']['scorer']['fairness_metrics'] = ['roc']
            self._settings['pipeline']['attributes'] = {
                'mitigating': 'gender',
                'discriminated': '_1._3', # any non male
                'included': []
            }
            self._settings['pipeline']['features'] = ['continuous' for _ in range(4)]

        if self._settings['pipeline']['dataset'] == 'eedi2':
            path = '{}/eedi/data_dictionary2.pkl'.format(self._settings['paths']['data'])
            label = 'binary_label'
            self._settings['pipeline']['nclasses'] = 2
            self._settings['pipeline']['attributes'] = {
                'mitigating': 'gender',
                'discriminated': '_1._3', # any non male
                'included': []
            }
            self._settings['pipeline']['features'] = ['continuous' for _ in range(4)]
        
        if self._settings['pipeline']['dataset'] == 'oulad':
            path = '{}/oulad/data_dictionary.pkl'.format(self._settings['paths']['data'])
            label = 'binary_label'
            self._settings['pipeline']['nclasses'] = 2
            self._settings['pipeline']['attributes'] = {
                'mitigating': 'gender.disability',
                'discriminated': '_1_1._1_0._0_1',
                'included': []
            }
        
        if self._settings['pipeline']['dataset'] == 'student-performance-por':
            path = '{}/student-performance-por/data_dictionary.pkl'.format(self._settings['paths']['data'])
            label = 'binary_label'
            self._settings['pipeline']['nclasses'] = 2
            self._settings['pipeline']['attributes'] = {
                'mitigating': 'sex',
                'discriminated': '_1',
                'included': []
            }
            self._settings['pipeline']['features'] = ['continuous' for _ in range(30)]

        if self._settings['pipeline']['dataset'] == 'student-performance-math':
            path = '{}/student-performance-math/data_dictionary.pkl'.format(self._settings['paths']['data'])
            label = 'binary_label'
            self._settings['pipeline']['nclasses'] = 2
            self._settings['pipeline']['attributes'] = {
                'mitigating': 'sex',
                'discriminated': '_1',
                'included': []
            }
        
        if self._settings['pipeline']['dataset'] == 'cyprus':
            path = '{}/cyprus/data_dictionary.pkl'.format(self._settings['paths']['data'])
            label = 'binary_label'
            self._settings['pipeline']['nclasses'] = 2
            self._settings['pipeline']['attributes'] = {
                'mitigating': 'sex.age',
                'discriminated': '_1_2._2_3'
            }
            self._settings['preprocessors']['luong']['continuous'] = [i for i in range(15)]
            self._settings['preprocessors']['luong']['discrete'] = [i for i in range(15, 27)]
            self._settings['preprocessors']['luong']['categorical'] = [i for i in range(27, 30)]


        with open(path, 'rb') as fp:
            data_dictionary = pickle.load(fp)
        features = [data_dictionary['data'][idx]['features'] for idx in data_dictionary['data']]
        if self._settings['pipeline']['dataset'] == 'xuetangx':
            fs = pd.DataFrame(features)
            fs = fs.drop(23, axis=1)
            features = np.array(fs).tolist()

        labels = [data_dictionary['data'][idx][label] for idx in data_dictionary['data']]
        demographics = self._get_demographics(data_dictionary['data'], data_dictionary['available_demographics'])
        return features, labels, demographics

    def _get_stratification_column(self):
        if self._settings['pipeline']['dataset'] == 'assistment':
            self._settings['crossvalidation']['stratifier_col'] = 'score'
        elif self._settings['pipeline']['dataset'] == 'cyprus':
            self._settings['crossvalidation']['stratifier_col'] = 'grade'
        elif self._settings['pipeline']['dataset'] == 'oulad':
            self._settings['crossvalidation']['stratifier_col'] = 'binary_label'
        elif self._settings['pipeline']['dataset'] == 'xuetangx':
            self._settings['crossvalidation']['stratifier_col'] = 'binary_label'
        elif self._settings['pipeline']['dataset'] == 'student-performance-por':
            self._settings['crossvalidation']['stratifier_col'] = 'binary_label'
        elif self._settings['pipeline']['dataset'] == 'student-performance-math':
            self._settings['crossvalidation']['stratifier_col'] = 'binary_label'
        elif self._settings['pipeline']['dataset'] == 'eedi':
            self._settings['crossvalidation']['stratifier_col'] = 'strat'
        elif self._settings['pipeline']['dataset'] == 'eedi2':
            self._settings['crossvalidation']['stratifier_col'] = 'strat'

    def load_sequences(self):
        self._get_stratification_column()
        features, labels, demographics = self._select_dataset()
        
        return features, labels, demographics, self._settings
        


