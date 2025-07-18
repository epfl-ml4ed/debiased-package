import yaml
import logging
from crossvalidation.nonnested_crossvalidation import NonNestedXVal
# from ml.gridsearches.transfer_gridsearch import TransferSupervisedGridSearch

from crossvalidation.scorers.binary_scorer import BinaryClfScorer

from crossvalidation.splitters.splitter import Splitter
from crossvalidation.splitters.stratified_kfold import StratifiedKSplit

from crossvalidation.nested_crossvalidation import NestedXVal

from crossvalidation.gridsearches.supervised_gridsearch import SupervisedGridSearch

from predictors.decision_tree import DTClassifier
from predictors.logistic_regression import LogisticRegressionClassifier
from predictors.oulad_smotenn import SmoteENNRFBoostClassifier
from predictors.portugal_garf import GARFClassifier
from predictors.xuetangx_svc import StandardScalingSVCClassifier

class CrossValMaker:
    """This script assembles the machine learning component and creates the training pipeline according to:
    
        - splitter
        - sampler
        - model
        - xvalidator
        - scorer
    """
    
    def __init__(self, settings:dict):
        logging.debug('initialising the xval')
        self._name = 'training maker'
        self._notation = 'trnmkr'
        self._settings = dict(settings)
        self._experiment_root = self._settings['experiment']['root_name']
        self._experiment_name = settings['experiment']['name']
        self._pipeline_settings = self._settings['pipeline']
        
        self._build_pipeline()
        

    def get_gridsearch_splitter(self):
        return self._gs_splitter

    def get_scorer(self):
        return self._scorer

    def get_model(self):
        return self._model

    def _choose_splitter(self, splitter:str) -> Splitter:
        if splitter == 'stratkf':
            return StratifiedKSplit
    
    def _choose_outer_splitter(self):
        self._outer_splitter = self._choose_splitter(self._pipeline_settings['outer_splitter'])

    def _choose_gridsearch_splitter(self):
        self._gs_splitter = self._choose_splitter(self._pipeline_settings['gs_splitter'])
            
    def _choose_model(self):
        logging.debug('model: {}'.format(self._pipeline_settings['predictor']))

        if self._settings['baseline']:
            if self._pipeline_settings['dataset'] == 'xuetangx':
                self._pipeline_settings['predictor'] = 'sssvc'
                self._settings['crossvalidation']['nfolds'] = 5
                self._pipeline_settings['crossvalidator'] = 'nonnested'
            if self._pipeline_settings['dataset'] == 'eedi':
                self._pipeline_settings['predictor'] = 'lr'
                self._settings['crossvalidation']['nfolds'] = 3
            if self._pipeline_settings['dataset'] == 'eedi2':
                self._pipeline_settings['predictor'] = 'lr'
                self._settings['crossvalidation']['nfolds'] = 3
            if self._pipeline_settings['dataset'] == 'oulad':
                self._pipeline_settings['predictor'] = 'smotennrf'
            if self._pipeline_settings['dataset'] in ['student-performance-por', 'student-performance-math']:
                self._pipeline_settings['predictor'] = 'garf'

        if self._pipeline_settings['predictor'] == 'lstm':
            self._model = "model"
            gs_path = './configs/gridsearch/gs_lstm.yaml'

        if self._pipeline_settings['predictor'] == 'decision_tree':
            self._model = DTClassifier
            gs_path = './configs/gridsearch/gs_dt.yaml'

        if self._pipeline_settings['predictor'] == 'garf':
            self._model = GARFClassifier
            gs_path = './configs/gridsearch/gs_garf.yaml'
        
        if self._pipeline_settings['predictor'] == 'smotennrf':
            self._model = SmoteENNRFBoostClassifier
            gs_path = './configs/gridsearch/gs_smotennrf.yaml'

        if self._pipeline_settings['predictor'] == 'sssvc':
            self._model = StandardScalingSVCClassifier
            gs_path = './configs/gridsearch/gs_sssvc.yaml'

        if self._pipeline_settings['predictor'] == 'lr':
            self._model = LogisticRegressionClassifier
            gs_path = './configs/gridsearch/gs_lr.yaml'


        if self._settings['pipeline']['gridsearch'] != 'nogs':
            with open(gs_path, 'r') as fp:
                gs = yaml.load(fp, Loader=yaml.FullLoader)
                self._settings['crossvalidation']['nested_xval']['paramgrid'] = gs
                print('gridsearch')
                print(gs)
                    
    def _choose_scorer(self):
        self._scorer = BinaryClfScorer

    def _choose_gridsearcher(self):
        if self._pipeline_settings['gridsearch'] == 'supgs':
            self._gridsearch = SupervisedGridSearch
            
    def _choose_crossvalidator(self):
        if not self._settings['preprocessing']:
            self._choose_gridsearcher()
            if self._pipeline_settings['crossvalidator'] == 'nested' and self._settings['baseline']:
                self._crossval = NestedXVal
            if self._pipeline_settings['crossvalidator'] == 'nonnested' and self._settings['baseline']:
                self._crossval = NonNestedXVal
            self._crossval = self._crossval(self._settings, self._gridsearch, self._gs_splitter, self._outer_splitter, self._model, self._scorer)
    
    def train(self, X:list, y:list, demographics:list):
        results = self._crossval.crossval(X, y, demographics)
        return results

    def _build_pipeline(self):
        # self._choose_splitter()
        # self._choose_inner_splitter()
        self._choose_outer_splitter()
        self._choose_gridsearch_splitter()
        self._choose_model()
        self._choose_scorer()
        self._choose_crossvalidator()
        
    
        