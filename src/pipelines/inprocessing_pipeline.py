from distutils.errors import LibError
import yaml
import logging
from crossvalidation.gridsearches.inprocessing_gridsearch import InProcessingGridSearch
from crossvalidation.inprocess_nested import InProcessingNestedXVal
from crossvalidation.inprocessing_nonnested import InProcessingNonNestedXVal
from crossvalidation.nonnested_crossvalidation import NonNestedXVal
# from ml.gridsearches.transfer_gridsearch import TransferSupervisedGridSearch

from crossvalidation.scorers.binary_scorer import BinaryClfScorer

from crossvalidation.splitters.splitter import Splitter
from crossvalidation.splitters.stratified_kfold import StratifiedKSplit

from crossvalidation.nested_crossvalidation import NestedXVal

from crossvalidation.gridsearches.supervised_gridsearch import SupervisedGridSearch
# from mitigation.inprocessing.alghamdi import AlghamdiInProcessor
from mitigation.inprocessing.chakraborty_in import ChakrabortyInProcessor
from mitigation.inprocessing.chen import ChenInProcessor
from mitigation.inprocessing.liu import LiuInProcessor
# from mitigation.inprocessing.do import DoInProcessor
# from mitigation.inprocessing.mary import MaryInProcessor
# from mitigation.inprocessing.chuang import ChuangInProcessor
from mitigation.inprocessing.oneto import OnetoInProcessor
from mitigation.inprocessing.zafar import ZafarInProcessor
from mitigation.inprocessing.fish import FishInProcessor
from mitigation.inprocessing.gao import GaoInProcessor
from mitigation.inprocessing.grari2 import Grari2InProcessor
from mitigation.inprocessing.iosifidis_adafair import IosifidisInProcessor
from mitigation.inprocessing.islam import IslamInProcessor
from mitigation.inprocessing.kilbertus import KilbertusInProcessor
from mitigation.inprocessing.schreuder import SchreuderInProcessor
from mitigation.inprocessing.test import TestInProcessor
# from mitigation.inprocessing.sikdar import SikdarInProcessor
# from mitigation.inprocessing.zhang import ZhangInProcessor
# from mitigation.inprocessing.wang_peerloss import WangPeerLossInProcessor

from predictors.decision_tree import DTClassifier
from predictors.logistic_regression import LogisticRegressionClassifier
from predictors.oulad_smotenn import SmoteENNRFBoostClassifier
from predictors.portugal_garf import GARFClassifier
from predictors.xuetangx_svc import StandardScalingSVCClassifier

class InProcessingCrossValMaker:
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

        if self._pipeline_settings['dataset'] == 'xuetangx':
            self._settings['crossvalidation']['nfolds'] = 5

        if self._pipeline_settings['dataset'] == 'eedi':
            self._settings['crossvalidation']['nfolds'] = 3

        if self._pipeline_settings['dataset'] == 'eedi2':
            self._settings['crossvalidation']['nfolds'] = 3


        if self._pipeline_settings['inprocessor'] == 'chakraborty': 
            self._inprocessor = ChakrabortyInProcessor
            self._inprocessing_gs = './configs/gridsearch/fake.yaml'

        if self._pipeline_settings['inprocessor'] == 'chakraborty_ab': 
            self._settings['inprocessors']['chakraborty']['goals'] = 'AB'
            self._inprocessor = ChakrabortyInProcessor
            self._inprocessing_gs = './configs/gridsearch/fake.yaml'

        if self._pipeline_settings['inprocessor'] == 'chakraborty_cd': 
            self._settings['inprocessors']['chakraborty']['goals'] = 'CD'
            self._inprocessor = ChakrabortyInProcessor
            self._inprocessing_gs = './configs/gridsearch/fake.yaml'

        if self._pipeline_settings['inprocessor'] == 'chen': 
            self._inprocessor = ChenInProcessor
            self._inprocessing_gs = './configs/gridsearch/inprocessing/gs_chen.yaml'

        if self._pipeline_settings['inprocessor'] == 'gao': 
            self._inprocessor = GaoInProcessor
            self._inprocessing_gs = './configs/gridsearch/inprocessing/gs_gao.yaml'

        if self._pipeline_settings['inprocessor'] == 'grari': 
            self._inprocessor = Grari2InProcessor
            self._inprocessing_gs = './configs/gridsearch/inprocessing/gs_grari.yaml'

        if self._pipeline_settings['inprocessor'] == 'islam': 
            self._inprocessor = IslamInProcessor
            self._inprocessing_gs = './configs/gridsearch/inprocessing/gs_islam.yaml'

        if self._pipeline_settings['inprocessor'] == 'kilbertus': 
            self._inprocessor = KilbertusInProcessor
            self._inprocessing_gs = './configs/gridsearch/inprocessing/gs_kilbertus.yaml'

        if self._pipeline_settings['inprocessor'] == 'liu': 
            self._inprocessor = LiuInProcessor
            self._inprocessing_gs = './configs/gridsearch/inprocessing/gs_liu.yaml'

        if self._pipeline_settings['inprocessor'] == 'oneto': 
            self._inprocessor = OnetoInProcessor
            self._inprocessing_gs = './configs/gridsearch/fake.yaml'

        if self._pipeline_settings['inprocessor'] == 'schreuder':
            self._inprocessor = SchreuderInProcessor
            self._inprocessing_gs = './configs/gridsearch/inprocessing/gs_schreuder.yaml'

        if self._pipeline_settings['inprocessor'] == 'zafar': 
            self._inprocessor = ZafarInProcessor
            self._inprocessing_gs = './configs/gridsearch/fake.yaml'

        if self._pipeline_settings['inprocessor'] == 'test':
            self._inprocessor = TestInProcessor
            self._inprocessing_gs = './configs/gridsearch/fake.yaml'

        

        

        ##### Cemetery
        # if self._pipeline_settings['inprocessor'] == 'alghamdi': 
        #     self._inprocessor = AlghamdiInProcessor
        #     self._inprocessing_gs = './configs/gridsearch/fake.yaml'

        # if self._pipeline_settings['inprocessor'] == 'chuang': 
        #     self._inprocessor = ChuangInProcessor
        #     self._inprocessing_gs = './configs/gridsearch/fake.yaml'

        # if self._pipeline_settings['inprocessor'] == 'do': 
        #     self._inprocessor = DoInProcessor
        #     self._inprocessing_gs = './configs/gridsearch/fake.yaml'

        # if self._pipeline_settings['inprocessor'] == 'iosifidis': 
        #     self._inprocessor = IosifidisInProcessor
        #     self._inprocessing_gs = './configs/gridsearch/inprocessing/gs_iosifidisadafair.yaml'

        # if self._pipeline_settings['inprocessor'] == 'li': 
        #     self._inprocessor = LiInProcessor
        #     self._inprocessing_gs = './configs/gridsearch/fake.yaml'

        # if self._pipeline_settings['inprocessor'] == 'mary': 
        #     self._inprocessor = MaryInProcessor
        #     self._inprocessing_gs = './configs/gridsearch/fake.yaml'

        # if self._pipeline_settings['inprocessor'] == 'sikdar':
        #     self._inprocessor = SikdarInProcessor
            # self._inprocessing_gs = './configs/gridsearch/fake.yaml'

        # if self._pipeline_settings['inprocessor'] == 'wangpeerloos':
        #     self._inprocessor = WangPeerLossInProcessor
        #     self._inprocessing_gs = './configs/gridsearch/fake.yaml'

        # if self._pipeline_settings['inprocessor'] == 'zhang':
        #     self._inprocessor = ZhangInProcessor
        #     self._inprocessing_gs = './configs/gridsearch/fake.yaml'


        if self._settings['pipeline']['gridsearch'] != 'nogs':
            with open(self._inprocessing_gs, 'r') as fp:
                gs = yaml.load(fp, Loader=yaml.FullLoader)
                self._settings['crossvalidation']['nested_xval']['paramgrid'] = gs
                print('gridsearch')
                print(gs)
                    
    def _choose_scorer(self):
        # if self._settings['pipeline']['nclasses'] == 2:
        self._scorer = BinaryClfScorer

    def _choose_gridsearcher(self):
        if self._pipeline_settings['gridsearch'] == 'supgs':
            self._gridsearch = InProcessingGridSearch
            
    def _choose_crossvalidator(self):
        if not self._settings['preprocessing']:
            self._choose_gridsearcher()
            if self._pipeline_settings['crossvalidator'] == 'nested':
                self._crossval = InProcessingNestedXVal

            if self._pipeline_settings['crossvalidator'] == 'nonnested':
                self._crossval = InProcessingNonNestedXVal
            self._crossval = self._crossval(self._settings, self._gridsearch, self._gs_splitter, self._outer_splitter, self._inprocessor, self._scorer)
    
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
        
    
        

        