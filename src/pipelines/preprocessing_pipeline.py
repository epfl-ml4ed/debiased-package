import yaml
import logging
from crossvalidation.gridsearches.gridsearch import GridSearch
# from ml.gridsearches.transfer_gridsearch import TransferSupervisedGridSearch

from crossvalidation.scorers.binary_scorer import BinaryClfScorer
from crossvalidation.scorers.preprocessing_scorer import PreProcessingScorer

from crossvalidation.splitters.splitter import Splitter
from crossvalidation.splitters.stratified_kfold import StratifiedKSplit

from crossvalidation.nonnested_preprocessing import NonNestedPreProcessingXVal
from crossvalidation.nestedpreproc_nonnested import NestedPreProcXVal
from crossvalidation.synchronised_nested import SyncNestedXVal

from crossvalidation.gridsearches.preprocessing_gridsearch import PreProcessingGridSearch
from crossvalidation.gridsearches.sync_gridsearch import SynchronisedGridSearch
from mitigation.preprocessing.calders import CaldersPreProcessor
# from mitigation.preprocessing.celis import CelisPreProcessor
from mitigation.preprocessing.chakraborty import ChakrabortyPreProcessor
from mitigation.preprocessing.cock import CockPreProcessor
from mitigation.preprocessing.cohausz import CohauszPreProcessor
from mitigation.preprocessing.dablain import DablainPreProcessor
from mitigation.preprocessing.iosifidis_resampledattribute import IosifidisResamplingAttributePreProcessor
from mitigation.preprocessing.iosifidis_resampletarget import IosifidisResamplingTargetPreProcessor
from mitigation.preprocessing.iosifidis_smoteattribute import IosifidisSmoteAttributePreProcessor
from mitigation.preprocessing.iosifidis_smotetarget import IosifidisSmoteTargetPreProcessor
from mitigation.preprocessing.jiang import JiangPreProcessor
from mitigation.preprocessing.kamiran2 import Kamiran2PreProcessor
from mitigation.preprocessing.li import LiPreProcessor


from mitigation.preprocessing.notimplementable.disparate_impact_remover import DisparateImpactRemover
from mitigation.preprocessing.kamiran import KamiranPreProcessor
from mitigation.preprocessing.lahoti import LahotiPreProcessor
from mitigation.preprocessing.rebalance import RebalancePreProcessor
from mitigation.preprocessing.luong import LuongPreProcessor
from mitigation.preprocessing.alabdulmohsin import AlabdulmohsinPreProcessor
from mitigation.preprocessing.salazar import SalazarPreProcessor
from mitigation.preprocessing.singh import SinghSamplePreProcessor
from mitigation.preprocessing.notimplementable.singh_smote import SinghSmotePreProcessor
from mitigation.preprocessing.smote import SmotePreProcessor
from mitigation.preprocessing.notimplementable.zehlike import ZehlikePreProcessor
from mitigation.preprocessing.yan import YanPreProcessor
from mitigation.preprocessing.zelaya_over import ZelayaOverPreProcessor
from mitigation.preprocessing.zelaya_psp import ZelayaPSPPreProcessor
from mitigation.preprocessing.zelaya_smote import ZelayaSMOTEPreProcessor
from mitigation.preprocessing.zelaya_under import ZelayaUnderPreProcessor
from mitigation.preprocessing.zemel import ZemelPreProcessor

from predictors.decision_tree import DTClassifier
from predictors.logistic_regression import LogisticRegressionClassifier
from predictors.portugal_garf import GARFClassifier
from predictors.oulad_smotenn import SmoteENNRFBoostClassifier
from predictors.xuetangx_svc import StandardScalingSVCClassifier

class PreProcessingCrossValMaker:
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

    def _choose_preprocessor(self):
        if self._pipeline_settings['preprocessor'] == 'disparate_impact':
            self._preprocessor = DisparateImpactRemover
            gs_path = './configs/gridsearch/gs_disparate_impact.yaml'

        if self._pipeline_settings['preprocessor'] == 'rebalance':
            self._preprocessor = RebalancePreProcessor
            self._preprocessing_gs = './configs/gridsearch/preprocessing/gs_rebalance.yaml'

        ####### Relabelling
        if self._pipeline_settings['preprocessor'] == 'alabdulmohsin':
            self._preprocessor = AlabdulmohsinPreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/preprocessing/gs_alabdulmohsin.yaml'

        if self._pipeline_settings['preprocessor'] == 'kamiran':
            self._preprocessor = KamiranPreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/fake.yaml' # dummy path because pre-processor does not have any hyperparameters
        
        if self._pipeline_settings['preprocessor'] == 'luong':
            self._preprocessor = LuongPreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/preprocessing/gs_luong.yaml'

        ####### Representation
        if self._pipeline_settings['preprocessor'] == 'cohausz':
            self._preprocessor = CohauszPreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/fake.yaml'

        if self._pipeline_settings['preprocessor'] == 'lahoti':
            self._preprocessor = LahotiPreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/preprocessing/gs_lahoti.yaml'

        if self._pipeline_settings['preprocessor'] == 'li':
            self._preprocessor = LiPreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/fake.yaml'
        
        if self._pipeline_settings['preprocessor'] == 'zemel':
            self._preprocessor = ZemelPreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/preprocessing/gs_zemel.yaml'

        ####### Sampling
        if self._pipeline_settings['preprocessor'] == 'calders':
            self._preprocessor = CaldersPreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/preprocessing/gs_calders.yaml'

        # if self._pipeline_settings['preprocessor'] == 'celis': # Need to find installation that works
        #     self._preprocessor = CelisPreProcessor
        #     self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
        #     self._preprocessing_gs = './configs/gridsearch/preprocessing/gs_celis.yaml'

        if self._pipeline_settings['preprocessor'] == 'chakraborty':
            self._preprocessor = ChakrabortyPreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/fake.yaml'

        if self._pipeline_settings['preprocessor'] == 'chawla':
            self._preprocessor = SmotePreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/preprocessing/gs_chawla.yaml'

        if self._pipeline_settings['preprocessor'] == 'cock':
            self._preprocessor = CockPreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/preprocessing/gs_cock.yaml'

        if self._pipeline_settings['preprocessor'] == 'dablain':
            self._preprocessor = DablainPreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/preprocessing/gs_dablain.yaml'

        if self._pipeline_settings['preprocessor'] == 'iosifidisrestarget':
            self._preprocessor = IosifidisResamplingTargetPreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/fake.yaml'

        if self._pipeline_settings['preprocessor'] == 'iosifidisresattribute':
            self._preprocessor = IosifidisResamplingAttributePreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/fake.yaml'

        if self._pipeline_settings['preprocessor'] == 'iosifidissmotetarget':
            self._preprocessor = IosifidisSmoteTargetPreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/fake.yaml'

        if self._pipeline_settings['preprocessor'] == 'iosifidissmoteattribute':
            self._preprocessor = IosifidisSmoteAttributePreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/fake.yaml'

        if self._pipeline_settings['preprocessor'] == 'jiang': # needs python < 3.11
            self._preprocessor = JiangPreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/preprocessing/gs_jiang.yaml'


        if self._pipeline_settings['preprocessor'] == 'kamiran2':
            self._preprocessor = Kamiran2PreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/fake.yaml' # dummy path because pre-processor does not have any hyperparameters

        if self._pipeline_settings['preprocessor'] == 'salazar':
            self._preprocessor = SalazarPreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/preprocessing/gs_fawos.yaml'
        
        if self._pipeline_settings['preprocessor'] == 'singh':
            self._preprocessor = SinghSamplePreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/preprocessing/gs_fawos.yaml'

        if self._pipeline_settings['preprocessor'] == 'yan':
            self._preprocessor = YanPreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/preprocessing/gs_yan.yaml'

        if self._pipeline_settings['preprocessor'] == 'zelayaunder':
            self._preprocessor = ZelayaUnderPreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/preprocessing/gs_zelaya.yaml'
        
        if self._pipeline_settings['preprocessor'] == 'zelayaover':
            self._preprocessor = ZelayaOverPreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/preprocessing/gs_zelaya.yaml'
        
        if self._pipeline_settings['preprocessor'] == 'zelayasmote':
            self._preprocessor = ZelayaSMOTEPreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/preprocessing/gs_zelaya.yaml'
        
        if self._pipeline_settings['preprocessor'] == 'zelayapsp':
            self._preprocessor = ZelayaPSPPreProcessor
            self._settings['crossvalidation']['nestedpreproc_xval']['optim_scoring'] = 'roc'
            self._preprocessing_gs = './configs/gridsearch/preprocessing/gs_zelaya.yaml'
        
        
            
    def _choose_model(self):
        logging.debug('model: {}'.format(self._pipeline_settings['predictor']))
        ### Associte dataset to a model
        if self._pipeline_settings['dataset'] == 'xuetangx':
            self._pipeline_settings['predictor'] = 'sssvc'
            self._settings['crossvalidation']['nfolds'] = 5
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

        ### Select the model
        if self._pipeline_settings['predictor'] == 'lstm':
            self._model = "model"
            self._model_gs = './configs/gridsearch/gs_lstm.yaml'

        if self._pipeline_settings['predictor'] == 'decision_tree':
            self._model = DTClassifier
            self._model_gs = './configs/gridsearch/gs_dt.yaml'

        if self._pipeline_settings['predictor'] == 'garf':
            self._model = GARFClassifier
            if self._pipeline_settings['dataset'] == 'student-performance-por':
                self._model_gs = './configs/gridsearch/gs_garf_port.yaml'

            if self._pipeline_settings['dataset'] == 'student-performance-math':
                self._model_gs = './configs/gridsearch/gs_garf_math.yaml'

        if self._pipeline_settings['predictor'] == 'smotennrf':
            self._model = SmoteENNRFBoostClassifier
            self._model_gs = './configs/gridsearch/gs_smotennrf.yaml'

        if self._pipeline_settings['predictor'] == 'sssvc':
            self._model = StandardScalingSVCClassifier
            self._model_gs = './configs/gridsearch/gs_sssvc.yaml'

        if self._pipeline_settings['predictor'] == 'lr':
            self._model = LogisticRegressionClassifier
            self._model_gs = './configs/gridsearch/gs_lr.yaml'
            

    def _choose_scorer(self):
        # if self._settings['pipeline']['nclasses'] == 2:
        self._scorer = BinaryClfScorer

    def _choose_crossvalidator(self):
        if self._pipeline_settings['crossvalidator'] == 'nonnested':
            self._crossval = NonNestedPreProcessingXVal
            self._gridsearch = GridSearch # decoy, not needed
            
        if self._pipeline_settings['crossvalidator'] == 'nestedpreproc':
            self._crossval = NestedPreProcXVal
            self._gridsearch = PreProcessingGridSearch
            with open(self._preprocessing_gs, 'r') as fp:
                gs = yaml.load(fp, Loader=yaml.FullLoader)
                self._settings['crossvalidation']['nestedpreproc_xval']['paramgrid'] = gs
                    

        if self._pipeline_settings['crossvalidator'] == 'sync':
            self._crossval = SyncNestedXVal
            self._gridsearch = SynchronisedGridSearch
            with open(self._preprocessing_gs, 'r') as fp:
                gs = yaml.load(fp, Loader=yaml.FullLoader)
                self._settings['crossvalidation']['sync_xval']['paramgrid'] = gs
            with open(self._model_gs, 'r') as fp:
                model_gs = yaml.load(fp, Loader=yaml.FullLoader)
                self._settings['crossvalidation']['sync_xval']['paramgrid'].update(model_gs)

        self._crossval = self._crossval(self._settings, self._gridsearch, self._gs_splitter, self._outer_splitter, self._preprocessor, self._model, self._scorer)
    
    def train(self, X:list, y:list, demographics:list):
        results = self._crossval.crossval(X, y, demographics)
        return results

    def _build_pipeline(self):
        self._choose_outer_splitter()
        self._choose_gridsearch_splitter()
        self._choose_preprocessor()
        self._choose_model()
        self._choose_scorer()
        self._choose_crossvalidator()
        
    
        