import pickle
import yaml
import logging

from crossvalidation.scorers.binary_scorer import BinaryClfScorer
from crossvalidation.scorers.preprocessing_scorer import PreProcessingScorer

from crossvalidation.splitters.splitter import Splitter
from crossvalidation.splitters.stratified_kfold import StratifiedKSplit

from crossvalidation.synchronised_postnested import SyncPostNestedXVal
from crossvalidation.gridsearches.postprocessing_gridsearch import PostProcessingGridSearch
from mitigation.postprocessing.kamiranpost import KamiranPostProcessor
from mitigation.postprocessing.nguyen import NguyenPostProcessor
from mitigation.postprocessing.pleiss import PleissPostProcessor
from mitigation.postprocessing.snel import SnelPostProcessor
from predictors.decision_tree import DTClassifier
from predictors.logistic_regression import LogisticRegressionClassifier
from predictors.portugal_garf import GARFClassifier
from predictors.oulad_smotenn import SmoteENNRFBoostClassifier
from predictors.xuetangx_svc import StandardScalingSVCClassifier

class PostProcessingCrossValMaker:
    """This script assembles the machine learning component and creates the post processing training pipeline according to:
    
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

    def _choose_postprocessor(self):
        if self._pipeline_settings['postprocessor'] == 'kamiran':
            self._postprocessor = KamiranPostProcessor
            self._postprocessing_gs = './configs/gridsearch/fake.yaml'
        
        if self._pipeline_settings['postprocessor'] == 'pleiss':
            self._postprocessor = PleissPostProcessor
            self._postprocessing_gs = './configs/gridsearch/postprocessing/gs_pleiss.yaml'

        if self._pipeline_settings['postprocessor'] == 'snel':
            self._postprocessor = SnelPostProcessor
            self._postprocessing_gs = './configs/gridsearch/fake.yaml'
            
    def _choose_model(self):
        logging.debug('model: {}'.format(self._pipeline_settings['predictor']))

        self._post_folds = self._settings['crossvalidation']['nfolds']
        ### Associte dataset to a model
        if self._pipeline_settings['dataset'] == 'xuetangx':
            self._pipeline_settings['predictor'] = 'sssvc'
            self._settings['crossvalidation']['nfolds'] = 5
            base_path = '{}baselines-models/xuetangx/dataxuetangx_modelx_mitigationdummyxx/2025_02_05_1/models/svc_fold_{}.pkl'
            result_path = '{}baselines-models/xuetangx/dataxuetangx_modelx_mitigationdummyxx/2025_02_05_1/results/trnonnested_xval_msssvc_modelseeds126_all_folds.pkl'.format(self._settings['paths']['experiments'])
            self._post_folds = 1
            

        if self._pipeline_settings['dataset'] == 'eedi':
            self._pipeline_settings['predictor'] = 'lr'
            self._settings['crossvalidation']['nfolds'] = 1
            base_path = '{}baselines-models/eedi/baselines/2025_02_18_0/models/lr_best_model_f{}.pkl'
            result_path = '{}baselines-models/eedi/baselines/2025_02_18_0/results/nested_xval_mlogr_modelseeds897_all_folds.pkl'.format(self._settings['paths']['experiments'])
            self._post_folds = 1

        if self._pipeline_settings['dataset'] == 'eedi2':
            self._pipeline_settings['predictor'] = 'lr'
            self._settings['crossvalidation']['nfolds'] = 1
            base_path = '{}baselines-models/eedi2/baselines/2025_02_18_0/models/models/lr_best_model_f{}.pkl'
            result_path = '{}baselines-models/eedi2/baselines/2025_02_18_0/models/results/nested_xval_mlogr_modelseeds984_all_folds.pkl.pkl'.format(self._settings['paths']['experiments'])
            self._post_folds = 1

        if self._pipeline_settings['dataset'] == 'oulad':
            self._pipeline_settings['predictor'] = 'smotennrf'
            base_path = '{}baselines-models/oulad/baselines/larger-grid/2025_02_17_0/models/smotenn_best_model_f{}.pkl'
            result_path = '{}baselines-models/oulad/baselines/larger-grid/2025_02_17_0/results/nested_xval_msmotennrf_modelseeds170_all_folds.pkl'.format(self._settings['paths']['experiments'])
            
        if self._pipeline_settings['dataset'] == 'student-performance-por':
            self._pipeline_settings['predictor'] = 'garf'
            base_path = '{}baselines-models/portugal/baselines/2025_02_04_0/models/garf_best_model_f{}.pkl'
            result_path = '{}baselines-models/portugal/baselines/2025_02_04_0/results/nested_xval_mgarf_modelseeds898_all_folds.pkl'.format(self._settings['paths']['experiments'])


        if self._pipeline_settings['dataset'] == 'student-performance-math':
            self._pipeline_settings['predictor'] = 'garf'
            base_path = '{}baselines-models/math/baselines/2025_02_04_0/models/garf_best_model_f{}.pkl'
            result_path = '{}baselines-models/math/baselines/2025_02_04_0/results/nested_xval_mgarf_modelseeds524_all_folds.pkl'.format(self._settings['paths']['experiments'])
            

        self.models = []
        self.train_indices = []
        self.test_indices = []
        with open(result_path, 'rb') as fp:
            results = pickle.load(fp)

        for f in range(self._post_folds):
            with open(base_path.format(self._settings['paths']['experiments'], f), 'rb') as fp:
                self.models.append(pickle.load(fp).model)

            self.train_indices.append([tidx for tidx in results[f]['train_index']])
            self.test_indices.append([tedx for tedx in results[f]['test_index']])


    def _choose_scorer(self):
        # if self._settings['pipeline']['nclasses'] == 2:
        self._scorer = BinaryClfScorer

    def _choose_crossvalidator(self):
        self._crossval = SyncPostNestedXVal
        self._gridsearch = PostProcessingGridSearch
        with open(self._postprocessing_gs, 'r') as fp:
            gs = yaml.load(fp, Loader=yaml.FullLoader)
            self._settings['crossvalidation']['post_cval']['paramgrid'] = gs


        self._crossval = self._crossval(
            self._settings, self._gridsearch, self._gs_splitter, self._postprocessor, self._scorer
        )
    
    def train(self, X:list, y:list, demographics:list):
        train_features = []
        train_ground_truths = []
        train_demographics = []
        test_features = []
        test_ground_truths = []
        test_demographics = []
        for f in range(self._post_folds):
            train_features.append(
                [X[tidx] for tidx in self.train_indices[f]]
            )
            test_features.append(
                [X[tedx] for tedx in self.test_indices[f]]
            )

            train_ground_truths.append([
                y[tidx] for tidx in self.train_indices[f]
            ])
            test_ground_truths.append([
                y[tedx] for tedx in self.test_indices[f]
            ])

            train_demographics.append([
                demographics[tidx] for tidx in self.train_indices[f]
            ])
            test_demographics.append([
                demographics[tedx] for tedx in self.test_indices[f]
            ])
        results = self._crossval.crossval(
            self.models,
            train_features, train_ground_truths,
            test_features, test_ground_truths,
            train_demographics, test_demographics
        )
        return results

    def _build_pipeline(self):
        self._choose_outer_splitter()
        self._choose_gridsearch_splitter()
        self._choose_postprocessor()
        self._choose_model()
        self._choose_scorer()
        self._choose_crossvalidator()
        
    
        