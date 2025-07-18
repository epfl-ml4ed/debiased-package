import pickle
import yaml
import numpy as np
import argparse
from pipelines.inprocessing_pipeline import InProcessingCrossValMaker

from utils.config_handler import ConfigHandler
from pipelines.feature_pipeline import FeaturePipeline

from pipelines.preprocessing_pipeline import PreProcessingCrossValMaker
from pipelines.crossvalidation_pipeline import CrossValMaker
from pipelines.postprocessing_pipeline import PostProcessingCrossValMaker


def preprocessing(settings):
    settings['pipeline']['crossvalidator'] = 'sync'
    handler = ConfigHandler(settings)
    settings = handler.get_experiment_name()
    pipeline = FeaturePipeline(settings)
    features, labels, demographics, settings = pipeline.load_sequences()

    print(len(features), len(labels), len(demographics))
    for _ in range(settings['experiment']['model_seeds_n']):
        seed = np.random.randint(settings['experiment']['max_seed'])
        settings['seeds']['model'] = seed
        ml_pipeline = PreProcessingCrossValMaker(settings)
        ml_pipeline.train(features, labels, demographics)

def baseline(settings):
    settings['pipeline']['crossvalidator'] = 'nested'
    handler = ConfigHandler(settings)
    settings = handler.get_experiment_name()
    pipeline = FeaturePipeline(settings)
    features, labels, demographics, settings = pipeline.load_sequences()
    if settings['test']:
        features = features[:100]
        labels = labels[:100]
        demographics = demographics[:100]
    print(len(features), len(labels), len(demographics))

    for _ in range(settings['experiment']['model_seeds_n']):
        seed = np.random.randint(settings['experiment']['max_seed'])
        settings['seeds']['model'] = seed
        ml_pipeline = CrossValMaker(settings)
        ml_pipeline.train(features, labels, demographics)

def inprocessing(settings):
    settings['pipeline']['crossvalidator'] = 'nested'
    handler = ConfigHandler(settings)
    settings = handler.get_experiment_name()
    pipeline = FeaturePipeline(settings)
    features, labels, demographics, settings = pipeline.load_sequences()

    print(len(features), len(labels), len(demographics))
    for _ in range(settings['experiment']['model_seeds_n']):
        seed = np.random.randint(settings['experiment']['max_seed'])
        settings['seeds']['model'] = seed
        ml_pipeline = InProcessingCrossValMaker(settings)
        ml_pipeline.train(features, labels, demographics)

def postprocessing(settings):
    handler = ConfigHandler(settings)
    settings = handler.get_experiment_name()
    pipeline = FeaturePipeline(settings)
    features, labels, demographics, settings = pipeline.load_sequences()

    print(len(features), len(labels), len(demographics))
    for _ in range(settings['experiment']['model_seeds_n']):
        seed = np.random.randint(settings['experiment']['max_seed'])
        settings['seeds']['model'] = seed
        ml_pipeline = PostProcessingCrossValMaker(settings)
        ml_pipeline.train(features, labels, demographics)

def handle_arparse(settings):
    if settings['test']:
        settings['crossvalidation']['nfolds'] = 2

    if settings['dataset'] != '':
        if settings['dataset'] == 'fh2t':
            settings['pipeline']['dataset'] = 'fh2t'
        if settings['dataset'] == 'xuetangx':
            settings['pipeline']['dataset'] = 'xuetangx'
        if settings['dataset'] == 'eedi':
            settings['pipeline']['dataset'] = 'eedi'
        if settings['dataset'] == 'eedi2':
            settings['pipeline']['dataset'] = 'eedi2'
        if settings['dataset'] == 'oulad':
            settings['pipeline']['dataset'] = 'oulad'
        if settings['dataset'] == 'portugal':
            settings['pipeline']['dataset'] = 'student-performance-por'
        if settings['dataset'] == 'math':
            settings['pipeline']['dataset'] = 'student-performance-math'

    if settings['experiment_name'] != '':
        settings['experiment']['root_name'] = settings['experiment_name']

    if settings['pre_type'] != '':
        settings['pipeline']['preprocessor'] = settings['pre_type']
    if settings['post_type'] != '':
        settings['pipeline']['postprocessor'] = settings['post_type']
    if settings['in_type'] != '':
        settings['pipeline']['inprocessor'] = settings['in_type']

    return settings

def main(settings):
    settings = handle_arparse(settings)
    
    if settings['preprocessing']:
        preprocessing(settings)
    if settings['baseline']:
        baseline(settings)
    if settings['inprocessing']:
        inprocessing(settings)
    if settings['postprocessing']:
        postprocessing(settings)

if __name__ == '__main__': 
    with open('./configs/experiment_config.yaml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        
    parser = argparse.ArgumentParser(description='Plot the results')

    # Tasks
    parser.add_argument('--preprocessing', dest='preprocessing', default=False, action='store_true')
    parser.add_argument('--inprocessing', dest='inprocessing', default=False, action='store_true')
    parser.add_argument('--postprocessing', dest='postprocessing', default=False, action='store_true')
    parser.add_argument('--baseline', dest='baseline', default=False, action='store_true')
    parser.add_argument('--cluster', dest='cluster', default=False, action='store_true')
    parser.add_argument('--test', dest='test', default=False, action='store_true')

    # Arguments
    parser.add_argument('--dataset', dest='dataset', default='', action='store')
    parser.add_argument('--name', dest='experiment_name', default='', action='store')
    parser.add_argument('--pre', dest='pre_type', default='', action='store')
    parser.add_argument('--post', dest='post_type', default='', action='store')
    parser.add_argument('--in', dest='in_type', default='', action='store')

    
    settings.update(vars(parser.parse_args()))
    with open('./configs/default_config.yaml', 'r') as f:
        default_settings = yaml.load(f, Loader=yaml.FullLoader)
    keys = ['experiment', 'pipeline', 'crossvalidation']
    for k in keys:
        settings[k].update(default_settings[k])
    for k in default_settings:
        if k not in settings:
            settings[k] = default_settings[k]
    main(settings)