import pickle
import yaml
import numpy as np
import argparse

#from utils.config_handler import ConfigHandler
from pipelines.feature_pipeline import FeaturePipeline

from plotters.data_report import DataReport
from plotters.statistics_report import StatisticsReport
from plotters.classification_report import ClassificationReport
from pipelines.crossvalidation_pipeline import CrossValMaker


def data_report(settings):
    settings['crossvalidation'] = {}
    pipeline = FeaturePipeline(settings)
    features, labels, demographics, settings = pipeline.load_sequences()

    reporter = DataReport(settings)
    reporter.plot_everything(features, demographics, labels)

def classification_report(settings):
    experiment_names = [
        # ['baselines-models/oulad', 'oulad'],
        ['baselines-models/xuetangx', 'xuetangx'],
        # ['baselines-models/math', 'student-performance-math'],
        # ['baselines-models/portugal', 'student-performance-por'],
        ['baselines-models/eedi', 'eedi'],
        ['baselines-models/eedi2', 'eedi2'],
        # ['preprocessing/oulad', 'oulad'],
        # ['preprocessing/xuetangx', 'xuetangx'],
        # ['preprocessing/math', 'student-performance-math'],
        # ['preprocessing/portugal', 'student-performance-por'],
        # ['inprocessing/oulad', 'oulad'],
        # ['inprocessing/xuetangx', 'xuetangx'],
        # ['inprocessing/math', 'student-performance-math'],
        # ['inprocessing/portugal', 'student-performance-por'],
        # ['postprocessing/oulad', 'oulad'],
        # ['postprocessing/xuetangx', 'xuetangx'],
        # ['postprocessing/math', 'student-performance-math'],
        # ['postprocessing/portugal', 'student-performance-por'],
    ]
    for exp in experiment_names:
        settings['experiment']['name'] = exp[0]
        settings['pipeline']['dataset'] = exp[1]
        settings = handle_configs(settings)
        # Format Settings
        settings['crossvalidation'] = {
            'scorer': {
                'scoring_metrics': settings['classification']['metrics'],
                'fairness_metrics': settings['fairness']['metrics']
            }
        }

        # Load data
        pipeline = FeaturePipeline(settings)
        features, labels, demographics, settings = pipeline.load_sequences()

        # Plot
        reporter = ClassificationReport(settings)
        reporter.plot(features, labels, demographics)

def compare(settings):
    experiment_names = [
        # ['baselines-models/oulad', 'oulad'],
        # ['baselines-models/xuetangx', 'xuetangx'],
        # ['baselines-models/math', 'student-performance-math'],
        # ['baselines-models/portugal', 'student-performance-por'],
        ['baselines-models/eedi', 'eedi'],
        # ['baselines-models/eedi2', 'eedi2'],
        
        # ['preprocessing/oulad', 'oulad'],
        # ['preprocessing/xuetangx', 'xuetangx'],
        # ['preprocessing/math', 'student-performance-math'],
        # ['preprocessing/portugal', 'student-performance-por'],
        # ['inprocessing/oulad', 'oulad'],
        # ['inprocessing/xuetangx', 'xuetangx'],
        # ['inprocessing/math', 'student-performance-math'],
        # ['inprocessing/portugal', 'student-performance-por'],
        # ['postprocessing/oulad', 'oulad'],
        # ['postprocessing/xuetangx', 'xuetangx'],
        # ['postprocessing/math', 'student-performance-math'],
        # ['postprocessing/portugal', 'student-performance-por'],
    ]
    for exp in experiment_names:
        settings['experiment']['name'] = exp[0]
        settings['pipeline']['dataset'] = exp[1]
        settings = handle_configs(settings)
        # Format Settings
        settings['crossvalidation'] = {
            'scorer': {
                'scoring_metrics': settings['classification']['metrics'],
                'fairness_metrics': settings['fairness']['metrics']
            }
        }

        # Load data
        pipeline = FeaturePipeline(settings)
        features, labels, demographics, settings = pipeline.load_sequences()

        # Plot

        reporter = StatisticsReport(settings)
        reporter.compare(features, labels, demographics)


def handle_configs(settings):
    if settings['pipeline']['dataset'] == 'xuetangx':
        settings['fairness']['attributes'] = ['gender']
        settings['n_bootstraps'] = 100
    if settings['pipeline']['dataset'] == 'eedi':
        settings['fairness']['attributes'] = ['gender']
        settings['n_bootstraps'] = 10
    if settings['pipeline']['dataset'] == 'eedi2':
        settings['fairness']['attributes'] = ['gender']
        settings['n_bootstraps'] = 10
    if settings['pipeline']['dataset'] == 'oulad':
        settings['fairness']['attributes'] = ['gender', 'disability']
    if settings['pipeline']['dataset'] == 'student-performance-por':
        settings['fairness']['attributes'] = ['sex']
    if settings['pipeline']['dataset'] == 'student-performance-math':
        settings['fairness']['attributes'] = ['sex']

    return settings

def main(settings):
    settings = handle_configs(settings)
    if settings['data']:
        data_report(settings)
    if settings['diagnostic']:
        classification_report(settings)
    if settings['compare']:
        compare(settings)

if __name__ == '__main__': 
    with open('./configs/plot_config.yaml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    parser = argparse.ArgumentParser(description='Plot the results')

    # Tasks
    parser.add_argument('--data', dest='data', default=False, action='store_true')
    parser.add_argument('--diagnostic', dest='diagnostic', default=False, action='store_true')
    parser.add_argument('--compare', dest='compare', default=False, action='store_true')

    # Functions
    parser.add_argument('--show', dest='show', default=False, action='store_true')
    parser.add_argument('--save', dest='save', default=False, action='store_true')
    
    # Update parameters
    settings.update(vars(parser.parse_args()))
    with open('./configs/default_config.yaml', 'r') as f:
        default_config = yaml.load(f, Loader=yaml.FullLoader)
    settings['plotter'] = default_config['plotter']
    main(settings)