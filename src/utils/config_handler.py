import os
from os import path as pth
from datetime import datetime
import pickle

class ConfigHandler:
    def __init__(self, settings:dict):
        self._settings = settings
        
    def get_settings(self):
        return dict(self._settings)

    def get_experiment_name(self):
        """Creates the experiment name in the following path:
            '../experiments/experiment root/yyyy_mm_dd_index/'
            index being the first index in increasing order starting from 0 that does not exist yet.
            
            This function:
            - returns the experiment config name 
            - creates the folder with the right experiment name at ../experiments/experiment root/yyyy_mm_dd_index
            - dumps the config in the newly created folder

        Args:
            settings ([type]): read config

        Returns:
            [str]: Returns the name of the experiment in the format of 'yyyy_mm_dd_index'
        """
        ##### get data paths
        if self._settings['cluster']:
            self._settings['paths']['data'] = '/volume/cock/groupies/data'
            self._settings['paths']['experiments'] = '/volume/cock/groupies/experiments/'

        if self._settings['pipeline']['preprocessor'] != 'x':
            particule = 'pre'
            proc_name = self._settings['pipeline']['preprocessor']
        elif self._settings['pipeline']['inprocessor'] != 'x':
            particule = 'in'
            proc_name = self._settings['pipeline']['inprocessor']
        elif self._settings['pipeline']['postprocessor'] != 'x':
            particule = 'post'
            proc_name = self._settings['pipeline']['postprocessor']
        else:
            particule = 'baseline'
            proc_name = self._settings['pipeline']['predictor']
        path = '/data{}_model{}_mitigation{}{}/'.format(
            self._settings['pipeline']['dataset'].replace('.', '-'),
            self._settings['pipeline']['predictor'].replace('.', '-'),
            particule,
            proc_name
            # self._settings['pipeline']['preprocessor'], 
            # self._settings['pipeline']['inprocessor'],
            # self._settings['pipeline']['postprocessor']
        )
        today = datetime.today().strftime('%Y-%m-%d')
        today = today.replace('-', '_')
        starting_index = 0
        
        # first index
        experiment_name = '{}{}{}{}_{}/'.format(
            self._settings['paths']['experiments'],
            self._settings['experiment']['root_name'], path, today, starting_index
        )
        while (pth.exists(experiment_name)):
            starting_index += 1
            experiment_name = '{}{}{}{}_{}/'.format(
                self._settings['paths']['experiments'],
                self._settings['experiment']['root_name'], path, today, starting_index
            )
            
        self._experiment_path = experiment_name
        os.makedirs(self._experiment_path, exist_ok=True)
        self._settings['experiment']['name'] = self._experiment_path

        with open(self._settings['experiment']['name'] + 'config.pkl', 'wb') as fp:
            pickle.dump(self._settings, fp)

        return self._settings
      