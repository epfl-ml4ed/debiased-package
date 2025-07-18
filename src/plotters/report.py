
import os
import numpy as np

class Report:
    def __init__(self, settings:dict):
        self._settings = dict(settings)

        self._colours = [
            '#0C1618', '#004346', '#3F7267', '#7D9D8B', '#FAF4D3', '#E6D06A', '#DCBE35', '#D1AC00', '#7389AE',
            '#001219', '#005F73', '#0A9396', '#E9D8A6', '#EE9B00', '#CA6702', '#BB3E03', '#AE2012', '#9B2226'
        ]

    def get_colours(self, size, name='none'):
        if name == 'baseline':
            return ['#004346']

        return ['#004346' for _ in range(size)]
        
        # if size >= len(self._colours):
        #     replace = True
        # else:
        #     replace = False
        # surplus = np.abs(len(self._colours) - size)
        # return np.random.choice(self._colours, size=size, replace=replace)
        # # return ['#004346']

    def get_reporting_path(self, base_path):
        self._report_path = '../{}/{}/'.format(base_path, self._settings['pipeline']['dataset'])
        os.makedirs(self._report_path, exist_ok=True)