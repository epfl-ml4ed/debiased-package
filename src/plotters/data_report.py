
from distutils.errors import DistutilsTemplateError
import os
import numpy as np
import pandas as pd

import plotly.express as px
from matplotlib import pyplot as plt

from plotters.report import Report
from collections import Counter


class DataReport(Report):
    """This script reports statistics on the datasets
    """
    
    def __init__(self, settings:dict):
        self._settings = dict(settings)
        self.get_reporting_path('data_reports')
        self._colours = [
            '#001219','#005F73', '#OA9396', '#94D2BD', '#E9D8A6', 
            '#EE9B00', '#CA6702', '#BB3E03', '#AE2012', '#9B2226']

    def get_percentage_df(self, attributes: list, demographic_name: str): 
        counts = Counter(attributes)
        perc = {k: v/len(attributes) for k, v in counts.items()}
        perc_df = pd.DataFrame()
        perc_df[demographic_name] = [k for k in perc.keys()]
        perc_df['percentages'] = perc_df[demographic_name].apply(lambda x: perc[x])
        perc_df['demographic_name'] = demographic_name
        perc_df['colours'] = [ch for ch in np.random.choice(self._colours, size=len(perc_df), replace=False)]
        return perc_df


    def plot_bars(self, percentages, name):
        fig = px.bar(
            percentages, x=name, y='percentages', color='colours', #color_discrete_sequence='colours',
            opacity=0.8, 
        )
        fig.update_layout(
            plot_bgcolor='white', showlegend=False
        )
        fig.update_xaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='white'
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='white',
            gridcolor='lightgrey'
        )
        #fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
        if self._settings['save']:
            fig.write_image('{}/percentages_{}.png'.format(self._report_path, name))

        if self._settings['show']:
            fig.show()

    def plot_everything(self, features, demographics, labels):

        # Labels
        label_perc = self.get_percentage_df(labels, 'labels')
        self.plot_bars(label_perc, 'labels')

        # demographics
        for demo in demographics[0].keys():
            demo_perc = self.get_percentage_df(
                [demographics[student][demo] for student in range(len(demographics))], demo
            )
            self.plot_bars(demo_perc, demo)
        


        