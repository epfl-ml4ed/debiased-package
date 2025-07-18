import pandas as pd

from mitigation.preprocessing.FAWOS.models.FeatureType import FeatureType
from mitigation.preprocessing.FAWOS.models.dataset import Dataset


class FeatureTypeContinuous(FeatureType):

    def encode(self,
               dataset: Dataset,
               feature_name: str,
               feature_values_train: pd.Series,
            #    feature_values_raw_test: pd.Series
    ):

        return super().encode(dataset, feature_name, feature_values_train)
        # , feature_values_raw_test)


    def inverse_encode(self, dataset: Dataset, feature_name: str, feature_value):

        return super().inverse_encode(dataset, feature_value)