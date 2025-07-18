import numpy as np
from mitigation.preprocessing.FAWOS.models.Feature import Feature
from mitigation.preprocessing.FAWOS.models.FeatureTypeCategorical import FeatureTypeCategorical
from mitigation.preprocessing.FAWOS.models.FeatureTypeContinuous import FeatureTypeContinuous
from mitigation.preprocessing.FAWOS.models.FeatureTypeOrdinal import FeatureTypeOrdinal
from mitigation.preprocessing.FAWOS.models.SensitiveClass import SensitiveClass
from mitigation.preprocessing.FAWOS.models.TargetClass import TargetClass
from mitigation.preprocessing.FAWOS.models.dataset import Dataset


class Training(Dataset):
    # https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

    def __init__(
        self, 
        # test_size, oversampling_factor, safe_weight, borderline_weight, rare_weight,
        settings, preprocessor_settings, x_train, y_train, demographics_train
    ):
        name = 'training'
        unique_targets = [nu for nu in np.unique(y_train)]
        unique_targets.sort()
        target_class = TargetClass('target', *unique_targets)
        discriminated = settings['pipeline']['attributes']['discriminated']
        privileged = [uv for uv in np.unique(demographics_train) if uv not in discriminated]
        sensitive_class = SensitiveClass(
            'demographic', 
            privileged, discriminated
        )

        type_function_map = {
            'continuous': self.get_continuous_feature,
            'ordinal': self.get_ordinal_feature,
            'categorical': self.get_categorical_feature
        }
        features = [
            type_function_map[settings['pipeline']['features'][feat_index]](
                np.array(x_train)[:, feat_index], feat_index
            ) for feat_index in range(len(x_train[0]))
        ]

        super().__init__(name, target_class, [sensitive_class], features, 0, preprocessor_settings['oversampling_factor'],
                         preprocessor_settings['safe_weight'], preprocessor_settings['borderline_weight'], preprocessor_settings['rare_weight'])

    def create_raw_transformed_dataset(self):
        raw_dataset = self.get_raw_dataset()

        old = raw_dataset['age'] >= 25  # http://ieeexplore.ieee.org/document/4909197/
        raw_dataset.loc[old, 'age'] = "adult"
        young = raw_dataset['age'] != "adult"
        raw_dataset.loc[young, 'age'] = "young"

        positive = raw_dataset['credit'] == 1
        raw_dataset.loc[positive, 'credit'] = "Positive"
        negative = raw_dataset['credit'] == 2
        raw_dataset.loc[negative, 'credit'] = "Negative"

        raw_transformed_dataset_filename = self.get_raw_transformed_dataset_filename()
        f = open(raw_transformed_dataset_filename, "w+")
        f.write(raw_dataset.to_csv(index=False))
        f.close()

    def get_ordinal_feature(self, feats, index):
        name = 'f{}'.format(index)
        order = [uv for uv in np.unique(feats)]
        order.sort()
        feature_type = FeatureTypeOrdinal(order)
        should_standardize = False
        return Feature(name, feature_type, should_standardize)

    def get_continuous_feature(self, feats, index):
        name = 'f{}'.format(index)
        feature_type = FeatureTypeContinuous()
        should_standardize = True
        return Feature(name, feature_type, should_standardize)

    def get_categorical_feature(self, feats, index):
        name = 'f{}'.format(index)
        feature_type = FeatureTypeCategorical()
        should_standardize = False
        return Feature(name, feature_type, should_standardize)
