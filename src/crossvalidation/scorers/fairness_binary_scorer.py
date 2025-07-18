import numpy as np
from crossvalidation.scorers.binary_scorer import BinaryClfScorer

class BinaryFairnessScorer(BinaryClfScorer):
    """This class takes care of computing fairness metrics

    Args:
        Scorer (Scorer): Inherits from Scorer
    """

    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'fairness scorer'
        self._notation = 'fscr'
        self._metrics = [
            'tp', 'fp', 'pp', 'fn', 'equalised_odds', 
            'equal_opportunity', 'demographic_parity'
        ] + self._settings['crossvalidation']['scorer']['scoring_metrics']

    def get_metrics(self):
        return self._metrics
        
    # Utility scores per demographics
    def _split_scores(self, y_true: list, y_pred: list, y_probs: list, demographics:list) -> dict:
        metrics = self._settings['crossvalidation']['scorer']['scoring_metrics']
        demos = np.unique(demographics)
        scores = {}
        for score in metrics:
            scores[score] = {}
            if score in self._score_dictionary:
                scores[score] = {}
                for demo in demos:
                    indices = [i for i in range(len(demographics)) if demographics[i] == demo]
                    demo_true = [y_true[idx] for idx in indices]
                    demo_pred = [y_pred[idx] for idx in indices]
                    demo_probs = [y_probs[idx] for idx in indices]
                    demo_values = [demographics[idx] for idx in indices]
                    scores[score][demo] = self._score_dictionary[score](demo_true, demo_pred, demo_probs, demo_values)
            scores[score]['all'] = self._score_dictionary[score](y_true, y_pred, y_probs, demographics)
        return scores

    # Rates per Demographics
    """Parts of the code dedicated to fairness metric measures
    demographics is the list of the corresponding demographics with regards to y_true
    """
    def _true_positive(self, y_true:list, y_pred:list, y_probs:list, demographics:list):
        demos = np.unique(demographics)
        scores = {}
        for demo in demos:
            try:
                indices = [i for i in range(len(demographics)) if demographics[i] == demo]
                demo_true = [y_true[idx] for idx in indices]
                demo_pred = [y_pred[idx] for idx in indices]

                positive = [i for i in range(len(demo_true)) if demo_true[i] == 1]
                yt = np.array([demo_true[i] for i in positive])
                yp = np.array([demo_pred[i] for i in positive])
                if len(np.array(yp).shape) == 2:
                    yp = [yyp[0] for yyp in yp]
                s = sum(yt == yp) / len(positive)
                scores[demo] = s
            except ZeroDivisionError:
                scores[demo] = -1
        return scores

    def _compute_single_tp(self, y_true:list, y_pred:list, y_probs:list) -> float:
        try:
            positive = [i for i in range(len(y_true)) if y_true[i] == 1]
            yt = np.array([y_true[i] for i in positive])
            yp = np.array([y_pred[i] for i in positive])
            s = sum(yt == yp) / len(positive)
        except ZeroDivisionError:
            s = -1
        return s

    def _compute_single_tn(self, y_true:list, y_pred:list, y_probs:list) -> float:
        try:
            negative = [i for i in range(len(y_true)) if y_true[i] == 0]
            yt = np.array([y_true[i] for i in negative])
            yp = np.array([y_pred[i] for i in negative])
            s = sum(yt == yp) / len(negative)
        except ZeroDivisionError:
            s = -1
        return s

    def _false_positive(self, y_true:list, y_pred:list, y_probs:list, demographics:list):
        demos = np.unique(demographics)
        scores = {}
        for demo in demos:
            try:
                indices = [i for i in range(len(demographics)) if demographics[i] == demo]
                demo_true = [y_true[idx] for idx in indices]
                demo_pred = [y_pred[idx] for idx in indices]

                negatives = [i for i in range(len(demo_true)) if demo_true[i] == 0]
                yt = np.array([demo_true[i] for i in negatives])
                yp = np.array([demo_pred[i] for i in negatives])
                if len(np.array(yp).shape) == 2:
                    yp = [yyp[0] for yyp in yp]
                s = sum(yt != yp) / len(negatives)
                scores[demo] = s
            except ZeroDivisionError:
                scores[demo] = -1
        return scores

    def _compute_single_fp(self, y_true:list, y_pred:list, y_probs:list) -> float:
        try:
            negatives = [i for i in range(len(y_true)) if y_true[i] == 0]
            yf = np.array([y_true[i] for i in negatives])
            yp = np.array([y_pred[i] for i in negatives])
            s = sum(yf != yp) / len(negatives)
        except ZeroDivisionError:
            s = -1
        return s

    def _positive_pred(self, y_true:list, y_pred:list, y_probs:list, demographics:list):
        demos = np.unique(demographics)
        scores = {}
        for demo in demos:
            indices = [i for i in range(len(demographics)) if demographics[i] == demo]
            demo_pred = [y_pred[idx] for idx in indices]
            positive = [yy for yy in demo_pred if yy == 1]
            s = len(positive) / len(indices)
            scores[demo] = s
        return scores

    def _compute_single_positive_pred(self, y_true:list, y_pred:list, y_probs:list):
        try:
            pred_pos = [i for i in range(len(y_pred)) if y_pred[i] == 1]
            s = len(pred_pos) / len(y_pred)
        except ZeroDivisionError:
            s = -1

        return s

    def _false_negative(self, y_true: list, y_pred:list, y_probs:list, demographics:list):
        demos = np.unique(demographics)
        scores = {}
        for demo in demos:
            try:
                indices = [i for i in range(len(demographics)) if demographics[i] == demo]
                demo_true = [y_true[idx] for idx in indices]
                demo_pred = [y_pred[idx] for idx in indices]

                pos_idx = [i for i in range(len(demo_true)) if demo_true[i] == 1]
                ps = len(pos_idx)
                tps = len([demo_pred[idx] for idx in pos_idx if demo_pred[idx] == 1])
                fns = ps - tps

                scores[demo] = fns / (fns + tps)
            except ZeroDivisionError:
                scores[demo] = -1
                continue
        return scores

    def _compute_single_fn(self, y_true:list, y_pred:list, y_probs:list) -> float: 
        try:
            pos_idx = [i for i in range(len(y_true)) if y_true[i] == 1]
            ps = len(pos_idx)
            tps = len([y_pred[idx] for idx in pos_idx if y_pred[idx] == 1])
            fns = ps - tps
            s = fns / (fns + tps)
        except ZeroDivisionError:
            s = -1
        return s

    # Mehrabi 
    def _equalised_odds(self, y_true, y_pred, y_probs, demographics):
        """Mehrabi:The definition of equalized odds, provided by Reference [63], states that “A predictor ˆY 
        satisfies equalized odds with respect to protected attribute A and outcome Y, if ˆ Y and A are independent 
        conditional on Y. P( ˆ Y=1|A=0,Y =y) = P( ˆ Y=1|A=1,Y =y) , y∈{0,1}.” This means that the probability of a 
        person in the positive class being correctly assigned a positive outcome and the probability of a person in a 
        negative class being incorrectly assigned a positive outcome should both be the same for the protected and 
        unprotected group members [145]. In other words, the equalized odds definition states that the protected and 
        unprotected groups should have equal rates for true positives and false positives.
        """
        true_positives = self._true_positive(y_true, y_pred, y_probs, demographics)
        false_positives = self._false_positive(y_true, y_pred, y_probs, demographics)

        unique_demo = np.unique(demographics)
        differences = {'tp': {}, 'fp': {}}
        max_diff_tp = 0
        max_diff_fp = 0
        for d in range(len(unique_demo)-1):
            for dd in range(d+1, len(unique_demo)):
                differences['tp']['{}_{}'.format(unique_demo[d], unique_demo[dd])] = np.abs(true_positives[unique_demo[d]] - true_positives[unique_demo[dd]])
                differences['fp']['{}_{}'.format(unique_demo[d], unique_demo[dd])] = np.abs(false_positives[unique_demo[d]] - false_positives[unique_demo[dd]])

                if differences['tp']['{}_{}'.format(unique_demo[d], unique_demo[dd])] > max_diff_tp:
                    max_diff_tp = differences['tp']['{}_{}'.format(unique_demo[d], unique_demo[dd])]
                if differences['fp']['{}_{}'.format(unique_demo[d], unique_demo[dd])] > max_diff_fp:
                    max_diff_fp = differences['fp']['{}_{}'.format(unique_demo[d], unique_demo[dd])]

        differences['max_diff'] = {'tp': max_diff_tp, 'fp': max_diff_fp}
        return differences

    def _equal_opportunity(self, y_true, y_pred, y_probs, demographics):
        """Mehrabi:“A binary predictor ˆ Y satisfies equal opportunity with respect to A and Y if 
        P( ˆ Y=1|A=0,Y=1) = P( ˆ Y=1|A=1,Y=1)” [63]. This means that the probability of a person in a 
        positive class being assigned to a positive outcome should be equal for both protected and unprotected 
        (female and male) group members [145]. In other words, the equal opportunity definition states that 
        the protected and unprotected groups should have equal true positive rates."""
        true_positives = self._true_positive(y_true, y_pred, y_probs, demographics)

        unique_demo = np.unique(demographics)
        differences = {}
        max_diff_tp = 0
        for d in range(len(unique_demo)-1):
            for dd in range(d+1, len(unique_demo)):
                differences['{}_{}'.format(unique_demo[d], unique_demo[dd])] = np.abs(true_positives[unique_demo[d]] - true_positives[unique_demo[dd]])
                if differences['{}_{}'.format(unique_demo[d], unique_demo[dd])] > max_diff_tp:
                    max_diff_tp = differences['{}_{}'.format(unique_demo[d], unique_demo[dd])]

        differences['max_diff'] = max_diff_tp
        return differences

    def _demographic_parity(self, y_true, y_pred, y_probs, demographics):
        """Mehrabi: Also known as statistical parity. “A predictor ˆ Y satisfies demographic parity if 
        P( ˆ Y|A=0)=P(ˆ Y|A = 1)” [48, 87]. The likelihood of a positive outcome [145] should be the same 
        regardless of whether the person is in the protected (e.g., female) group."""
        positive_preds = self._positive_pred(y_true, y_pred, y_probs, demographics)

        unique_demo = np.unique(demographics)
        differences = {}
        max_diff_pospred = 0
        for d in range(len(unique_demo)-1):
            for dd in range(d+1, len(unique_demo)):
                differences['{}_{}'.format(unique_demo[d], unique_demo[dd])] = np.abs(positive_preds[unique_demo[d]] - positive_preds[unique_demo[dd]])
                if differences['{}_{}'.format(unique_demo[d], unique_demo[dd])] > max_diff_pospred:
                    max_diff_pospred = differences['{}_{}'.format(unique_demo[d], unique_demo[dd])]

        differences['max_diff'] = max_diff_pospred
        return differences

    def get_fairness_scores(self, y_true:list, y_pred:list, y_probs:list, demographics:list) -> dict:
        """Returns dictionary with as first level keys the metrics, and as second level keys the
        demographics.

        Args:
            y_true (list): real labels
            y_pred (list): predicted labels (binary)
            y_probs (list): predicted labels (probability)
            demographics (list): corresponding demographics
            metrics (list): metrics to compute the scores for

        Returns:
            results (dict):
                score: 
                    demo0: value
                    ...
                    demon: value
        """
        
        scores = {}
        for attribute in self._settings['fairness']['attributes']:
            demographic_attributes = [dem[attribute] for dem in demographics]
            scores[attribute] = {}
            # Differences in TPR, FPR, PP, FN
            scores[attribute]['tp'] = self._true_positive(y_true, y_pred, y_probs, demographic_attributes)
            scores[attribute]['fp'] = self._false_positive(y_true, y_pred, y_probs, demographic_attributes)
            scores[attribute]['pp'] = self._positive_pred(y_true, y_pred, y_probs, demographic_attributes)
            scores[attribute]['fn'] = self._false_negative(y_true, y_pred, y_probs, demographic_attributes)

            # Differential rates of utility scores across different demographics such as roc, acc, etc.
            s = self._split_scores(y_true, y_pred, y_probs, demographic_attributes)
            scores[attribute].update(s)

            # Mehrabi
            scores[attribute]['equalised_odds'] = self._equalised_odds(y_true, y_pred, y_probs, demographic_attributes)
            scores[attribute]['equal_opportunity'] = self._equal_opportunity(y_true, y_pred, y_probs, demographic_attributes)
            scores[attribute]['demographic_parity'] = self._demographic_parity(y_true, y_pred, y_probs, demographic_attributes)

        return scores



        

