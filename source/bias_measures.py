import collections
import copy

import numpy as np
from fairlearn.metrics import MetricFrame
from scipy.stats import pearsonr

from source.training import evaluate_reg_accuracy


def PCC_score(attr, preds):

    dict_max = collections.defaultdict(list)
    if len(np.unique(attr)) == 2:
        r, p_val = pearsonr(attr, preds)
        r = abs(r)
    else:
        for val in np.unique(attr):
            attr_new = copy.deepcopy(attr)
            attr_new = np.where(attr_new == val, 1, attr_new)
            attr_new = np.where(attr_new != 1, 0, attr_new)

            r, p_val = pearsonr(attr_new, preds)
            dict_max['r'].append(abs(r))
            dict_max['p_val'].append(p_val)
        r = round(np.max(dict_max['r']), 3)
        p_val = round(dict_max['p_val'][np.where(dict_max['r'] == np.amax(dict_max['r']))[0][0]], 3)
        print(round(r, 3), round(p_val, 3))
    return round(r, 2), round(p_val, 2)


def equal_accuracy_diff(y_test, y_pred, sensitive_att_arr):
  grouped_on_EQA = MetricFrame(evaluate_reg_accuracy,
                              y_test, y_pred,
                              sensitive_features=sensitive_att_arr)
  results = grouped_on_EQA.difference(method='between_groups')
  return round(max(results), 2) if isinstance(results, list) else round(results, 2)

def test_bias(gender, test_preds, test_labels):
    print("PCC score: ", PCC_score(gender, test_preds))
    print("EA score: ", equal_accuracy_diff(test_labels, test_preds, gender))
    return PCC_score(gender, test_preds)[0], equal_accuracy_diff(test_labels, test_preds,
                                                                 gender), evaluate_reg_accuracy(test_labels, test_preds)
