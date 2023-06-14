import sklearn
import numpy as np


def codemix_nlu_multi_fi(items):
    preds, golds = zip(*items)
    preds = np.array(preds)
    golds = np.array(golds)
    avg_f1 = sklearn.metrics.f1_score(golds, preds, average='macro')
    return avg_f1
