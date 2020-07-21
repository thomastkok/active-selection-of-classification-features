from math import sqrt

import sklearn.metrics as skm


def mcc(y_true, y_pred):
    """Returns the Matthews correlation coefficient, or phi coefficient."""
    tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred).ravel()
    if 0 in [tp + fp, tp + fn, tn + fp, tn + fn]:
        return 0
    return (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
