# -*- coding: utf-8 -*-
"""
Functions measure learning performance

@author: Estefany Suarez
"""

import numpy as np
from sklearn import metrics


def r2_score(
    y_true, y_pred, sample_weight=None, multioutput='uniform_average',
    **kwargs
):
    """
    Coefficient of determination of the regression R^2.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth target values.
    y_pred : numpy.ndarray
        Predicted target values.
    sample_weight : numpy.ndarray 
        Sample weights.
    multioutput : str 
        Defines aggregating of multiple output scores:
        
        ‘raw_values’ :
        Returns a full set of scores in case of 
        multioutput input.

        ‘uniform_average’ :
        Scores of all outputs are averaged with 
        uniform weight.

        ‘variance_weighted’ :
        Scores of all outputs are averaged, weighted 
        by the variances of each individual output.

    Returns
    -------
    score : float or ndarray of floats.
        A floating point value or an array of floating 
        point values, one for each individual target.
    """
    func = getattr(metrics, 'r2_score')
    return func(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)


def mean_squared_error(
    y_true, y_pred, sample_weight=None, multioutput='uniform_average',
    **kwargs
):
    """
    Mean squared error.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth target values.
    y_pred : numpy.ndarray
        Predicted target values.
    sample_weight : numpy.ndarray 
        Sample weights.
    multioutput : str 
        Defines aggregating of multiple output scores:
        
        ‘raw_values’ :
        Returns a full set of scores in case of 
        multioutput input.

        ‘uniform_average’ :
        Scores of all outputs are averaged with 
        uniform weight.

    Returns
    -------
    error : float or ndarray of floats.
        A floating point value or an array of floating 
        point values, one for each individual target.
    """
    func = getattr(metrics, 'mean_squared_error')
    return func(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput, squared=True)


def root_mean_squared_error(
    y_true, y_pred, sample_weight=None, multioutput='uniform_average',
    normalize=True, **kwargs
):
    """
    Root mean squared error. If normalize is True, the error is
    normalized by the variance of y_true.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth target values.
    y_pred : numpy.ndarray
        Predicted target values.
    sample_weight : numpy.ndarray 
        Sample weights.
    multioutput : str 
        Defines aggregating of multiple output scores:
        
        ‘raw_values’ :
        Returns a full set of scores in case of 
        multioutput input.

        ‘uniform_average’ :
        Scores of all outputs are averaged with 
        uniform weight.
    normalize : bool
        If True normalizes error by variance of y_true

    Returns
    -------
    error : float or ndarray of floats.
        A floating point value or an array of floating 
        point values, one for each individual target.
    """
    func = getattr(metrics, 'mean_squared_error')
    if normalize: 
        return func(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput, squared=False) / y_true.var(axis=0)
    else: 
        return func(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput, squared=False)


def mean_absolute_error(
    y_true, y_pred, sample_weight=None, multioutput='uniform_average',
    **kwargs
):
    """
    Mean absolute error.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth target values.
    y_pred : numpy.ndarray
        Predicted target values.
    sample_weight : numpy.ndarray 
        Sample weights.
    multioutput : str 
        Defines aggregating of multiple output scores:
        
        ‘raw_values’ :
        Returns a full set of scores in case of 
        multioutput input.

        ‘uniform_average’ :
        Scores of all outputs are averaged with 
        uniform weight.

    Returns
    -------
    error : float or ndarray of floats.
        A floating point value or an array of floating 
        point values, one for each individual target.
    """
    func = getattr(metrics, 'mean_absolute_error')
    return func(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)


def corrcoef(
    y_true, y_pred, multioutput='uniform_average', nonnegative=None,
    **kwargs
): 
    """
    Pearson's correlation coefficient.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth target values.
    y_pred : numpy.ndarray
        Predicted target values.
    multioutput : str
        Defines aggregating of multiple output scores.
    nonnegative : str
        Defines whether return the abosulate or 
        squared value:

        'squared' :
        Squares the values of the correlations

        'absolute' :
        Takes the absolute value of the correlations

    Returns
    -------
    score : float or ndarray of floats.
    A floating point value or an array of floating 
    point values, one for each individual target.

    """

    if y_true.ndim <= 1:
        r = np.corrcoef(y_true, y_pred)[0][1]
        if nonnegative == 'squared':
            return r**2
        elif nonnegative == 'absolute':
            return abs(r)
        else:
            return r

    elif y_true.ndim == 2:
        n_outputs = y_pred.shape[1]

        r = []
        for output in range(n_outputs):
            r.append(np.corrcoef(y_true[:, output], y_pred[:, output])[0][1])

        if nonnegative == 'squared':
            r = np.square(r)
        elif nonnegative == 'absolute':
            r = np.abs(r)

        if multioutput == 'uniform_average':
            return np.mean(r)
        elif multioutput == 'raw_values':
            return r
        elif multioutput == 'sum':
            return np.sum(r)


def accuracy_score(
    y_true, y_pred, sample_weight=None, normalize=True,
    **kwargs
):
    """
    Accuracy score.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth target values.
    y_pred : numpy.ndarray
        Predicted target values.
    sample_weight : numpy.ndarray 
        Sample weights.
    normalize : bool 
        If False, return the number of correctly classified 
        samples. Otherwise, return the fraction of correctly 
        classified samples.

    Returns
    -------
    score : float.
        A floating point value.
    """

    func = getattr(metrics, 'accuracy_score')
    return func(y_true, y_pred, sample_weight=sample_weight, normalize=normalize)


def balanced_accuracy_score(
    y_true, y_pred, sample_weight=None, adjusted=None,
    **kwargs
):
    """
    Balance accuracy score. Good to deal with 
    imbalanced datasets.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth target values.
    y_pred : numpy.ndarray
        Predicted target values.
    sample_weight : numpy.ndarray 
        Sample weights.
    adjusted : bool 
        When true, the result is adjusted for chance, so 
        that random performance would score 0, while 
        keeping perfect performance at a score of 1.
    
    Returns
    -------
    score : float.
        A floating point value.
    """

    func = getattr(metrics, 'balanced_accuracy_score')
    return func(y_true, y_pred, sample_weight=sample_weight, adjusted=adjusted)


def f1_score(
    y_true, y_pred, sample_weight=None, average='weighted', **kwargs
):
    """
    F1-score.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth target values.
    y_pred : numpy.ndarray
        Predicted target values.
    sample_weight : numpy.ndarray 
        Sample weights.
    average : str 
        This parameter is required for multiclass targets. 
        If None, the scores for each class are returned. 
        Otherwise, this determines the type of averaging 
        performed on the data:

        'micro':
        Calculate metrics globally by counting the total true 
        positives, false negatives and false positives.

        'macro':
        Calculate metrics for each label, and find their unweighted 
        mean. This does not take label imbalance into account.

        'weighted':
        Calculate metrics for each label, and find their average 
        weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
    
    Returns
    -------
    score : float.
        A floating point value.
    """

    func = getattr(metrics, 'f1_score')
    return func(y_true, y_pred, sample_weight=sample_weight, average=average, zero_division='warn')


def precision_score(
    y_true, y_pred, sample_weight=None, average='weighted',
    **kwargs
):
    """
    Precision score. 

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth target values.
    y_pred : numpy.ndarray
        Predicted target values.
    sample_weight : numpy.ndarray 
        Sample weights.
    average : str 
        This parameter is required for multiclass targets. 
        If None, the scores for each class are returned. 
        Otherwise, this determines the type of averaging 
        performed on the data:

        'micro':
        Calculate metrics globally by counting the total true 
        positives, false negatives and false positives.

        'macro':
        Calculate metrics for each label, and find their unweighted 
        mean. This does not take label imbalance into account.

        'weighted':
        Calculate metrics for each label, and find their average 
        weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
    
    Returns
    -------
    score : float.
        A floating point value.
    """

    func = getattr(metrics, 'precision_score')
    return func(y_true, y_pred, sample_weight=sample_weight, average=average, zero_division='warn')


def recall_score(
    y_true, y_pred, sample_weight=None, average='weighted',
    **kwargs
):
    """
    Recall score. 

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth target values.
    y_pred : numpy.ndarray
        Predicted target values.
    sample_weight : numpy.ndarray 
        Sample weights.
    average : str 
        This parameter is required for multiclass targets. 
        If None, the scores for each class are returned. 
        Otherwise, this determines the type of averaging 
        performed on the data:

        'micro':
        Calculate metrics globally by counting the total true 
        positives, false negatives and false positives.

        'macro':
        Calculate metrics for each label, and find their unweighted 
        mean. This does not take label imbalance into account.

        'weighted':
        Calculate metrics for each label, and find their average 
        weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
    
    Returns
    -------
    score : float.
        A floating point value.
    """

    func = getattr(metrics, 'recall_score')
    return func(y_true, y_pred, sample_weight=sample_weight, average=average, zero_division='warn')
