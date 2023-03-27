# -*- coding: utf-8 -*-
"""
Functions to train the readout module to perform
tasks

@author: Estefany Suarez
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, RidgeClassifier
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from . import performance


def check_xy_dims(x, y):
    """
    Check that X,Y have the right dimensions
    # TODO
    """
    x_train, x_test = x
    y_train, y_test = y

    if ((x_train.ndim == 1) and (x_test.ndim == 1)):
        x_train = x_train[:, np.newaxis]
        x_test = x_test[:, np.newaxis]
    elif ((x_train.ndim > 2) and (x_test.ndim > 2)):
        x_train = x_train.squeeze()
        x_test = x_test.squeeze()

    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    return x_train, x_test, y_train, y_test


def regression(
    x, y, model=None, metric='r2_score', model_kws=None, metric_kws=None,
    **kwargs
):
    """
    Regression tasks
    # TODO
    """

    # get train and test samples
    x_train, x_test = x
    y_train, y_test = y

    # get sample_weights
    sample_weight_train, sample_weight_test = kwargs.pop(
        'sample_weight', (None, None))

    # specify default model
    if model is None:
        model = Ridge(**model_kws)

    # fit model on training data
    model.fit(x_train, y_train, sample_weight_train)

    # calculate model metric on test data
    if metric == 'score':
        # by default, use score method of model
        metric_value = model.score(
            x_test, y_test, sample_weight=sample_weight_test)
    else:
        func = getattr(performance, metric)
        y_pred = model.predict(x_test)
        metric_value = func(
            y_test, y_pred, sample_weight=sample_weight_test, **metric_kws)

    return metric_value, model


def multioutput_regression(*args, **kwargs):
    """
    #TODO
    """
    return regression(*args, **kwargs)


def classification(
    x, y, model=None, metric='accuracy_score', model_kws=None,
    metric_kws=None, **kwargs
):
    """
    Classification tasks
    # TODO
    """

    # get train and test samples
    x_train, x_test = x
    y_train, y_test = y

    # get sample_weights
    sample_weight_train, sample_weight_test = kwargs.pop(
        'sample_weight', (None, None))

    # specify default model
    if model is None:
        model = RidgeClassifier(**model_kws)

    # fit model on training data
    try:
        model.fit(x_train, y_train, sample_weight_train)
    except TypeError:
        # Note: multi-class classification uses OneVsRest strategy. OneVsRest
        # does not admit sample_weight arguments. Only non-zero sample 
        # points are used instead.
        model.fit(x_train[np.nonzero(y_train)], y_train[np.nonzero(y_train)])

        if metric == 'score':
            metric_value = model.score(x_test[np.nonzero(y_test)],
                                       y_test[np.nonzero(y_test)])
            return metric_value, model
        else:
            sample_weight_test = None

    # calculate model metric on test data
    if metric == 'score':
        # by default, use score method of model
        metric_value = model.score(
            x_test, y_test, sample_weight=sample_weight_test)
    else:
        func = getattr(performance, metric)
        y_pred = model.predict(x_test)
        metric_value = func(
            y_test, y_pred, sample_weight=sample_weight_test, **metric_kws)

    return metric_value, model


def binary_classification(*args, **kwargs):
    """
    #TODO
    """
    return classification(*args, **kwargs)


def multiclass_classification(*args, **kwargs):
    """
    #TODO
    """
    # model = kwargs.pop('model', None)
    # model_kws = kwargs.pop('model_kwargs', {})
    # if model is None:
    #     model = OneVsRestClassifier(RidgeClassifier(
    #         alpha=0.0, fit_intercept=False, **model_kws))

    # return classification(model=model, *args, **kwargs)
    return classification(*args, **kwargs)


def multilabel_classification(*args, **kwargs):
    """
    #TODO
    """
    print("Multi-label classification problems are not supported.")
    exit()


def select_model(y):
    """
    Select the right model depending on the nature of the target
    variable
    # TODO
    """
    if isinstance(y, list):
        y = np.vstack(y)

    if y.dtype in [np.float32, np.float64]:
        if y.squeeze().ndim == 1:
            return regression  # regression
        else:
            return multioutput_regression  # multilabel regression

    elif y.dtype in [np.int32, np.int64]:
        if y.squeeze().ndim == 1:
            if len(np.unique(y)) == 2:
                return binary_classification
            else:
                return multiclass_classification
        else:
            return multilabel_classification


def run_task(reservoir_states, y, metric, **kwargs):
    """
    #TODO
    Function that calls the method to run the task specified by 'task'

    Parameters
    ----------
    reservoir_states : tuple of numpy.ndarrays
        simulated reservoir states for training and test; the shape of each
        numpy.ndarray is n_samples, n_reservoir_nodes
    y : tuple of numpy.ndarrays
        training and test targets or output labels; the shape of each
        numpy.ndarray is n_samples, n_labels
    metric : str
    kwargs : other keyword arguments are passed to one of the following
        functions:
            memory_capacity_task(); delays=None, t_on=0
            pattern_recognition_task(); pttn_lens

    Returns
    -------
    df_res : pandas.DataFrame
        data frame with task scores
    """

    # print('\n PERFORMING TASK ...')

    # verify dimensions of x and y
    x_train, x_test, y_train, y_test = check_xy_dims(
        x=reservoir_states, y=y)

    # select training model
    func = select_model(y=y_train)

    # make metric a tuple to enable different metrics on the same model
    if isinstance(metric, str):
        metric = (metric,)

    # fit model
    metrics = dict()
    for m in metric:
        metrics[m], model = func(
            x=(x_train, x_test),
            y=(y_train, y_test),
            metric=m, **kwargs
            )

        # print(f'\t\t {m} = {metrics[m]}')

    df_res = pd.DataFrame(data=metrics, index=[0])

    return df_res, model
