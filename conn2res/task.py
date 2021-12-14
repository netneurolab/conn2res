# -*- coding: utf-8 -*-
"""
Functions to train the readout module to perform
tasks

@author: Estefany Suarez
"""

import numpy as np
import pandas as pd
import scipy as sp
import mdp

from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import Ridge, RidgeClassifier, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


def check_xy_dims(x,y):
    """
    Check that X,Y have the right dimensions
    #TODO 
    """
    x_train, x_test = x
    y_train, y_test = y

    if not ((x_train.squeeze().ndim == 2) and (x_test.ndim == 2)):
        x_train = x_train.squeeze()[:, np.newaxis]
        x_test  = x_test.squeeze()[:, np.newaxis]
    else:
        x_train = x_train.squeeze()
        x_test  = x_test.squeeze()

    y_train = y_train.squeeze()
    y_test  = y_test.squeeze()

    return x_train, x_test, y_train, y_test


def regression(x, y, **kwargs):
    """
    Regression tasks
    #TODO 
    """

    x_train, x_test, y_train, y_test = check_xy_dims(x,y)

    # model = LinearRegression(fit_intercept=False, **kwargs).fit(x_train, y_train)
    model = Ridge(fit_intercept=False, alpha=0.5, **kwargs).fit(x_train, y_train)
    score = model.score(x_test, y_test)

    # with np.errstate(divide='ignore', invalid='ignore'):
    #     # perf = np.abs(np.corrcoef(y_test, y_pred)[0][1])
    #     plt.scatter(y_test, y_pred)
    #     plt.show()
    #     plt.close()

    return score


def classification(x, y, **kwargs):
    """
    Classification tasks
    #TODO 
    """

    x_train, x_test, y_train, y_test = check_xy_dims(x,y)

    model = RidgeClassifier(alpha=0.0, fit_intercept=True, **kwargs)
    # model = MultiOutputRegressor(RidgeClassifier(alpha=0.5, fit_intercept=True, **kwargs))
    model.fit(x_train, y_train)

    # score = model.score(x_test, y_test)
    score = accuracy_score(y_test, model.predict(x_test))
    print(f'\tscore : {score}')

    return score 


def run_task(task, reservoir_states, target, **kwargs):
    """
    #TODO
    Function that calls the method to run the task specified by 'task'

    Parameters
    ----------
    task : {'regression', 'classification'}
    reservoir_states : tuple of numpy.ndarrays
        simulated reservoir states for training and test; the shape of each
        numpy.ndarray is n_samples, n_reservoir_nodes
    target : tuple of numpy.ndarrays
        training and test targets or output labels; the shape of each
        numpy.ndarray is n_samples, n_labels
    kwargs : other keyword arguments are passed to one of the following
        functions:
            memory_capacity_task(); delays=None, t_on=0
            pattern_recognition_task(); pttn_lens

    Returns
    -------
    df_res : pandas.DataFrame
        data frame with task scores
    """
    # score = regression(x=reservoir_states, y=target, **kwargs)

    score = classification(x=reservoir_states, y=target, **kwargs)
    df_res = pd.DataFrame(data=[score],
                          columns=['score'])

    return df_res


