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
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestRegressor

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

    model = Ridge(fit_intercept=False, alpha=0.5, **kwargs).fit(x_train, y_train)
    score = model.score(x_test, y_test)

    return score


def multiOutputRegression(x, y, **kwargs):
    """
    Multiple Output Regression tasks
    #TODO 
    """

    x_train, x_test, y_train, y_test = check_xy_dims(x,y)
    model = MultiOutputRegressor(Ridge(fit_intercept=False, alpha=0.5, **kwargs)).fit(x_train, y_train)
    
    # estimate score
    y_pred = model.predict(x_test)
    n_outputs = y_pred.shape[1]

    score = []
    for output in range(n_outputs):
        score.append(np.abs((np.corrcoef(y_test[:,output], y_pred[:,output])[0][1])))

    # for i in range(20):
    #     corr = np.round(np.corrcoef(y_test[:,i], y_pred[:,i])[0][1], 2)
    #     plt.scatter(y_test[:,i], y_pred[:,i], s=2, label=f'Tau={i+1} - {corr}')
    # plt.legend()
    # plt.show()
    # plt.close()

    # print('\n')
    # print(score)

    return np.sum(score)


def classification(x, y, **kwargs):
    """
    Classification tasks
    #TODO 
    """

    x_train, x_test, y_train, y_test = check_xy_dims(x,y)
    model = RidgeClassifier(alpha=0.0, fit_intercept=True, **kwargs).fit(x_train, y_train)
    
    # estimate score
    #TODO - average accuracy across classes or something like this
    # score = model.score(x_test, y_test)
    score = accuracy_score(y_test, model.predict(x_test))
   
    # ConfusionMatrixDisplay.from_predictions(y_test, model.predict(x_test))
    # plt.show()
    # plt.close()

    return score 


def multiOutputClassification(x, y, **kwargs):
    """
    Multiple Output Classification tasks
    #TODO 
    """

    x_train, x_test, y_train, y_test = check_xy_dims(x,y)
    model = MultiOutputRegressor(RidgeClassifier(alpha=0.5, fit_intercept=True, **kwargs)).fit(x_train, y_train)

    # estimate score
    #TODO - average accuracy across outputs????
    # score = model.score(x_test, y_test)
    score = accuracy_score(y_test, model.predict(x_test))
    
    # ConfusionMatrixDisplay.from_predictions(y_test, model.predict(x_test))
    # plt.show()
    # plt.close()

    return score 


def run_task(reservoir_states, target, **kwargs):
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

    func = select_stat_model(y=target)

    score = func(x=reservoir_states, y=target, **kwargs)
    
    df_res = pd.DataFrame(data=[score],
                          columns=['score'])

    return df_res


def select_stat_model(y):
    """
    Select the right model depending on the nature of the target
    variable
    #TODO 
    """
    if isinstance(y, tuple): y = y[0]

    if y.dtype in [np.float32, np.float64]:
        if y.ndim > 1: 
            return multiOutputRegression
        else: 
            return regression

    elif y.dtype in [np.int32, np.int64]:
        if y.ndim > 1: 
            return multiOutputClassification
        else:
            return classification


