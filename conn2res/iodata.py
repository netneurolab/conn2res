# -*- coding: utf-8 -*-
"""
Functions for generating input/output data for tasks

@author: Estefany Suarez
"""

import numpy as np
from gym import spaces
import neurogym as ngym


NEUROGYM_TASKS = [
                'AntiReach',
                # 'Bandit',
                'ContextDecisionMaking',
                # 'DawTwoStep',
                'DelayComparison',
                'DelayMatchCategory',
                'DelayMatchSample',
                'DelayMatchSampleDistractor1D',
                'DelayPairedAssociation',
                # 'Detection',  # TODO: Temporary removing until bug fixed
                'DualDelayMatchSample',
                # 'EconomicDecisionMaking',
                'GoNogo',
                'HierarchicalReasoning',
                'IntervalDiscrimination',
                'MotorTiming',
                'MultiSensoryIntegration',
                # 'Null',
                'OneTwoThreeGo',
                'PerceptualDecisionMaking',
                'PerceptualDecisionMakingDelayResponse',
                'PostDecisionWager',
                'ProbabilisticReasoning',
                'PulseDecisionMaking',
                'Reaching1D',
                'Reaching1DWithSelfDistraction',
                'ReachingDelayResponse',
                'ReadySetGo',
                'SingleContextDecisionMaking',
                'SpatialSuppressMotion',
                # 'ToneDetection'  # TODO: Temporary removing until bug fixed
            ]


CONN2RES_TASKS = [
                'MemoryCapacity',
                # 'TemporalPatternRecognition'  
                ]


def get_available_tasks():
    return NEUROGYM_TASKS


def unbatch(x):
    """
    Removes batch_size dimension from array

    Parameters
    ----------
    x : numpy.ndarray
        array with dimensions (seq_len, batch_size, features)

    Returns
    -------
    new_x : numpy.ndarray
        new array with dimensions (batch_size*seq_len, features)

    """
    #TODO right now it only works when x is (batch_first = False)
    return np.concatenate(x, axis=0)


def encode_labels(labels):
    """
        Binary encoding of categorical labels for classification 
        problems
        #TODO
    """

    enc_labels = -1 * np.ones((labels.shape[0], len(np.unique(labels))), dtype=np.int16)
    
    for i, label in enumerate(labels):
        enc_labels[i][label] = 1

    return enc_labels


def fetch_dataset(task, unbatch_data=True, **kwargs):
    """
    Fetches inputs and labels for 'task' from the NeuroGym 
    repository

    Parameters
    ----------
    task : {'AntiReach', 'Bandit', 'ContextDecisionMaking', 
    'DawTwoStep', 'DelayComparison', 'DelayMatchCategory',
    'DelayMatchSample', 'DelayMatchSampleDistractor1D',
    'DelayPairedAssociation', 'Detection', 'DualDelayMatchSample',
    'EconomicDecisionMaking', 'GoNogo', 'HierarchicalReasoning',
    'IntervalDiscrimination', 'MotorTiming', 'MultiSensoryIntegration',
    'OneTwoThreeGo', 'PerceptualDecisionMaking',
    'PerceptualDecisionMakingDelayResponse', 'PostDecisionWager',
    'ProbabilisticReasoning', 'PulseDecisionMaking',
    'Reaching1D', 'Reaching1DWithSelfDistraction',
    'ReachingDelayResponse', 'ReadySetGo',
    'SingleContextDecisionMaking', 'SpatialSuppressMotion',
    'ToneDetection'}
    Task to be performed
    unbatch_data : bool, optional
        If True, it adds an extra dimension to inputs and labels 
        that corresponds to the batch_size. Otherwise, it returns an
        observations by features array for the inputs, and a one
        dimensional array for the labels.

    Returns
    -------
    inputs : numpy.ndarray
        array of observations by features
    labels : numpy.ndarray
        unidimensional array of labels 
    """

    if task in NEUROGYM_TASKS:
        # create a Dataset object from NeuroGym
        dataset = ngym.Dataset(task+'-v0')
    
        # get inputs and labels for 'task'
        inputs, labels = dataset()
        
        if unbatch_data:
            return unbatch(inputs), unbatch(labels)

    elif task in CONN2RES_TASKS:
        # create a native conn2res Dataset 
        inputs, labels = create_dataset(task, **kwargs)

    return inputs, labels


def create_dataset(task, tau_max=20, **kwargs):

    if task == 'MemoryCapacity':
        x = np.random.uniform(-1, 1, (1000+tau_max))[:,np.newaxis]
        y = np.hstack([x[tau_max-tau:-tau][:,np.newaxis] for tau in range(1,tau_max+1)])

    return x,y


def split_dataset(data, frac_train=0.7):
    """
    Splits data into training and test sets according to
    'frac_train'

    Parameters
    ----------
    data : numpy.ndarray
        data array to be split
    frac_train : float, from 0 to 1
        fraction of samples in training set

    Returns
    -------
    training_set : numpy.ndarray
        array with training observations
    test_set     : numpy.ndarray
        array with test observations
    """
    n_train = int(frac_train * data.shape[0])

    return data[:n_train], data[n_train:]