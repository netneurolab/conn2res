# -*- coding: utf-8 -*-
"""
Functions for generating input/output data for tasks

@author: Estefany Suarez
"""

import numpy as np
from gym import spaces
import neurogym as ngym

def unbatch(x):
    """
        Transforms (batch, seq_len, features)
        to (batch*seq_len, features)
        #TODO
    """
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


def fetch_dataset(task, task_kwargs=None, n_trials=500, unbatch_data=True):
    """
        Fetches inputs and labels from the NeuroGym repository
        #TODO
    """
    if task_kwargs:
        # duration of a trial in time steps
        t_max_trial = int(np.sum([duration for _, duration in task_kwargs['timing'].items()])/task_kwargs['dt'])
        # print(f't_max_trial = {t_max_trial}')

        # default batch size
        batch_size = 10 

        # number of trials per batch
        # n_trials = np.ceil(n_trials/batch_size)

        # estimation of seq_len based on number of trials per batch
        seq_len = int(n_trials * t_max_trial) # this is the seq_len per batch
        print(f'seq_len = {seq_len}')

        # fetch inputs and labels from NeuroGym 
        dataset = ngym.Dataset(task+'-v0', env_kwargs=task_kwargs, 
                               batch_size=batch_size, seq_len=seq_len, 
                              )

    else: 
        dataset = ngym.Dataset(task+'-v0')

    env = dataset.env
    # print(env.ob.shape)
    # ob_size = env.observation_space.shape
    act_size = env.action_space.n
    # print(f'observation size = {ob_size}')
    # print(f'action size = {act_size}')

    inputs, labels = dataset()
    if unbatch_data:
        return unbatch(inputs), unbatch(labels)
    else:
        return inputs, labels


def split_dataset(data, frac_train=0.7):
    """
        Splits data into training and test sets according to
        'frac_train'
        #TODO
    """
    n_train = int(frac_train * data.shape[0])

    return data[:n_train], data[n_train:]