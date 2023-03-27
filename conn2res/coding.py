# -*- coding: utf-8 -*-
"""
Intermediate functions that handle the selection of
the readout nodes before performing the task.

@author: Estefany Suarez
"""

from re import L
import numpy as np
import pandas as pd
from scipy.linalg import block_diag

from .task import run_task


def get_modules(module_assignment):
    """
    # TODO
    """
    # get module ids
    module_ids = np.unique(module_assignment)
    readout_modules = [np.where(module_assignment == i)[0] for i in module_ids]

    return module_ids, readout_modules


def encoder(reservoir_states, target, readout_modules=None,
            readout_nodes=None, metric='score', return_model=False, **kwargs):
    """
    Function that defines the set(s) of readout nodes based on whether
    'readout_nodes', 'readout_modules' or None is provided. It then calls
    the 'run_task' function in the task module.

    The 'encoder' defines readout nodes for an encoding set up.

    Parameters
    ----------
    task : {'sgnl_recon', 'pttn_recog'}
        Task to be performed. Use 'sgnal_recon' to measure the fading memory
        property, or 'pttn_recog' to measure the separation property
    reservoir_states : list or tuple of numpy.ndarrays
        simulated reservoir states for training and test; the shape of each
        numpy.ndarray is n_samples, n_reservoir_nodes
    target : list or tuple of numpy.ndarrays
        training and test targets or output labels; the shape of each
        numpy.ndarray is n_samples, n_labels
    # TODO update readout_modules doc
    readout_modules : (N,) list or numpy.ndarray, optional
        an array that specifies to which module each of the nodes in the
        reservoir belongs to. N is the number of nodes in the reservoir. These
        modules are used to define sets of readout_nodes. If provided,
        readout_nodes is ignored.
    readout_nodes : (N,) list or numpy.ndarray, optional
        specifies the set of nodes from which the signals will be extracted from
        'reservoir_states' to perform 'task'
    kwargs : other keyword arguments are passed to one of the following
        functions:
            conn2res.task.run_task()

    Returns
    -------
    df_encoding : pandas.DataFrame
        data frame with task scores

    """

    # use multiple subsets of readout nodes designated by readout_modules 
    if readout_modules is not None:

        if isinstance(readout_modules, np.ndarray):
            module_ids, readout_modules = get_modules(readout_modules)

        elif isinstance(readout_modules, dict):
            module_ids = list(readout_modules.keys())
            readout_modules = list(readout_modules.values())

        elif isinstance(readout_modules, list):
            module_ids = np.arange(len(readout_modules))

        # perform task using as readout nodes every module in readout_modules
        df_encoding = []
        model = []
        for i, readout_nodes in enumerate(readout_modules):

            print(
                f'\t   -- Module : {module_ids[i]} with {len(readout_nodes)} nodes --')

            # create temporal dataframe
            df_module, model_module = run_task(reservoir_states=(
                reservoir_states[0][:, readout_nodes], reservoir_states[1][:, readout_nodes]), y=target, metric=metric, **kwargs)  # reservoir_states[:,:,readout_nodes],

            df_module['module'] = module_ids[i]
            df_module['n_nodes'] = len(readout_nodes)

            # get encoding scores
            df_encoding.append(df_module)
            model.append(model_module)

        df_encoding = pd.concat(df_encoding)

    elif readout_nodes is not None:
        # use a subset of output nodes as readout nodes
        df_encoding, model = run_task(reservoir_states=(
            reservoir_states[0][:, readout_nodes], reservoir_states[1][:, readout_nodes]), y=target, metric=metric, **kwargs)

        df_encoding['n_nodes'] = len(readout_nodes)

    else:
        # use all output nodes as readout nodes
        df_encoding, model = run_task(reservoir_states=reservoir_states,
                                      y=target, metric=metric, **kwargs)

    if return_model:
        return df_encoding, model
    else:
        return df_encoding


def time_average_samples(seq_len, data, sample_weight, operation=None):
    """
    Time averages reservoir states or subsamples labels before they are entered into the encoder (task)

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """

    # make sure that sample weights are in the right format
    if isinstance(sample_weight, tuple):
        sample_weight = list(sample_weight)
    if isinstance(sample_weight, np.ndarray):
        sample_weight = [sample_weight]

    # make sure that data are in the right format
    if isinstance(data, tuple):
        data = list(data)
    elif isinstance(data, np.ndarray):
        data = [data]

    if len(data) != len(sample_weight):
        raise ValueError(
            'data and sample_weight should have the same number of assigned variables')

    argout = []
    for i, _ in enumerate(data):
        # number of trials
        n_trials = int(data[i].shape[0] / seq_len)

        # reshape data
        data[i] = np.split(data[i], n_trials, axis=0)
        sample_weight[i] = np.split(sample_weight[i], n_trials, axis=0)

        tmp = []

        for j in range(n_trials):
            # get indices of unique weights (in order of occurence!)
            _, ia = np.unique(sample_weight[i][j], return_index=True)
            idx_sample = np.sort(ia)

            if operation == 'time_average':
                # reformat sample weight to enable matrix multiplication
                idx_sample = np.append(idx_sample, sample_weight[i][j].size)
                sample_weight[i][j] = block_diag(
                    *[sample_weight[i][j][idx_sample[ii]:idx_sample[ii+1]] for ii, _ in enumerate(idx_sample[:-1])])

                # time average based on sample weight
                tmp.append(sample_weight[i][j] @ data[i][j])

            elif operation == 'subsample':
                # subsample labels
                tmp.append(data[i][j][idx_sample])

        argout.append(np.vstack(tmp))

    return tuple(argout)
