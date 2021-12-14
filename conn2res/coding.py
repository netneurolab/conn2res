# -*- coding: utf-8 -*-
"""
Intermediate functions before performing the task.

@author: Estefany Suarez
"""

import numpy as np
import pandas as pd

from .task import run_task

def encoder(task, reservoir_states, target, readout_modules=None,\
            readout_nodes=None):
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
    target : listo or tuple of numpy.ndarrays
        training and test targets or output labels; the shape of each
        numpy.ndarray is n_samples, n_labels
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
    if readout_modules is not None:

        # get unique modules identifiers
        module_ids = np.unique(readout_modules)

        # perform task using as readout nodes every module
        df_encoding = []
        for module in module_ids:

            # get set of readout nodes based on 'readout_modules'
            readout_nodes = np.where(readout_modules == module)[0]
            print(f'\t-------- Module : {module} with {len(readout_nodes)} nodes --------')

            # create temporal dataframe
            df_module = run_task(task=task,
                                 reservoir_states=(reservoir_states[0][:,readout_nodes], reservoir_states[1][:,readout_nodes]), # reservoir_states[:,:,readout_nodes],
                                 target=target,
                                 )

            df_module['module'] = module
            df_module['n_nodes'] = len(readout_nodes)

            #get encoding scores
            df_encoding.append(df_module)

        df_encoding = pd.concat(df_encoding)

    elif readout_nodes is not None:
        df_encoding = run_task(task=task,
                               reservoir_states=(reservoir_states[0][:,readout_nodes], reservoir_states[1][:,readout_nodes]),
                               target=target,
                               )

        df_encoding['n_nodes'] = len(readout_nodes)
    else:
        df_encoding = run_task(task=task,
                               reservoir_states=reservoir_states,
                               target=target,
                               )

    return df_encoding
