# -*- coding: utf-8 -*-
"""
Intermediate functions that handle the selection of 
the readout nodes before performing the task.

@author: Estefany Suarez
"""

import numpy as np
import pandas as pd
from .task import run_task


def get_modules(module_assignment):
    """
    #TODO
    """
    # get module ids 
    module_ids = np.unique(module_assignment)
    readout_modules = [np.where(module_assignment == i)[0] for i in module_ids]

    return  module_ids, readout_modules


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

    #TODO update readout_modules doc
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

        if isinstance(readout_modules, np.ndarray): 
            module_ids, readout_modules = get_modules(readout_modules)
        
        if isinstance(readout_modules, dict):
            readout_modules = list(readout_modules.values())
            module_ids      = list(readout_modules.keys())

        if isinstance(readout_modules, list):            
            module_ids = np.arange(len(readout_modules))
        
        # #TODO print error message
        # assert isinstance(readout_modules, list):
        #     "readout_modules should either be an array with
        #        module ids, a list of readout_nodes"

        # perform task using as readout nodes every module in readout_modules
        df_encoding = []
        for i, readout_nodes in enumerate(readout_modules):

            print(f'\t-------- Module : {module_ids[i]} with {len(readout_nodes)} nodes --------')

            # create temporal dataframe
            df_module = run_task(task=task,
                                 reservoir_states=(reservoir_states[0][:,readout_nodes], reservoir_states[1][:,readout_nodes]), # reservoir_states[:,:,readout_nodes],
                                 target=target,
                                 )

            df_module['module'] = module_ids[i]
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
