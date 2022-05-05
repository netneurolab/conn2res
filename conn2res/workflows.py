# -*- coding: utf-8 -*-
"""
Functions (or workflows) to measure the temporal and pattern
memory capacity of a reservoir

@author: Estefany Suarez
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.linalg import eigh

from . import iodata, reservoir, coding

def memory_capacity_reservoir(conn, input_nodes, output_nodes, readout_modules=None,
                              readout_nodes=None, resname='EchoStateNetwork',
                              alphas=None, input_gain=1.0, tau_max=20, plot_res=False,
                              plot_title=None, **kwargs):
    """
    #TODO
    Function that measures the memory capacity of a reservoir as
    a function of the dynamics (modulated by varying the spectral
    radius of the connectivity matric with the parameter alpha).

    Parameters
    ----------
    #TODO

    Returns
    -------
    df_res : pandas.DataFrame
        data frame with task scores
    """

    # scale conenctivity weights between [0,1]
    conn = (conn - conn.min())/(conn.max() - conn.min())
    n_reservoir_nodes = len(conn)

    # normalize connectivity matrix by the spectral radius
    ew, _ = eigh(conn)
    conn  = conn / np.max(ew)

    # get dataset for memory capacity task
    x, y = iodata.fetch_dataset('MemoryCapacity', tau_max=tau_max)

    # create input connectivity matrix
    w_in = np.zeros((1, n_reservoir_nodes))
    w_in[:,input_nodes] = input_gain

    # evaluate network performance across various dynamical regimes
    if alphas is None: alphas = np.linspace(0,2,11)

    df = []
    for alpha in alphas[1:]:

        print(f'\n----------------------- alpha = {alpha} -----------------------')

        # instantiate an Echo State Network object
        network = reservoir.reservoir(name=resname,
                                      w_ih=w_in,
                                      w_hh=alpha * conn.copy(),
                                      **kwargs
                                     )

        # simulate reservoir states; select only output nodes
        rs = network.simulate(ext_input=x)[:,output_nodes]

        # remove first tau_max points from reservoir states
        rs = rs[tau_max:]

        # split data into training and test sets
        rs_train, rs_test = iodata.split_dataset(rs)
        y_train, y_test = iodata.split_dataset(y)

        # perform task
        try:
            df_ = coding.encoder(reservoir_states=(rs_train, rs_test),
                                target=(y_train, y_test),
                                readout_modules=readout_modules,
                                readout_nodes=readout_nodes
                                )

            df_['alpha'] = np.round(alpha, 3)

            # reorganize the columns
            if 'module' not in df_.columns: df_['module'] = 'NA'
            if 'n_nodes' not in df_.columns: df_['n_nodes'] = 'NA'
            df.append(df_[['module', 'n_nodes', 'alpha', 'score']])

        except:
            pass

    df = pd.concat(df, ignore_index=True)
    df['score'] = df['score'].astype(float)

    if plot_res: plot(df, plot_title)

    return df


def memory_capacity_memreservoir(conn, int_nodes, ext_nodes, gr_nodes, readout_modules=None,
                                 readout_nodes=None, resname='MSSNetwork',
                                 alphas=None, input_gain=1.0, tau_max=20, plot_res=False,
                                 plot_title=None, **kwargs):
    """
    #TODO
    Function that measures the memory capacity of a reservoir as
    a function of the dynamics (modulated by varying the spectral
    radius of the connectivity matric with the parameter alpha).

    Parameters
    ----------
    #TODO

    Returns
    -------
    df_res : pandas.DataFrame
        data frame with task scores
    """

    # binarize connectivity matrix
    conn = conn.astype(bool).astype(int)

    # # normalize connectivity matrix by the spectral radius
    # ew, _ = eigh(conn)
    # conn  = conn / np.max(ew)

    # get dataset for memory capacity task
    x, y, _ = iodata.fetch_dataset('MemoryCapacity', tau_max=tau_max)
    x = np.tile(x, (1,len(ext_nodes)))

    # evaluate network performance across various dynamical regimes
    if alphas is None: alphas = [1.0]

    df = []
    for alpha in alphas:

        print(f'\n----------------------- alpha = {alpha} -----------------------')

        # instantiate a Memristive Network object
        network = reservoir.reservoir(name=resname,
                                      w=alpha * conn.copy(),
                                      int_nodes=int_nodes,
                                      ext_nodes=ext_nodes,
                                      gr_nodes=gr_nodes,
                                      **kwargs
                                     )

        # simulate reservoir states; select only output nodes
        rs = network.simulate(Vext=x, **kwargs)[:,int_nodes]

        # remove first tau_max points from reservoir states
        rs = rs[tau_max:]

        # split data into training and test sets
        rs_train, rs_test = iodata.split_dataset(rs)
        y_train, y_test = iodata.split_dataset(y)

        # perform task
        try:
            df_ = coding.encoder(reservoir_states=(rs_train, rs_test),
                                 target=(y_train, y_test),
                                 readout_modules=readout_modules,
                                 readout_nodes=readout_nodes
                                 )

            df_['alpha'] = np.round(alpha, 3)

            # reorganize the columns
            if 'module' not in df_.columns: df['module'] = 'NA'
            if 'n_nodes' not in df_.columns: df_['n_nodes'] = 'NA'
            df.append(df_[['module', 'n_nodes', 'alpha', 'score']])

        except:
            pass

    df = pd.concat(df, ignore_index=True)
    df['score'] = df['score'].astype(float)

    if plot_res: plot(df, plot_title)

    return df


def memory_capacity(resname, **kwargs):

    if resname == 'EchoStateNetwork':
        memory_capacity_reservoir(resname=resname, **kwargs)
    elif resname == 'MSSNetwork':
        memory_capacity_memreservoir(resname=resname, **kwargs)


def plot(df, title):

    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(num=1, figsize=(12,10))
    ax = plt.subplot(111)

    n_modules = len(np.unique(df['module']))
    palette = sns.color_palette('husl', n_modules+1)[:n_modules]

    if 'VIS' in list(np.unique(df['module'])):
        hue_order =['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN']
    else:
        hue_order = None

    sns.lineplot(data=df, x='alpha', y='score',
                 hue='module',
                 hue_order=hue_order,
                 palette=palette,
                 markers=True,
                 ax=ax)

    sns.despine(offset=10, trim=True)

    if title is not None: plt.title(f'Memory Capacity - {title}')
    else: plt.title('Memory Capacity')

    plt.plot()
    plt.show()

    pass
