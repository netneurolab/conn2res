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

def memory_capacity(conn, input_nodes, output_nodes, readout_modules=None, 
                    readout_nodes=None, resname='EchoStateNetwork', 
                    alphas=None, input_gain=1.0, tau_max=20, plot_res=False):
    """
    #TODO
    Function that measures the memory capacity of a reservoir as 
    a function of the dynamics (modulated by varying alpha) 

    Parameters
    ----------
    #TODO 

    Returns
    -------
    df_res : pandas.DataFrame
        data frame with task scores
    """


    # scale conenctivity weights between [0,1]
    conn = (conn-conn.min())/(conn.max()-conn.min())
    n_reservoir_nodes = len(conn)

    # normalize connectivity matrix by the spectral radius
    ew, _ = eigh(conn)
    conn  = conn/np.max(ew)

    # get dataset for memory capacity task 
    x, y = iodata.fetch_dataset('MemoryCapacity', tau_max=tau_max)

    # create input connectivity matrix
    w_in = np.zeros((1, n_reservoir_nodes))
    w_in[:,input_nodes] = input_gain 

    # evaluate network performance across various dynamical regimes
    if alphas is None: alphas = np.linspace(0,2,11) 
    
    df = []
    for alpha in alphas: 

        print(f'\n----------------------- alpha = {alpha} -----------------------')

        # instantiate an Echo State Network object
        network = reservoir.reservoir(name=resname,
                                      w_ih=w_in,
                                      w_hh=alpha*conn.copy(),
                                      activation_function='tanh'
                                    )

        # simulate reservoir states; select only output nodes
        rs = network.simulate(ext_input=x)[:,output_nodes]

        # remove first tau_max points from reservoir states
        rs = rs[tau_max:]

        # split data into training and test sets
        x_train, x_test = iodata.split_dataset(rs)
        y_train, y_test = iodata.split_dataset(y)

        # perform task
        df_ = coding.encoder(reservoir_states=(x_train, x_test),
                            target=(y_train, y_test),
                            readout_modules=readout_modules,
                            readout_nodes=readout_nodes
                            )

        df_['alpha'] = np.round(alpha, 3)

        # reorganize the columns
        if 'module' in df_.columns:
            df.append(df_[['module', 'n_nodes', 'alpha', 'score']])
        else:
            df.append(df_[['alpha', 'score']])

    df = pd.concat(df, ignore_index=True)
    df['score'] = df['score'].astype(float)

    if plot_res:
        sns.set(style="ticks", font_scale=2.0)  
        fig = plt.figure(num=1, figsize=(12,10))
        ax = plt.subplot(111)
        sns.lineplot(data=df, x='alpha', y='score', 
                     hue='module', 
                    #  hue_order=['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN'],
                     palette=sns.color_palette('husl', 7), 
                     markers=True, 
                     ax=ax)
        sns.despine(offset=10, trim=True)
        plt.title('Memory Capacity')
        plt.plot()
        plt.show()

    return df
