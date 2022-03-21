# -*- coding: utf-8 -*-
"""
Connectome-informed reservoir - Echo-State Network
=================================================
This example demonstrates how to use the conn2res toolbox to
perform a memory task using a human connectomed-informed
Echo-State network while playing with the dynamics of the reservoir
(Jaeger, 2000).
"""

###############################################################################
# First let's import the connectivity matrix we are going to use to define the
# connections of the reservoir.  For this we will be using the human connectome
# parcellated into 1015 brain regions following the Desikan  Killiany atlas
# (Desikan, et al., 2006).
import os 
import numpy as np

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_DIR, 'examples', 'data')

# load connectivity data
conn = np.load(os.path.join(DATA_DIR, 'connectivity.npy'))

# select one subject
subj_id = 0
conn = conn[:,:,subj_id]
n_reservoir_nodes = len(conn)

# scale conenctivity weights between [0,1]
conn = (conn-conn.min())/(conn.max()-conn.min())

# normalize connectivity matrix by the spectral radius.
from scipy.linalg import eigh

ew, _ = eigh(conn)
conn  = conn/np.max(ew)

###############################################################################
# Second let's get the data to perform the task. We first generate the data and
# then we split it into training and test sets. 'x' corresponds to the input
# signals and 'y' corresponds to the output labels.
from conn2res import iodata
from conn2res import reservoir, coding

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

tasks = iodata.get_available_tasks()
for task in tasks[:]:
    
    x, y = iodata.fetch_dataset(task)
    # y = iodata.encode_labels(y)

    n_samples  = x.shape[0]
    print(f'n_observations = {n_samples}')

    n_features = x.shape[1]
    print(f'n_features = {n_features}')

    try:    n_labels = y.shape[1] 
    except: n_labels = 1
    print(f'n_labels   = {n_labels}')

    fig, axs = plt.subplots(2,1, figsize=(10,10), sharex=True)
    axs = axs.ravel()
    axs[0].plot(x)
    axs[0].set_ylabel('Inputs')
    
    axs[1].plot(y)
    axs[1].set_ylabel('Outputs')

    plt.suptitle(task)
    plt.show()
    plt.close()

    # split data into training and test sets
    x_train, x_test = iodata.split_dataset(x)
    y_train, y_test = iodata.split_dataset(y)

    ###############################################################################
    # Third we will simulate the dynamics of the reservoir using the previously
    # generated input signal x (x_train and x_test).

    # define set of input nodes
    ctx  = np.load(os.path.join(DATA_DIR, 'cortical.npy'))
    subctx_nodes = np.where(ctx == 0)[0] # we use subcortical regions as input nodes

    input_nodes  = np.random.choice(subctx_nodes, n_features) # we select a randon set of input nodes
    output_nodes = np.where(ctx == 1)[0] # we use cortical regions as output nodes

    # create input connectivity matrix, which defines the connec-
    # tions between the input layer (source nodes where the input signal is
    # coming from) and the input nodes of the reservoir.
    w_in = np.zeros((n_features, n_reservoir_nodes))
    w_in[np.ix_(np.arange(n_features), input_nodes)] = 0.1 # factor that modulates the activation state of the reservoir

    # We will use resting-state networks as readout modules. These intrinsic networks
    # define different sets of output nodes
    rsn_mapping = np.load(os.path.join(DATA_DIR, 'rsn_mapping.npy'))
    rsn_mapping = rsn_mapping[output_nodes] # we select the mapping only for output nodes 

    # evaluate network performance across various dynamical regimes
    # we do so by varying the value of alpha 
    alphas = np.linspace(0,2,11) #np.linspace(0,2,41)
    df_subj = []
    for alpha in alphas[1:]:
        
        print(f'\n----------------------- alpha = {alpha} -----------------------')

        # instantiate an Echo State Network object
        ESN = reservoir.EchoStateNetwork(w=alpha * conn.copy(),
                                         w_in=w_in,
                                         activation_function='tanh',
                                        )

        # simulate reservoir states; select only output nodes.
        rs_train = ESN.simulate(ext_input=x_train)[:,output_nodes]
        rs_test  = ESN.simulate(ext_input=x_test)[:,output_nodes] 

        # perform task
        df = coding.encoder(reservoir_states=(rs_train, rs_test),
                            target=(y_train, y_test),
                            readout_modules=rsn_mapping,
                            # pttn_lens=()
                            )

        df['alpha'] = np.round(alpha, 3)

        # reorganize the columns
        if 'module' in df.columns:
            df_subj.append(df[['module', 'n_nodes', 'alpha', 'score']])
        else:
            df_subj.append(df[['alpha', 'score']])
        
    df_subj = pd.concat(df_subj, ignore_index=True)
    df_subj['score'] = df_subj['score'].astype(float)
    df_subj['alpha'] = df_subj['alpha'].astype(float)

    print(df_subj.head(5))
    
    #############################################################################
    # Now we plot the performance curve
    sns.set(style="ticks", font_scale=2.0)  
    fig = plt.figure(num=1, figsize=(12,10))
    ax = plt.subplot(111)
    sns.lineplot(data=df_subj, x='alpha', y='score', 
                 hue='module', 
                 # hue_order=['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN'],
                 palette=sns.color_palette('husl', 7), 
                 markers=True, 
                 ax=ax)
    sns.despine(offset=10, trim=True)
    plt.title(task)
    plt.plot()
    plt.show()