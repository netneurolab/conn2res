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
# First, let's import the connectivity matrices we are going to use to define the
# connections of the reservoir.  For this we will be using the human connectome
# parcellated into 1015 brain regions following the Desikan Killiany atlas
# (Desikan, et al., 2006).

from os import truncate
import numpy as np

# load connectivity data
conn = np.load('/Users/laurasuarez/OneDrive - McGill University/Repos/conn2res/data/connectivity.npy')
n_reservoir_nodes = len(conn)
n_subjs = conn.shape[-1]


###############################################################################
# Second, let's get the data to perform the task. We first generate the data and
# then we split it into training and test sets. 'x' corresponds to the input
# signals and 'y' corresponds to the output labels.

from conn2res import iodata
import matplotlib.pyplot as plt

task = 'GoNogo' #'PerceptualDecisionMaking' # 
x, y = iodata.fetch_dataset(task)

n_features = x.shape[1]
print(f'n_features = {n_features}')

n_obs = x.shape[0]
print(f'n_observations = {n_obs}')

# split data into training and test sets
x_train, x_test = iodata.split_dataset(x)
y_train, y_test = iodata.split_dataset(y)


###############################################################################
# Third, let's define the set of input and output nodes, the input connectivity 
# matrix, and the readout modules

ctx  = np.load('/Users/laurasuarez/OneDrive - McGill University/Repos/conn2res/data/cortical.npy')
subctx_nodes = np.where(ctx == 0)[0] # we use subcortical regions as input nodes

input_nodes  = np.random.choice(subctx_nodes, n_features) # we select a randon set of input nodes
output_nodes = np.where(ctx == 1)[0] # we use cortical regions as output nodes

# create input connectivity matrix, which defines the connec-
# tions between the input layer (source nodes where the input 
# signal is coming from) and the input nodes of the reservoir.
w_in = np.zeros((n_features, n_reservoir_nodes))
w_in[np.ix_(np.arange(n_features), input_nodes)] = 0.1 # factor that modulates the activation state of the reservoir

# We will use resting-state networks as readout modules. These intrinsic networks
# define different sets of output nodes
rsn_mapping = np.load('/Users/laurasuarez/OneDrive - McGill University/Repos/conn2res/data/rsn_mapping.npy')
rsn_mapping = rsn_mapping[output_nodes] # [np.where(ctx == 1)] # we select the mapping only for cortical regions


###############################################################################
# Fourth, for every subject, we will simulate the dynamics of the reservoir 
# using the previously generated input signal x (x_train and x_test).
from scipy.linalg import eigh
import pandas as pd
from conn2res import reservoir, coding

for subj in range(n_subjs):

    # select connectivity matrix of subject
    w = conn[:,:,subj]

    # scale conenctivity weights between [0,1]
    w = (w-w.min())/(w.max()-w.min())

    # normalize connectivity matrix by the spectral radius.
    ew, _ = eigh(w)
    w = w/np.max(ew)

    # evaluate network performance across various dynamical regimes
    # we do so by varying the value of alpha 
    alphas = np.linspace(0,2,11) #np.linspace(0,2,41)
    df_encoding = []
    for alpha in alphas[1:]:
        
        print(f'\n----------------------- alpha = {alpha} -----------------------')

        # instantiate an Echo State Network object
        ESN = reservoir.EchoStateNetwork(w_ih=w_in,
                                         w_hh=alpha*w,
                                         activation_function='tanh',
                                        )

        # simulate reservoir states; select only output nodes.
        rs_train = ESN.simulate(ext_input=x_train)[:,output_nodes]
        rs_test  = ESN.simulate(ext_input=x_test)[:,output_nodes] 

        # perform task
        df = coding.encoder(task=task,
                            reservoir_states=(rs_train, rs_test),
                            target=(y_train, y_test),
                            readout_modules=rsn_mapping,
                            )

        df['alpha'] = np.round(alpha, 3)

        # reorganize the columns
        if 'module' in df.columns:
            df_encoding.append(df[['module', 'n_nodes', 'alpha', 'score']])
        else:
            df_encoding.append(df[['alpha', 'score']])
        
    df_encoding = pd.concat(df_encoding, ignore_index=True)
    df_encoding.to_csv(f'/Users/laurasuarez/Desktop/{task}_{subj}.csv')
    

    #############################################################################
    # # Now we plot the results
    # import seaborn as sns

    # df_encoding['score'] = df_encoding['score'].astype(float)

    # sns.lineplot(data=df_encoding, x='alpha', y='score')#, hue='module')
    # plt.title(task)
    # plt.plot()
    # plt.show()