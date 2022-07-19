# -*- coding: utf-8 -*-
"""
Connectome-informed reservoir - Echo-State Network
=================================================
This example demonstrates how to use the conn2res toolbox 
to perform a task using a human connectomed-informed
Echo-State network (Jaeger, 2000).
"""

from conn2res import reservoir, coding, plotting
import pandas as pd
from conn2res import iodata
from scipy.linalg import eigh
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

###############################################################################
# First let's import the connectivity matrix we are going to use to define the
# connections of the reservoir.  For this we will be using the human connectome
# parcellated into 1015 brain regions following the Desikan  Killiany atlas
# (Desikan, et al., 2006).

# load connectivity data
conn = iodata.load_file('connectivity.npy')

# select one subject
subj_id = 0
conn = conn[:, :, subj_id]
n_reservoir_nodes = len(conn)

# scale conenctivity weights between [0,1]
conn = (conn - conn.min())/(conn.max() - conn.min())

# normalize connectivity matrix by the spectral radius.

ew, _ = eigh(conn)
conn = conn / np.max(ew)

###############################################################################
# Second let's get the data to perform the task. We first generate the data and
# then we split it into training and test sets. 'x' corresponds to the input
# signals and 'y' corresponds to the output labels.

# get trial-based dataset for task
task = 'PerceptualDecisionMaking'
n_trials = 1000
x, y = iodata.fetch_dataset(task, n_trials=n_trials)

# visualize task data
iodata.visualize_data(task, x, y, plot=True)

# split trials into training and test sets
x_train, x_test = iodata.split_dataset(x)
y_train, y_test = iodata.split_dataset(y)

###############################################################################
# Third we will simulate the dynamics of the reservoir using the previously
# generated input signal x (x_train and x_test).

# define set of input nodes
ctx = iodata.load_file('cortical.npy')
# we use subcortical regions as input nodes
subctx_nodes = np.where(ctx == 0)[0]

n_features = x_train.shape[1]
# we select a randon set of input nodes
input_nodes = np.random.choice(subctx_nodes, n_features)
output_nodes = np.where(ctx == 1)[0]  # we use cortical regions as output nodes

# create input connectivity matrix, which defines the connec-
# tions between the input layer (source nodes where the input signal is
# coming from) and the input nodes of the reservoir.
w_in = np.zeros((n_features, n_reservoir_nodes))
# factor that modulates the activation state of the reservoir
w_in[np.ix_(np.arange(n_features), input_nodes)] = 10.0 * np.eye(n_features)

# We will use resting-state networks as readout modules. These intrinsic networks
# define different sets of output nodes
rsn_mapping = iodata.load_file('rsn_mapping.npy')
# we select the mapping only for output nodes
rsn_mapping = rsn_mapping[output_nodes]

# evaluate network performance across various dynamical regimes
# we do so by varying the value of alpha

alphas = np.linspace(0, 2, 11)[1:]
df_subj = []
for alpha in alphas:

    print(f'\n----------------------- alpha = {alpha} -----------------------')

    # instantiate an Echo State Network object
    ESN = reservoir.EchoStateNetwork(w_ih=w_in,
                                     w_hh=alpha * conn.copy(),
                                     activation_function='tanh',
                                     )

    # simulate reservoir states; select only output nodes.
    rs_train = ESN.simulate(ext_input=x_train)[:, output_nodes]
    rs_test = ESN.simulate(ext_input=x_test)[:, output_nodes]

    # perform task
    df = coding.encoder(reservoir_states=(rs_train, rs_test),
                        target=(y_train, y_test),
                        readout_modules=rsn_mapping,
                        )

    df['alpha'] = np.round(alpha, 3)

    # reorganize the columns
    if 'module' in df.columns:
        df_subj.append(df[['module', 'n_nodes', 'alpha', 'score']])
    else:
        df_subj.append(df[['alpha', 'score']])

df_subj = pd.concat(df_subj, ignore_index=True)
df_subj['score'] = df_subj['score'].astype(float)

############################################################################
# Now we plot the performance curve

plotting.plot_performance_curve(df_subj, task)
