# -*- coding: utf-8 -*-
"""
Connectome-informed reservoir - Memristive Network
=================================================
This example demonstrates how to use the #TODO toolbox to
perform a memory task using a human connectomed-informed
Memristive network
"""

###############################################################################
# First let's import the connectivity matrix we are going to use to define the
# connections of the reservoir.  For this we will be using the human connectome
# parcellated into 1015 brain regions following the Desikan  Killiany atlas
# (Desikan, et al., 2006).

import numpy as np

# load connectivity data
conn = np.load('C:/Users/User/OneDrive - McGill University/Repos/conn2res/data/connectivity.npy')

# select one subject
subj_id = 10
conn = conn[:,:,subj_id]
n_reservoir_nodes = len(conn)

# binarize connectivity matrix
conn = conn.astype(bool).astype(int)


###############################################################################
# Second let's get the data to perform the task. We first generate the data and
# then we split it into training and test sets. 'x' corresponds to the input
# signals and 'y' corresponds to the output labels.

import pandas as pd
from conn2res import iodata

task_name = 'mem_cap'
task_args = {'task':task_name,
             'n_samples':100,
             'gain':3,
             'n_patterns':10,
             'pttn_len':20,
             'frac_train':0.7
            }

pttn_lens = task_args['pttn_len']*np.ones(int(0.5*task_args['n_patterns']*task_args['n_samples']), dtype=int)

x, y = iodata.generate_io(task=task_name, n_samples=1000)

###############################################################################
# Third we will simulate the dynamics of the reservoir using the previously
# generated input signal x (x_train and x_test).

# define set of input nodes
ctx  = np.load('C:/Users/User/OneDrive - McGill University/Repos/conn2res/data/cortical.npy')
input_nodes  = np.where(ctx == 0)[0] # we use subcortical regions as input nodes
output_nodes = np.where(ctx == 1)[0] # we use cortical regions as output nodes

# split data into training and test sets
x_train, x_test = x
y_train, y_test = y

n_features = x_train.shape[1]
n_labels   = y_train.shape[1]
n_grounded_nodes = 1

# create input connectivity matrix, which defines the connec-
# tions between the input layer (source nodes where the input signal is
# coming from) and the input nodes of the reservoir.


#%%----------------------------------------------------------
# create memristive network object
from conn2res import reservoir

MMN = reservoir.MSSNetwork(w=conn,
                           i_nodes=np.setdiff1d(np.where(ctx == 1)[0], [37]),
                           e_nodes=np.where(ctx == 0)[0],
                           gr_nodes=[37]
                           )


#%%
w = MMN._W
G = MMN._G
Ga = MMN._Ga
Gb = MMN._Gb
Nb = MMN._Nb
NMSS = MMN.NMSS

#%%
v_external = 10*np.ones((100,15)) #*np.random.rand(100,15)
test = np.asarray(MMN.simulate(v_external))


#%%
import matplotlib.pyplot as plt

plt.plot(test)
plt.show()

#%%
# # evaluate network performance across various dynamical regimes
# from conn2res import reservoir, coding
#
# alphas = np.linspace(0,2,41)
# df_encoding = []
# for alpha in alphas:
#
#     # instantiate an Echo State Network object
#     ESN = reservoir.EchoStateNetwork(w_ih=w_in,
#                                      w_hh=alpha*conn.copy(),
#                                      nonlinearity='tanh',
#                                      )
#
#     # simulate reservoir states; select only output nodes.
#     x_train = ESN.simulate(ext_input=x_train)[:,output_nodes]
#     x_test  = ESN.simulate(ext_input=x_test)[:,output_nodes]
#
#     # perform task
#     df = coding.encoder(task=task_name,
#                         reservoir_states=(x_train, x_test),
#                         target=(y_train, y_test),
#                         readout_modules=rsn_mapping,
#                         pttn_lens
#                         )
#     df['alpha'] = np.round(alpha, 3)
#
#     df_encoding.append(df)
#
# df_encoding = pd.concat(df_encoding)
# df_encoding.to_csv('C:/Users/User/Desktop/test_df_encoding.csv')
#

###############################################################################
#
# conn = np.load('/Users/laurasuarez/OneDrive - McGill University/Repos/conn2res/data/connectivity.npy')
# ctx  = np.load('/Users/laurasuarez/OneDrive - McGill University/Repos/conn2res/data/cortical.npy')
# rsn_mapping = np.load('/Users/laurasuarez/OneDrive - McGill University/Repos/conn2res/data/rsn_mapping.npy')
