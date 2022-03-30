# -*- coding: utf-8 -*-
"""
Connectome-informed reservoir - Memristive Network
=================================================
This example demonstrates how to use the conn2res toolbox 
to perform a task using a human connectomed-informed
Memristive network
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

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
subj_id = 10
conn = conn[:,:,subj_id]
n_reservoir_nodes = len(conn)

# binarize connectivity matrix
conn = conn.astype(bool).astype(int)

# normalize connectivity matrix by the spectral radius.
from scipy.linalg import eigh

ew, _ = eigh(conn)
conn  = conn/np.max(ew)

###############################################################################
# Second let's get the data to perform the task. We first generate the data and
# then we split it into training and test sets. 'x' corresponds to the input
# signals and 'y' corresponds to the output labels.

from conn2res import iodata
import matplotlib.pyplot as plt

task = 'GoNogo' #'PerceptualDecisionMaking' # 
x, y = iodata.fetch_dataset(task)

# # visualizing input/output data 
# print(f'\n----{task}-----')
# x_labels = [f'i{n+1}' for n in range(x.shape[1])]
# plt.plot(x[:], label=x_labels)
# plt.plot(y[:], label='label')
# plt.legend()
# plt.suptitle(task)
# plt.show()
# plt.close()

n_features = x.shape[1]
print(f'n_features = {n_features}')

# n_labels   = y.shape[1]
# print(f'n_labels = {n_labels}')

n_obs = x.shape[0]
print(f'n_observations = {n_obs}')

# split data into training and test sets
x_train, x_test = iodata.split_dataset(x)
y_train, y_test = iodata.split_dataset(y)

###############################################################################
# Third we will simulate the dynamics of the reservoir using the previously
# generated input signal x (x_train and x_test).

# define sets of internal, external and ground nodes
ctx  = np.load(os.path.join(DATA_DIR, 'cortical.npy'))

nodes = np.arange(n_reservoir_nodes)
gr_nodes  = np.random.choice(np.where(ctx == 1)[0], 1) # we select a single random ground node from cortical regions
ext_nodes = np.random.choice(np.where(ctx == 0)[0], n_features) # we select a random set of input nodes from subcortical regions
int_nodes = np.setdiff1d(nodes, np.union1d(gr_nodes,ext_nodes)) # we use the reamining cortical regions as output nodes

# However, because not all subcortical nodes are used as input nodes in this case, we create 
# a new set of output nodes that include (only) all cortical regions but those corresponding 
# to grounded nodes
output_nodes = np.setdiff1d(np.where(ctx == 1)[0], gr_nodes) # nodes actually used to perform the task

# We will use resting-state networks as readout modules. These intrinsic networks
# define different sets of output nodes
rsn_mapping = np.load(os.path.join(DATA_DIR, 'rsn_mapping.npy'))
rsn_mapping = rsn_mapping[output_nodes] # we select the mapping only for output nodes 

# evaluate network performance across various dynamical regimes
# we do so by varying the value of alpha 
import pandas as pd
from conn2res import reservoir, coding

alphas = np.linspace(0,2,11) 
df_subj = []
for alpha in alphas[1:]:
    
    print(f'\n----------------------- alpha = {alpha} -----------------------')

    # instantiate an Memristive Network object
    MMN = reservoir.MSSNetwork(w=alpha*conn.copy(),
                               i_nodes=int_nodes,
                               e_nodes=ext_nodes,
                               gr_nodes=gr_nodes
                               )

    # simulate reservoir states; select only readout nodes.
    rs_train = MMN.simulate(Vext=x_train[:], mode='forward')[:,output_nodes]
    rs_test  = MMN.simulate(Vext=x_test[:],  mode='backward')[:,output_nodes] 
    
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
   
#############################################################################
# Now we plot the performance curve
import seaborn as sns

sns.set(style="ticks", font_scale=2.0)  
fig = plt.figure(num=1, figsize=(12,10))
ax = plt.subplot(111)
sns.lineplot(data=df_subj, x='alpha', y='score', 
             hue='module', 
             hue_order=['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN'],
             palette=sns.color_palette('husl', 7), 
             markers=True, 
             ax=ax)
sns.despine(offset=10, trim=True)
plt.title(task)
plt.plot()
plt.show()
