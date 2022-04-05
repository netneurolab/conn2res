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
n_reservoir_nodes = len(conn)

# select one subject
subj_id = 0
conn = conn[:,:,subj_id]

# define set of nodes
ctx  = np.load(os.path.join(DATA_DIR, 'cortical.npy'))

# set of nodes for Echo State Network
input_nodes  = np.where(ctx == 0)[0] # we use subcortical regions as input nodes
output_nodes = np.where(ctx == 1)[0] # we use cortical regions as output nodes

#set of nodes for MSSNetwork
nodes = np.arange(n_reservoir_nodes) # this is the set of all nodes in the network
gr_nodes  = np.random.choice(np.where(ctx == 1)[0], 1) # we select a single random ground node from cortical regions
ext_nodes = np.where(ctx == 0)[0] # we select a random set of input nodes from subcortical regions
int_nodes = np.setdiff1d(nodes, np.union1d(gr_nodes,ext_nodes)) # we use the reamining cortical regions as output nodes

# We will use resting-state networks as readout modules. These intrinsic networks
# define different sets of output nodes
rsn_mapping = np.load(os.path.join(DATA_DIR, 'rsn_mapping.npy'))

# Evaluate memory capacity
from conn2res import workflows


# echo state network
MC = workflows.memory_capacity(resname='EchoStateNetwork',
                                conn=conn,
                                input_nodes=input_nodes,
                                output_nodes=output_nodes,
                                readout_modules=rsn_mapping[output_nodes],
                                alphas=np.linspace(0,4,21),
                                input_gain=1.0,
                                tau_max=16,
                                plot_res=True,
                                )

# memristive network
MC = workflows.memory_capacity(resname='MSSNetwork',
                                conn=conn,
                                int_nodes=int_nodes,
                                ext_nodes=ext_nodes,
                                gr_nodes=gr_nodes,
                                readout_modules=rsn_mapping[int_nodes],
                                alphas=np.linspace(0,4,21),
                                input_gain=1.0,
                                tau_max=16,
                                plot_res=True,
                                )


