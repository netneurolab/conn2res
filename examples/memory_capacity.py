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

# define set of input nodes
ctx  = np.load(os.path.join(DATA_DIR, 'cortical.npy'))
input_nodes  = np.where(ctx == 0)[0] # we use subcortical regions as input nodes
output_nodes = np.where(ctx == 1)[0] # we use cortical regions as output nodes

# We will use resting-state networks as readout modules. These intrinsic networks
# define different sets of output nodes
rsn_mapping = np.load(os.path.join(DATA_DIR, 'rsn_mapping.npy'))

# Evaluate the memory capacity of an echo state network or
# metastable switch memristor network
from conn2res import workflows

MC = workflows.memory_capacity(conn=conn,
                               input_nodes=input_nodes,
                               output_nodes=output_nodes,
                               rsn_mapping=rsn_mapping,
                               resname='MSSNetwork',
                               alphas=np.linspace(0, 4, 21),
                               input_gain=1.0,
                               tau_max=16,
                               plot_res=True,
                               res_kwargs={'mode': 'forward'}
                               )

