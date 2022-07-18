# -*- coding: utf-8 -*-
"""
Connectome-informed reservoir - Echo-State Network
=================================================
This example demonstrates how to use the conn2res toolbox to
perform a memory task using a human connectomed-informed
Echo-State network while playing with the dynamics of the reservoir
(Jaeger, 2000).
"""

from conn2res import reservoir, workflows, iodata
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

###############################################################################
# First let's import the connectivity matrix we are going to use to define the
# connections of the reservoir.  For this we will be using the human connectome
# parcellated into 1015 brain regions following the Desikan  Killiany atlas
# (Desikan, et al., 2006).

# load connectivity data of one subject
conn = reservoir.Conn(subj_id=55)

# set of nodes for Echo State Network
input_nodes = conn.get_nodes('subctx')
output_nodes = conn.get_nodes('ctx')

# set of nodes for MSSNetwork
# we select a single random ground node from cortical regions
gr_nodes = conn.get_nodes(
    'random', nodes_from=conn.get_nodes('ctx'), n_nodes=1)
# we select external nodes as random set of input nodes from subcortical regions
ext_nodes = conn.get_nodes('subctx')
# we use the reamining cortical regions as output nodes
int_nodes = conn.get_nodes(
    'all', nodes_without=np.union1d(gr_nodes, ext_nodes))

# We will use resting-state networks as readout modules. These intrinsic networks
# define different sets of output nodes
rsn_mapping = iodata.load_file('rsn_mapping.npy')

# Evaluate memory capacity

# echo state network
MC = workflows.memory_capacity(resname='EchoStateNetwork',
                               conn=conn,
                               input_nodes=input_nodes,
                               output_nodes=output_nodes,
                               readout_modules=rsn_mapping[conn.idx_node][output_nodes],
                               alphas=np.linspace(0, 4, 21),
                               input_gain=1.0,
                               tau_max=16,
                               plot_res=True,
                               activation_function='linear',
                               )

# # memristive network
# MC = workflows.memory_capacity(resname='MSSNetwork',
#                                 conn=conn,
#                                 int_nodes=int_nodes,
#                                 ext_nodes=ext_nodes,
#                                 gr_nodes=gr_nodes,
#                                 # readout_modules=rsn_mapping[conn.idx_node][int_nodes],
#                                 alphas=[1.0],
#                                 input_gain=1.0,
#                                 tau_max=16,
#                                 plot_res=True,
#                                 mode='forward',
#                                 )
