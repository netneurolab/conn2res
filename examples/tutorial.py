# -*- coding: utf-8 -*-
"""
Connectome-informed reservoir - Echo-State Network
=======================================================================
This example demonstrates how to use the conn2res toolbox to implement
perform multiple tasks across dynamical regimes, and using different
types local dynamics
"""
import warnings
import os
import numpy as np
import pandas as pd
from sklearn.base import is_classifier, is_regressor
from conn2res.tasks import NeuroGymTask
from conn2res.connectivity import Conn
from conn2res.reservoir import EchoStateNetwork
from conn2res.readout import Readout
from conn2res import readout, plotting

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# #####################################################################
# First, let's initialize some constant variables
# #####################################################################

# project and figure directory 
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJ_DIR, 'figs')
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# number of runs for each task
N_RUNS = 1

# name of the tasks to be performed
TASKS = [
    'ContextDecisionMaking',
    'SingleContextDecisionMaking',
    'PerceptualDecisionMaking',
    'AntiReach',
    'ReachingDelayResponse'
]

# define metrics to evaluate readout's model performance
REG_METRICS = [
    'score',
    # 'r2_score',
    # 'mean_squared_error',
    'root_mean_squared_error',
    # 'mean_absolute_error',
    'corrcoef'
    ]

CLASS_METRICS = [
    'score',
    # 'accuracy_score',
    'balanced_accuracy_score',
    'f1_score',
    # 'precision_score',
    # 'recall_score'
]

# define alpha values to vary global reservoir dynamics
ALPHAS = np.linspace(0, 2, 11)[1:]

# select different activation functions to vary local dynamics
ACT_FCNS = ['tanh']  #, 'sigmoid']

# #####################################################################
# Second, let's create an instance of a NeuroGym task. To do so we need
# the name of task.
task = NeuroGymTask(name=TASKS[2])

# #####################################################################
# Third, let's import the connectivity matrix we are going to use to
# define the connections of the reservoir.  For this we will be using
# the human connectome parcellated into 1015 brain regions following
# the Desikan  Killiany atlas (Desikan, et al., 2006).

# load connectivity data of one subject
conn = Conn(subj_id=0)

# scale conenctivity weights between [0,1] and normalize by spectral its
# radius
conn.scale_and_normalize()

# #####################################################################
# Third we will simulate the dynamics of the reservoir. We will
# evaluate the effect of local network dynamics by using different
# activation functions. We will also evaluate network performance
# across dynamical regimes by parametrically tuning alpha, which
# corresponds to the spectral radius of the connectivity matrix
# (alpha parameter).
# #####################################################################
df_subj = []
for activation in ACT_FCNS:

    print(f'\n------ activation function = {activation} ------')

    df_runs = []
    for run in range(N_RUNS):

        print(f'\n\t\t--- run = {run} ---')

        # fetch data to perform task
        x, y = task.fetch_data(n_trials=1000)

        # visualize task dataset
        if run == 0:
            plotting.plot_iodata(
                x, y, title=task.name, savefig=True, fname=os.path.join(OUTPUT_DIR, f'io_{task.name}'),
                show=False
            )

        # split data into training and test sets
        x_train, x_test, y_train, y_test = readout.train_test_split(x, y)

        # We will define the set of input and output nodes. To do so, we
        # will use functional intrinsic networks (Yeo ,et al., 2011).
        # input nodes: a random set of brain regions in the visual system
        input_nodes = conn.get_nodes(
            'random', nodes_from=conn.get_nodes('VIS'),
            n_nodes=task.n_features
        )

        # output nodes: all brain regions in the somatomotor system
        output_nodes = conn.get_nodes('SM')

        # create input connectivity matrix to define connections between
        # the input layer (source nodes where the input signal is coming
        # from) and the input nodes of the reservoir.
        w_in = np.zeros((task.n_features, conn.n_nodes))
        w_in[:, input_nodes] = np.eye(task.n_features)

        # instantiate an Echo State Network object
        esn = EchoStateNetwork(w=conn.w, activation_function=activation)

        # instantiate a Readout object
        readout_module = Readout(estimator=readout.select_model(y))

        # defined performance metrics based on Readout's type of model
        if is_classifier(readout_module.model):
            metrics = CLASS_METRICS
        elif is_regressor(readout_module.model):
            metrics = REG_METRICS

        # iterate global dynamics using different alpha values
        df_alpha = []
        for alpha in ALPHAS:

            print(f'\n\t\t\t----- alpha = {alpha} -----')

            # scale connectivity matrix by alpha
            esn.w = alpha * conn.w

            # simulate reservoir states
            rs_train = esn.simulate(
                ext_input=x_train, w_in=w_in, input_gain=1,
                output_nodes=output_nodes
            )

            rs_test = esn.simulate(
                ext_input=x_test, w_in=w_in, input_gain=1,
                output_nodes=output_nodes
            )

            # visualize reservoir states
            if run == 0 and alpha == 1.0:
                plotting.plot_reservoir_states(
                    x=x_train, reservoir_states=rs_train,
                    title=task.name,
                    savefig=True, fname=os.path.join(OUTPUT_DIR, f'res_states_train_{task.name}'), show=False
                )
                plotting.plot_reservoir_states(
                    x=x_test, reservoir_states=rs_test,
                    title=task.name,
                    savefig=True, fname=os.path.join(OUTPUT_DIR, f'res_states_test_{task.name}'), show=False
                )

            # perform task
            df_res = readout_module.run_task(
                X=(rs_train, rs_test), y=(y_train, y_test),
                sample_weight='both', metric=metrics,
                readout_modules=None, readout_nodes=None,
            )

            # assign column with alpha value and append df_res
            # to df_alpha
            df_res['alpha'] = np.round(alpha, 3)
            df_alpha.append(df_res)

            # visualize diagnostic curves
            if run == 0 and alpha == 1.0 and is_classifier(readout_module.model):
                plotting.plot_diagnostics(
                    x=x_train, y=y_train, reservoir_states=rs_train,
                    trained_model=readout_module.model, title=task.name,
                    savefig=True, fname=os.path.join(OUTPUT_DIR, f'diag_train_{task.name}'), show=False
                )
                plotting.plot_diagnostics(
                    x=x_test, y=y_test, reservoir_states=rs_test,
                    trained_model=readout_module.model, title=task.name,
                    savefig=True, fname=os.path.join(OUTPUT_DIR, f'diag_test_{task.name}'), show=False
                )

        # concatenate results across alpha values and append
        # df_alpha to df_runs
        df_alpha = pd.concat(df_alpha, ignore_index=True)
        df_alpha['run'] = run
        print(df_alpha.head(len(ALPHAS)))
        df_runs.append(df_alpha)
    # concatenate results across runs and append
    # df_runs to df_subj
    df_runs = pd.concat(df_runs, ignore_index=True)
    df_runs['activation'] = activation
    if 'module' in df_runs.columns:
        df_subj.append(
            df_runs[['module', 'n_nodes', 'activation', 'run', 'alpha']
                    + metrics]
        )
    else:
        df_subj.append(df_runs[['activation', 'run', 'alpha'] + metrics])
# concatenate results across activation functions
df_subj = pd.concat(df_subj, ignore_index=True)

# save results
df_subj.to_csv(os.path.join(PROJ_DIR, 'figs', f'results{task.name}.csv'), index=False)

######################################################################
# visualize performance curve
df_subj = pd.read_csv(os.path.join(PROJ_DIR, 'figs', f'results{task.name}.csv'), index_col=False)

for metric in metrics:
    plotting.plot_performance(
        df_subj, x='alpha', y=metric, hue='activation',
        title=task.name, savefig=True, fname=os.path.join(OUTPUT_DIR, f'perf_{metric}'),
        show=False
    )
