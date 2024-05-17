# -*- coding: utf-8 -*-
"""
Connectome-informed reservoir - Memristive Network
=================================================
This example demonstrates how to use the conn2res toolbox 
to perform a task using a human connectomed-informed
Memristive network
"""
import warnings
import os
import numpy as np
import pandas as pd
from conn2res.tasks import Conn2ResTask
from conn2res.connectivity import Conn
from conn2res.reservoir import MSSNetwork
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
    'MemoryCapacity'
]

# define metrics to evaluate readout's model performance
METRICS = [
    'corrcoef',
]

# define alpha values to vary global reservoir dynamics
ALPHAS = [1.0]  # np.linspace(0, 2, 41)[1:]

for task_name in TASKS:

    print(f'\n---------------TASK: {task_name.upper()}---------------')

    OUTPUT_DIR = os.path.join(PROJ_DIR, 'figs', task_name)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # #####################################################################
    # Second, let's create an instance of a NeuroGym task. To do so we need
    # the name of task.
    # #####################################################################
    task = Conn2ResTask(name=task_name)

    # #####################################################################
    # Third, let's import the connectivity matrix we are going to use to
    # define the connections of the reservoir.  For this we will be using
    # the human connectome parcellated into 1015 brain regions following
    # the Desikan  Killiany atlas (Desikan, et al., 2006).
    # #####################################################################

    # load connectivity data of one subject
    conn = Conn(subj_id=0)

    # scale conenctivity weights between [0,1] and normalize by spectral its
    # radius
    conn.scale_and_normalize()

    # #####################################################################
    # Next, we will simulate the dynamics of the reservoir. We will evaluate
    # the effect of local network dynamics by using different activation
    # functions. We will also evaluate network performance across dynamical
    # regimes by parametrically tuning alpha, which corresponds to the
    # spectral radius of the connectivity matrix (alpha parameter).
    # #####################################################################
    df_runs = []
    for run in range(N_RUNS):
        print(f'\n\t\t--- run = {run} ---')

        # fetch data to perform task
        x, y = task.fetch_data(n_trials=500, input_gain=1)

        # visualize task dataset
        if run == 0:
            plotting.plot_iodata(
                x, y, title=task.name, savefig=True,
                fname=os.path.join(OUTPUT_DIR, f'io_{task.name}'),
                rc_params={'figure.dpi': 300, 'savefig.dpi': 300},
                show=False
            )

        # split data into training and test sets
        x_train, x_test, y_train, y_test = readout.train_test_split(x, y)

        # We will define the set of external, internal and ground nodes. We
        # will also define the set of readout nodes, which will be the ones
        # to be actually used to perform the task.
        gr_nodes = conn.get_nodes(
            'random',
            nodes_from=conn.get_nodes('ctx'),
            n_nodes=1
        )  # we select a single random node from ctx - GROUND

        ext_nodes = conn.get_nodes(
            'random',
            nodes_from=conn.get_nodes('subctx'),
            n_nodes=task.n_features
        )  # we select a random set of nodes from subctx - EXTERNAL/INPUT

        int_nodes = conn.get_nodes(
            'all',
            nodes_without=np.union1d(gr_nodes, ext_nodes),
            n_nodes=task.n_features
        )  # we select the reamining ctx and subctx - INTERNAL

        output_nodes = conn.get_nodes(
            'ctx',
            nodes_without=gr_nodes,
            n_nodes=task.n_features
        )  # we use the reamining ctx regions - READOUT/OUTPUT

        # instantiate an Metastable Switch Memristive network object
        mssn = MSSNetwork(
            w=conn.w,
            int_nodes=int_nodes,
            ext_nodes=ext_nodes,
            gr_nodes=gr_nodes,
            mode='backward'
        )

        # instantiate a Readout object
        readout_module = Readout(estimator=readout.select_model(y))

        # defined performance metrics based on Readout's type of model
        metrics = METRICS

        # iterate global dynamics using different alpha values
        df_alpha = []
        for alpha in ALPHAS:

            print(f'\n\t\t\t----- alpha = {alpha} -----')

            # scale connectivity matrix by alpha
            mssn.w = alpha * conn.w

            # simulate reservoir states
            rs_train = mssn.simulate(
                Vext=x_train
            )

            rs_test = mssn.simulate(
                Vext=x_test,
            )

            # visualize reservoir states
            if run == 0 and alpha == 1.0:
                plotting.plot_reservoir_states(
                    x=x_train, reservoir_states=rs_train,
                    title=task.name,
                    savefig=True,
                    fname=os.path.join(OUTPUT_DIR, f'res_states_train_{task.name}'),
                    rc_params={'figure.dpi': 300, 'savefig.dpi': 300},
                    show=False
                )
                plotting.plot_reservoir_states(
                    x=x_test, reservoir_states=rs_test,
                    title=task.name,
                    savefig=True,
                    fname=os.path.join(OUTPUT_DIR, f'res_states_test_{task.name}'),
                    rc_params={'figure.dpi': 300, 'savefig.dpi': 300},
                    show=False
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

        # concatenate results across alpha values and append
        # df_alpha to df_runs
        df_alpha = pd.concat(df_alpha, ignore_index=True)
        df_alpha['run'] = run
        df_runs.append(df_alpha)

    # concatenate results across runs and append
    # df_runs to df_subj
    df_runs = pd.concat(df_runs, ignore_index=True)
    if 'module' in df_runs.columns:
        df_subj = df_runs[
            ['module', 'n_nodes', 'run', 'alpha'] + metrics
        ]
    else:
        df_subj = df_runs[
            ['run', 'alpha'] + metrics
        ]

    df_subj.to_csv(
        os.path.join(OUTPUT_DIR, f'results_{task.name}.csv'),
        index=False
        )

    ###########################################################################
    # visualize performance curve
    df_subj = pd.read_csv(
                os.path.join(OUTPUT_DIR, f'results_{task.name}.csv'),
                index_col=False
                )

    for metric in metrics:
        plotting.plot_performance(
            df_subj, x='alpha', y=metric,
            title=task.name, savefig=True,
            fname=os.path.join(OUTPUT_DIR, f'perf_{task.name}_{metric}'),
            rc_params={'figure.dpi': 300, 'savefig.dpi': 300},
            show=False
        )
