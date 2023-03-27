# -*- coding: utf-8 -*-
"""
Connectome-informed reservoir - Echo-State Network
=======================================================================
This example demonstrates how to use the conn2res toolbox to implement
perform multiple tasks across dynamical regimes, and using different
types local dynamics
"""
import warnings
import numpy as np
import pandas as pd
from conn2res import iodata, reservoir, coding, plotting
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# #####################################################################
N_RUNS = 1
TASKS = [
    'ContextDecisionMaking',
    'SingleContextDecisionMaking',
    'PerceptualDecisionMaking',
    'AntiReach',
    'ReachingDelayResponse'
]
TASK = TASKS[0]
METRICS = {
    'ContextDecisionMaking': ['score', 'balanced_accuracy_score', 'f1_score'],
    'SingleContextDecisionMaking': ['score', 'balanced_accuracy_score', 'f1_score'],
    'PerceptualDecisionMaking': ['score', 'balanced_accuracy_score', 'f1_score'],
    'AntiReach': ['score', 'balanced_accuracy_score', 'f1_score'],
    'ReachingDelayResponse': ['score', 'corrcoef'],
}

# global dynamics: set range of alpha values
ALPHAS = np.linspace(0, 2, 11)[1:]

# local dynamics: select different activation functions
ACT_FCNS = ['tanh']  #, 'sigmoid']

# #####################################################################
# First let's import the connectivity matrix we are going to use to
# define the connections of the reservoir.  For this we will be using
# the human connectome parcellated into 1015 brain regions following
# the Desikan  Killiany atlas (Desikan, et al., 2006).

# load connectivity data of one subject
conn = reservoir.Conn(subj_id=0)

# scale conenctivity weights between [0,1] and normalize by spectral
# radius
conn.scale_and_normalize()

# #####################################################################
# Second we will define the set of input and output nodes. To do so, we
# will use functional intrinsic networks (Yeo ,et al., 2011).

# input nodes: a random set of brain regions in the visual system
n_features = iodata.get_n_features(TASK)
input_nodes = conn.get_nodes(
    'random', nodes_from=conn.get_nodes('VIS'), n_nodes=n_features)

# output nodes: all brain regions in the somatomotor system
output_nodes = conn.get_nodes('SM')

# create input connectivity matrix, which defines the connections
# between the input layer (source nodes where the input signal is
# coming from) and the input nodes of the reservoir.
w_in = np.zeros((n_features, conn.n_nodes))
w_in[:, input_nodes] = np.eye(n_features)

# #####################################################################
# Third we will simulate the dynamics of the reservoir. We will
# evaluate the effect of local network dynamics by using different
# activation functions. We will also evaluate network performance
# across dynamical regimes by parametrically tuning alpha, which
# corresponds to the spectral radius of the connectivity matrix
# (alpha parameter).
df_subj = []
for activation in ACT_FCNS:

    print(f'\n------ activation function = {activation} ------')

    df_runs = []
    for run in range(N_RUNS):

        print(f'\n\t\t--- run = {run} ---')

        # fetch data to perform task
        x, y = iodata.fetch_dataset(TASK, n_trials=1000)

        # visualize task dataset
        if run == 0:
            plotting.plot_iodata(
                x, y, title=TASK, savefig=True, fname=f'io_{TASK}', show=False)

        # get sample weights
        sample_weight = iodata.get_sample_weight(x, y)

        # split trials into training and test sets
        x_train, x_test, y_train, y_test = iodata.split_dataset(x, y, axis=0)
        sample_weight_train, sample_weight_test = iodata.split_dataset(
            sample_weight, axis=1)

        df_alpha = []
        for alpha in ALPHAS:

            print(f'\n\t\t\t----- alpha = {alpha} -----')

            # instantiate an Echo State Network object
            ESN = reservoir.EchoStateNetwork(
                w_ih=w_in,
                w_hh=alpha * conn.w,
                activation_function=activation,
                input_gain=1.0,
                input_nodes=input_nodes,
                output_nodes=output_nodes
            )

            # simulate reservoir states
            rs_train = ESN.simulate(ext_input=x_train)
            rs_test = ESN.simulate(ext_input=x_test)

            # perform task
            df_res, model = coding.encoder(
                reservoir_states=(rs_train, rs_test),
                target=(y_train, y_test),
                return_model=True,
                metric=METRICS[TASK],
                model_kws={'alpha': 0.0, 'fit_intercept': False},
                sample_weight=(sample_weight_train, sample_weight_test),
            )

            # assign column with alpha value and append df_res
            # to df_alpha
            df_res['alpha'] = np.round(alpha, 3)
            df_alpha.append(df_res)

            # plot diagnostic curves
            if run == 0 and alpha == 1.0:
                plotting.plot_diagnostics(
                    x=x_train, y=y_train, reservoir_states=rs_train,
                    trained_model=model, title=TASK, savefig=True,
                    fname=f'diag_train_{TASK}', show=False
                )
                plotting.plot_diagnostics(
                    x=x_test, y=y_test, reservoir_states=rs_test,
                    trained_model=model, title=TASK, savefig=True,
                    fname=f'diag_test_{TASK}', show=False
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
                    + METRICS[TASK]]
        )
    else:
        df_subj.append(df_runs[['activation', 'run', 'alpha'] + METRICS[TASK]])
# concatenate results across activation functions
df_subj = pd.concat(df_subj, ignore_index=True)
df_subj.to_csv(
    '/Users/laurasuarez/Library/CloudStorage/OneDrive-McGillUniversity/Repos/conn2res/figs/results.csv',
    index=False)

############################################################################
# Now we plot the performance curve
df_subj = pd.read_csv(
    '/Users/laurasuarez/Library/CloudStorage/OneDrive-McGillUniversity/Repos/conn2res/figs/results.csv',
    index_col=False)

for metric in METRICS[TASK]:
    plotting.plot_performance(
        df_subj, x='alpha', y=metric, hue='activation', title=TASK,
        savefig=True, fname=f'perf_{metric}', show=False)
