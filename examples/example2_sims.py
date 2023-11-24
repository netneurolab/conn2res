# -*- coding: utf-8 -*-
"""
Example 2: Anatomical inferences
=======================================================================
This example demonstrates how the toolbox can be used to make inferences
about regional heterogeneity or specificity for computational capacity.
Specifically, we implement the perceptual decision making task on a
single subject-level, connectome-informed reservoir. Cortical nodes
are stratified according to their affiliation with the canonical
intrinsic networks [1]. Brain regions in the visual network are used as
input nodes. To quantify task performance, each intrinsic networks
is used separately as a readout module.

[1] Yeo, B. T., Krienen, F. M., Sepulcre, J., Sabuncu, M. R., Lashkari,
D., Hollinshead, M., ... & Buckner, R. L. (2011). The organization of
the human cerebral cortex estimated by intrinsic functional
connectivity. Journal of neurophysiology.
"""
import warnings

import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"

import time
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from conn2res import tasks
from conn2res.tasks import NeuroGymTask
from conn2res.connectivity import Conn
from conn2res.reservoir import EchoStateNetwork
from conn2res.readout import Readout
from conn2res import readout, plotting, utils

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# -----------------------------------------------------
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_DIR, 'examples', 'data', 'human')
OUTPUT_DIR = os.path.join(PROJ_DIR, 'examples', 'results', 'example2')
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# -----------------------------------------------------
N_PROCESS = 32
TASK = 'ContextDecisionMaking'
METRIC = ['balanced_accuracy_score']
INPUT_GAIN = 1
ALPHAS = np.linspace(0, 2, 21)[1:]
RSN_MAPPING = np.load(os.path.join(DATA_DIR, 'rsn_mapping.npy'))

def run_workflow(filename=None):

    task = NeuroGymTask(name=TASK)
    x, y = task.fetch_data(n_trials=1000, input_gain=INPUT_GAIN)

    conn = Conn(subj_id=0)
    conn.scale_and_normalize()

    input_nodes = conn.get_nodes(
        'random', nodes_from=conn.get_nodes('VIS'),
        n_nodes=task.n_features,
    )

    output_nodes = conn.get_nodes(
        nodes_from=None,
        node_set='ctx',
    )

    w_in = np.zeros((task.n_features, conn.n_nodes))
    w_in[:, input_nodes] = np.eye(task.n_features)

    esn = EchoStateNetwork(w=conn.w, activation_function='tanh')
    readout_module = Readout(estimator=readout.select_model(y))

    x_train, x_test, y_train, y_test = readout.train_test_split(x, y)

    active_rsn_modules = RSN_MAPPING[conn.idx_node == 1]

    df_alpha = []
    for alpha in ALPHAS:

        print(f'\n\t\t\t----- alpha = {alpha} -----')

        esn.w = alpha * conn.w

        rs_train = esn.simulate(
            ext_input=x_train, w_in=w_in,
            output_nodes=output_nodes
        )

        rs_test = esn.simulate(
            ext_input=x_test, w_in=w_in,
            output_nodes=output_nodes
        )

        df_res = readout_module.run_task(
            X=(rs_train, rs_test), y=(y_train, y_test),
            sample_weight='both', metric=METRIC,
            readout_modules=active_rsn_modules[output_nodes], readout_nodes=None,
        )

        df_res['alpha'] = np.round(alpha, 3)
        df_alpha.append(df_res)

    df_alpha = pd.concat(df_alpha, ignore_index=True)
    df_alpha = df_alpha[['alpha', 'module', 'n_nodes', METRIC[0]]]
    df_alpha.to_csv(
        os.path.join(OUTPUT_DIR, f'{filename}_scores.csv'),
        index=False
    )


def run_experiment():

    # run workflow for different iters of X and Y
    params = []
    for i in range(500):
        params.append(
            {
                'filename': f'subj_0_xy_{i}'
            }
        )

    print('\nINITIATING PROCESSING TIME')
    t0 = time.perf_counter()

    pool = mp.Pool(processes=N_PROCESS)
    res = [pool.apply_async(run_workflow, (), p) for p in params]
    for r in res: r.get()
    pool.close()

    print('\nTOTAL PROCESSING TIME')
    print(time.perf_counter()-t0, "seconds process time")
    print('END')


def main():
    run_experiment()


if __name__ == '__main__':
    main()
