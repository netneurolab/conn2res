# -*- coding: utf-8 -*-
"""
Title
=======================================================================
This example demonstrates how ...
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from conn2res import plotting

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJ_DIR, 'examples', 'results')
FIG_DIR = os.path.join(PROJ_DIR, 'examples', 'figs')
if not os.path.isdir(FIG_DIR):
    os.makedirs(FIG_DIR)


def concat_results(connectome):

    scores = []
    for sample_id in range(500):
        df = pd.read_csv(
            os.path.join(OUTPUT_DIR, f'{connectome}_null_{sample_id}_scores.csv')
        ).reset_index(drop=True)

        df['sample_id'] = sample_id

        scores.append(
            df[['sample_id', 'alpha', 'module', 'n_nodes', 'corrcoef']]
        )

    scores = pd.concat(scores).reset_index(drop=True)

    scores.to_csv(
        os.path.join(OUTPUT_DIR, f'{connectome}_null_scores.csv'),
        index=False
        )


connectomes = [
    'consensus',
    'consensus_1',
    'consensus_2',
    'consensus_3',
    'consensus_4',
    'consensus_5'
]

# for each connectome, concatenate scores across nulls
for connectome in connectomes:
    print(f'\n--------{connectome}----------')
    concat_results(connectome)

# concatenate results across connectomes
scores = []
for connectome in connectomes:
    df = pd.read_csv(
        os.path.join(OUTPUT_DIR, f'{connectome}_empirical.csv')
    ).reset_index(drop=True)

    df['connectome'] = connectome

    scores.append(
        df[['connectome', 'alpha', 'module', 'n_nodes', 'corrcoef']]
    )

scores = pd.concat(scores).reset_index(drop=True)

scores.to_csv(
    os.path.join(OUTPUT_DIR, 'human_scores.csv'),
    index=False
    )


plotting.plot_performance(
    df=scores, x='alpha', normalize=True, hue='connectome',
    show=True, savefig=True, fname='human_performance_curve',
    ax_params={'ylabel': 'memory capacity'}
)













for connectome in connectomes:
    df_emp = pd.read_csv(
        os.path.join(OUTPUT_DIR, f'{connectome}_empirical_scores.csv')
    ).reset_index(drop=True)

    df_emp_avg = df_emp[['alpha', 'corrcoef']]
    df_emp_avg = df_emp_avg.groupby('alpha').mean().reset_index()

    df_null = pd.read_csv(
        os.path.join(OUTPUT_DIR, f'{connectome}_null_scores.csv')
    ).reset_index(drop=True)

    df_null_avg = df_null[['sample_id', 'alpha', 'corrcoef']]
    df_null_avg = df_null_avg.groupby(
        ['sample_id', 'alpha']
    ).mean().reset_index()

    sns.set(style="ticks", font_scale=1.0)
    fig = plt.figure(figsize=(15,5))
    ax = plt.subplot(111)

    sns.boxplot(
        data=df_null_avg,
        x='alpha',
        y='corrcoef',
        sym='',
        ax=ax
    )

    sns.scatterplot(
        x=np.arange(len(np.unique(df_null_avg['alpha']))),
        y=df_emp_avg['corrcoef'].values,
        ax=ax
    )

    plt.show()
    plt.close()