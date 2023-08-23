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

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = '/Users/laurasuarez/Library/CloudStorage/OneDrive-McGillUniversity/new_results/T2-1000trials_0.0001gain/human'  #os.path.join(PROJ_DIR, 'results', 'T2-1000trials_0.0001gain', 'human')

rsn_labels = ['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN']


def concat_results(exp):

    scores = []
    for sample_id in range(500):
        print(sample_id)
        df = pd.read_csv(os.path.join(OUTPUT_DIR, f'{exp}_null_{sample_id}_scores.csv')).reset_index(drop=True)
        df['sample_id'] = sample_id

        try:
            scores.append(df[['sample_id', 'alpha', 'module', 'n_nodes', 'corrcoef']])
        except KeyError:
            scores.append(df[['sample_id', 'alpha', 'corrcoef']])

    scores = pd.concat(scores).reset_index(drop=True)

    scores.to_csv(
        os.path.join(OUTPUT_DIR, f'{exp}_null_scores.csv'),
        index=False
        )



# ---------- HUMAN ----------
for exp in range(6):
    print(f'\n--------{exp}----------')
    concat_results(exp)

for exp in range(6):
    df_emp = pd.read_csv(os.path.join(OUTPUT_DIR, f'{exp}_empirical_scores.csv')).reset_index(drop=True)
    df_emp_avg = df_emp[['alpha', 'corrcoef']]
    df_emp_avg = df_emp_avg.groupby('alpha').mean().reset_index()

    df_null  = pd.read_csv(os.path.join(OUTPUT_DIR, f'{exp}_null_scores.csv')).reset_index(drop=True)
    df_null_avg = df_null[['sample_id', 'alpha', 'corrcoef']]
    df_null_avg = df_null_avg.groupby(['sample_id', 'alpha']).mean().reset_index()

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
