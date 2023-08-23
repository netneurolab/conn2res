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
OUTPUT_DIR = os.path.join(PROJ_DIR, 'results', 'results_new_generated_iodata', 'results_rsn_gain_15x0.0001')
# OUTPUT_DIR = os.path.join(PROJ_DIR, 'results', 'results_old_iodata_sig-and-relia', 'results_rsn_gain_15')

rsn_labels = ['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN']


def concat_results():

    scores = []
    for sample_id in range(1000):
        df = pd.read_csv(os.path.join(OUTPUT_DIR, f'res_null_{sample_id}.csv')).reset_index(drop=True)
        df['sample_id'] = sample_id

        try:
            scores.append(df[['sample_id', 'alpha', 'module', 'n_nodes', 'corrcoef']])
        except:
            scores.append(df[['sample_id', 'alpha', 'corrcoef']])

    scores = pd.concat(scores).reset_index(drop=True)

    scores.to_csv(
        os.path.join(OUTPUT_DIR, 'res_null.csv'),
        index=False
        )

concat_results()

df_emp = pd.read_csv(os.path.join(OUTPUT_DIR, 'res_empirical.csv')).reset_index(drop=True)
df_emp_avg = df_emp[['alpha', 'corrcoef']]
df_emp_avg = df_emp_avg.groupby('alpha').mean().reset_index()

df_null  = pd.read_csv(os.path.join(OUTPUT_DIR, 'res_null.csv')).reset_index(drop=True)
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
