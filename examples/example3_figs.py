# -*- coding: utf-8 -*-
"""
Example 3: Cross-species comparison
=======================================================================
This script includes the analysis code for example 3
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from conn2res import plotting

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJ_DIR, 'examples', 'results', 'example3')
FIG_DIR = os.path.join(PROJ_DIR, 'examples', 'figs')
if not os.path.isdir(FIG_DIR):
    os.makedirs(FIG_DIR)

# list of connectomes (empirical networks)
connectomes = [
    'drosophila',
    'mouse',
    'rat',
    'macaque_modha',
]

# #####################################################################
# Concatenate scores across empirical connectomes
# #####################################################################
emp_scores = []
for connectome in connectomes:
    df = pd.read_csv(
        os.path.join(OUTPUT_DIR, f'{connectome}_empirical_scores.csv')
    ).reset_index(drop=True)

    df['connectome'] = connectome

    emp_scores.append(
        df[['connectome', 'alpha', 'corrcoef']]
    )
emp_scores = pd.concat(emp_scores).reset_index(drop=True)
emp_scores.to_csv(
    os.path.join(OUTPUT_DIR, 'consensus_scores.csv'),
    index=False
    )

# plot empirical scores as a function of alpha
emp_scores = pd.read_csv(os.path.join(OUTPUT_DIR, 'consensus_scores.csv'),
            index_col=False
)

plotting.plot_performance(
    df=emp_scores.copy(), x='alpha', y='corrcoef', normalize=True, hue='connectome',
    show=True, savefig=True,
    fname=os.path.join(FIG_DIR, 'performance_curve_rewired_connectomes'),
    ax_params={'ylabel': 'memory capacity'},
    rc_params={'savefig.format': 'eps'}
)


# #####################################################################
# Concatenate scores across nulls models for each connectome
# #####################################################################
def concat_null_scores(connectome):
    """ Concatenate null scores per connectome
    """
    null_scores = []
    for sample_id in range(500):
        df = pd.read_csv(
            os.path.join(OUTPUT_DIR, f'{connectome}_null_{sample_id}_scores.csv')
        ).reset_index(drop=True)

        df['sample_id'] = sample_id

        null_scores.append(
            df[['sample_id', 'alpha', 'corrcoef']]
        )

    null_scores = pd.concat(null_scores).reset_index(drop=True)

    null_scores.to_csv(
        os.path.join(OUTPUT_DIR, f'{connectome}_null_scores.csv'),
        index=False
        )

for connectome in connectomes:
    print(f'\n--------{connectome}----------')
    concat_null_scores(connectome)


# #####################################################################
# scores - empirical vs null models
# #####################################################################
emp_scores = []
null_scores = []
for connectome in connectomes[:]:

    df_emp = pd.read_csv(
        os.path.join(OUTPUT_DIR, f'{connectome}_empirical_scores.csv')
    ).reset_index(drop=True)

    df_null = pd.read_csv(
        os.path.join(OUTPUT_DIR, f'{connectome}_null_scores.csv'),
    ).reset_index(drop=True)

    # identify alpha at which maximum performance occurs
    alpha_max = df_emp.loc[df_emp['corrcoef'] == np.max(df_emp['corrcoef']), 'alpha'].values[0]  # critical alpha value
    print(f'\nConnectome: {connectome} -alpha_max: {alpha_max}')

    # filter rows corresponding to critical alpha
    df_emp = df_emp.loc[df_emp['alpha'] == alpha_max, :].reset_index()
    df_null = df_null.loc[df_null['alpha'] == alpha_max, :].reset_index()

    # z-score null model values
    df_null['z-score'] = ((df_null['corrcoef']-df_null['corrcoef'].mean())/df_null['corrcoef'].std())
    df_null['connectome'] = connectome
    df_null = df_null[['connectome', 'alpha', 'corrcoef', 'z-score']]

    # z-score empirical values
    df_emp['z-score'] = ((df_emp['corrcoef']-df_null['corrcoef'].mean())/df_null['corrcoef'].std())
    df_emp['connectome'] = connectome
    df_emp = df_emp[['connectome', 'alpha', 'corrcoef', 'z-score']]

    # determine p-value
    count = len(np.where(df_null['z-score'].to_numpy() > df_emp['z-score'].values)[0])
    n_nulls = len(df_null)
    df_emp['pval'] = count / n_nulls
    print(f'Connectome: {connectome} - p-val: {count / n_nulls}')

    emp_scores.append(df_emp)
    null_scores.append(df_null)
emp_scores = pd.concat(emp_scores).reset_index(drop=True)
null_scores = pd.concat(null_scores).reset_index(drop=True)

# plot empirical vs null scores (z-scored) at critical alpha
sns.set(style="ticks", font_scale=1.0)
fig = plt.figure(figsize=(10,5))
ax = plt.subplot(111)
sns.scatterplot(
    data=emp_scores,
    x='connectome',
    y='z-score',
    hue='connectome',
    hue_order=connectomes,
    s=30,
    ax=ax
)
sns.stripplot(
    data=null_scores,
    x='connectome',
    y='z-score',
    order=connectomes,
    # width=0.5,
    s=2.5,
    jitter=0.3,
    ax=ax
)
ax.set_ylim(-5,5)
sns.despine(offset=10, trim=True,
            top=True, bottom=False,
            right=True, left=False)
plt.show()
fig.savefig(os.path.join(FIG_DIR, 'rewired_nulls_connectomes.eps'),
            transparent=True, bbox_inches='tight', dpi=300)
plt.close()
