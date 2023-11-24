# -*- coding: utf-8 -*-
"""
Example 1: Inferences on global network organization
=======================================================================
This script includes the analysis code for example 1
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from conn2res import plotting

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJ_DIR, 'examples', 'results', 'example1')
FIG_DIR = os.path.join(PROJ_DIR, 'examples', 'figs')
if not os.path.isdir(FIG_DIR):
    os.makedirs(FIG_DIR)

# list of consensus connectomes (empirical networks)
connectomes = [
    'consensus_0',
    'consensus_1',
    'consensus_2',
    'consensus_3',
    'consensus_4',
    'consensus_5'
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
        df[['connectome', 'alpha', 'module', 'n_nodes', 'corrcoef']]
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

# average scores across modules (i.e., resting state networks)
emp_scores = emp_scores.groupby(['connectome', 'alpha']).mean().reset_index()
emp_scores = emp_scores[['connectome', 'alpha', 'corrcoef']]

plotting.plot_performance(
    df=emp_scores.copy(), x='alpha', y='corrcoef', normalize=True, hue='connectome',
    show=True, savefig=True,
    fname=os.path.join(FIG_DIR, 'performance_curve_rewired_human'),
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
            df[['sample_id', 'alpha', 'module', 'n_nodes', 'corrcoef']]
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
# scores - empirical vs null models at critical point
# #####################################################################
alpha_crit = 1.0  # critical alpha value

# filter rows corresponding to critical alpha
emp_scores = emp_scores.loc[emp_scores['alpha'] == alpha_crit, :].reset_index()

# initialize columns for hypothesys testing
emp_scores['z-score'] = 0
emp_scores['pval'] = 0

# nonparametric hypothesis testing
null_scores = []
for connectome in connectomes[:]:
    df = pd.read_csv(
        os.path.join(OUTPUT_DIR, f'{connectome}_null_scores.csv'),
        index_col=False
    )

    # filter rows corresponding to critical alpha
    df = df.loc[df['alpha'] == alpha_crit, :].reset_index()
    df = df.groupby('sample_id').mean().reset_index()

    # z-score null model values
    df['z-score'] = ((df['corrcoef']-df['corrcoef'].mean())/df['corrcoef'].std())
    df['connectome'] = connectome
    df = df[['connectome', 'alpha', 'corrcoef', 'z-score']]

    # z-score empirical values
    emp_score = emp_scores.loc[emp_scores['connectome'] == connectome, 'corrcoef'].values
    z_score_emp = ((emp_score-df['corrcoef'].mean())/df['corrcoef'].std())
    emp_scores.loc[emp_scores['connectome'] == connectome, 'z-score'] = z_score_emp

    # determine p-value
    count = len(np.where(df['z-score'].to_numpy() > z_score_emp)[0])
    n_nulls = len(df)
    emp_scores.loc[emp_scores['connectome'] == connectome, 'pval'] = count / n_nulls
    print(f'Connectome: f{connectome} - p-val: {count / n_nulls}')

    null_scores.append(df)
null_scores = pd.concat(null_scores).reset_index(drop=True)

# plot empirical vs null scores (z-scored) at critical alpha
sns.set(style="ticks", font_scale=1.0)
fig = plt.figure(figsize=(5,5))
ax = plt.subplot(111)
sns.scatterplot(
    data=emp_scores,
    x='connectome',
    y='z-score',
    s=30,
    c='black',
    ax=ax
)
sns.stripplot(
    data=null_scores,
    x='connectome',
    y='z-score',
    # width=0.5,
    s=2.5,
    jitter=0.3,
    ax=ax
)
ax.set_ylim(-2.5, 12.5)
sns.despine(offset=10, trim=True,
            top=True, bottom=False,
            right=True, left=False)
plt.show()
fig.savefig(os.path.join(FIG_DIR, 'rewired_nulls_human.eps'),
            transparent=True, bbox_inches='tight', dpi=300)
plt.close()
