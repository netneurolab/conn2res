# -*- coding: utf-8 -*-
"""
Plotting functions

@author: Estefany Suarez
"""

import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJ_DIR, 'figs')
if not os.path.isdir(FIG_DIR):
    os.makedirs(FIG_DIR)


def plot_task(x, y, title, num=1, figsize=(12, 10), savefig=False, block=True):

    fig = plt.figure(num=num, figsize=figsize)
    ax = plt.subplot(111)

    # xlabels, ylabels
    try:
        x_labels = [f'I{n+1}' for n in range(x.shape[1])]
    except:
        x_labels = 'I1'
    try:
        y_labels = [f'O{n+1}' for n in range(y.shape[1])]
    except:
        y_labels = 'O1'

    plt.plot(x[:], label=x_labels)
    plt.plot(y[:], label=y_labels)
    plt.legend()
    plt.suptitle(title)

    sns.despine(offset=10, trim=True)
    if savefig:
        fig.savefig(fname=os.path.join(FIG_DIR, f'{title}_io.png'),
                    transparent=True, bbox_inches='tight', dpi=300)
    plt.show(block=block)


def plot_performance_curve(df, title, num=2, figsize=(12, 10), savefig=False, block=True):

    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(num=num, figsize=figsize)
    ax = plt.subplot(111)

    n_modules = len(np.unique(df['module']))
    palette = sns.color_palette('husl', n_modules+1)[:n_modules]

    if 'VIS' in list(np.unique(df['module'])):
        hue_order = ['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN']
    else:
        hue_order = None

    sns.lineplot(data=df, x='alpha', y='score',
                 hue='module',
                 hue_order=hue_order,
                 palette=palette,
                 markers=True,
                 ax=ax)

    sns.despine(offset=10, trim=True)
    if savefig:
        fig.savefig(fname=os.path.join(FIG_DIR, f'{title}_score.png'),
                    transparent=True, bbox_inches='tight', dpi=300)
    plt.title(title)
    plt.show(block=block)
