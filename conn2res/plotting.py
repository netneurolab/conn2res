# -*- coding: utf-8 -*-
"""
Plotting functions

@author: Estefany Suarez
"""

import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJ_DIR, 'figs')
if not os.path.isdir(FIG_DIR):
    os.makedirs(FIG_DIR)


class PCA:
    """
    Class that represents a simplified PCA object

    key features:
    - specifying the index of principal components we want to keep
    - principal components either unnormalized/normalized by singular values

    TODO
    """

    def __init__(self, idx_pcs=None, n_pcs=None, **kwargs):
        """
        Constructor class for time series
        """
        # indexes of principal components to keep
        if idx_pcs is not None:
            if isinstance(idx_pcs, list):
                idx_pcs = np.array(idx_pcs)
            if isinstance(idx_pcs, int):
                idx_pcs = np.array([idx_pcs])
            self.idx_pcs = idx_pcs
            self.n_pcs = len(self.idx_pcs)

        # number of principal components to keep
        if n_pcs is not None:
            self.setdefault('n_pcs', n_pcs)

    def setdefault(self, attribute, value):
        # add attribute (with given value) if not existing
        if not hasattr(self, attribute):
            setattr(self, attribute, value)

    def fit(self, data, full_matrices=False, **kwargs):
        # fit PCA
        self.u, self.s, self.vh = svd(data, full_matrices=full_matrices)

        # set number of principal components if not existing
        self.setdefault('n_pcs', self.s.size)

        # set indexes of principal components if not existing
        self.setdefault('idx_pcs', np.arange(self.n_pcs))

        return self

    def transform(self, data, normalize=False, **kwargs):
        # transform data into principal components
        pc = (data @ self.vh.T[:, self.idx_pcs]).reshape(-1, self.n_pcs)

        # normalize principal components by singular values (loop for efficiency)
        if normalize == True:
            for i in range(self.n_pcs):
                pc[:, i] /= self.s[self.idx_pcs[i]]

        return pc

    def fit_transform(self, data, **kwargs):
        # fit PCA
        self.fit(data, **kwargs)

        # transform data into principal components
        return self.transform(data, **kwargs)


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
    plt.title(title)
    if savefig:
        fig.savefig(fname=os.path.join(FIG_DIR, f'{title}_score.png'),
                    transparent=True, bbox_inches='tight', dpi=300)
    plt.show(block=block)


def plot_time_series(x, feature_set='orig', idx_ts=None, idx_features=None, n_features=None,
                     scaler=1, num=1, figsize=(12, 6), subplot=None, title=None, fname='time_course',
                     legend_label='', savefig=False, block=True, **kwargs):
    # transform data
    x = transform_data(
        x, feature_set, idx_features=idx_features, n_features=n_features, scaler=scaler, **kwargs)

    # open figure and create subplot
    plt.figure(num=num, figsize=figsize)
    if subplot is None:
        subplot = (1, 1, 1)
    plt.subplot(*subplot)

    # index of time series
    if idx_ts is None:
        idx_ts = np.arange(100)

    # plot data
    plt.plot(x[idx_ts])

    # plot legend
    legend = [f'{legend_label} {n+1}' for n in range(x.shape[1])]
    try:  # quick fix to get previously plotted legends
        lg = plt.gca().lines[-1].axes.get_legend()
        legend = [text.get_text() for text in lg.texts] + legend
    except:
        pass
    plt.legend(legend, loc='upper right', fontsize=22)

    # add title
    if title is not None:
        plt.title(f'{title} time course', fontsize=22)

    # set xtick/ythick fontsize
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    # set tight layout in case there are different subplots
    plt.tight_layout()

    if savefig:
        plt.savefig(fname=os.path.join(FIG_DIR, f'{fname}.png'),
                    transparent=True, bbox_inches='tight', dpi=300)
    plt.show(block=block)


def transform_data(data, feature_set, idx_features=None, n_features=None, scaler=1, **kwargs):
    if feature_set == 'pc':
        # transform data into principal components
        data = PCA(idx_pcs=idx_features,
                   n_pcs=n_features).fit_transform(data, **kwargs)

    elif feature_set == 'rnd':
        # update default number of features
        if n_features is None:
            n_features = 1

        # choose feature columns randomly
        data = data[:, np.random.choice(
            np.arange(data.shape[1]), size=n_features)]

    elif feature_set == 'df':
        # calculate decision function using model fitted on time series
        data = kwargs['model'].decision_function(data)

    elif feature_set == 'coeff':
        raise NotImplementedError

    # scale features
    data *= scaler

    return data