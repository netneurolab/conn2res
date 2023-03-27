# -*- coding: utf-8 -*-
"""
Plotting functions

@author: Estefany Suarez
"""
import os
import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, minmax_scale
import seaborn as sns
import matplotlib.pyplot as plt

from .utils import *
from .readout import _check_xy_type, _check_x_dims, _check_y_dims


PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJ_DIR, 'figs')
if not os.path.isdir(FIG_DIR):
    os.makedirs(FIG_DIR)


def transform_data(
    data, feature_set, idx_features=None, n_features=None,
    scaler=None, model=None, **kwargs
):
    """
    #TODO
    _summary_

    Parameters
    ----------
    data : _type_
        _description_
    feature_set : _type_
        _description_
    idx_features : _type_, optional
        _description_, by default None
    n_features : _type_, optional
        _description_, by default None
    scaler : _type_, optional
        _description_, by default None
    model : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    if feature_set == 'pca':
        # transform data into principal components
        pca = PCA(n_components=n_features)
        data = pca.fit_transform(data, **kwargs)

    elif feature_set == 'rnd':
        # update default number of features
        if n_features is None:
            n_features = 1

        # choose features randomly
        data = data[:, np.random.choice(
            np.arange(data.shape[1]), size=n_features)]

    elif feature_set == 'decfun':
        # calculate decision function using model fitted on time series
        data = model.decision_function(data)

    elif feature_set == 'pred':
        # calculate predicted labels
        data = model.predict(data)[:, np.newaxis]

    elif feature_set == 'coeff':
        # update default number of features
        if n_features is None:
            n_features = 5

        # get coefficient from model
        if model.coef_.ndim > 1:
            idx_class = kwargs.get('idx_class', 0)
            coef = model.coef_[idx_class, :]
        else:
            coef = model.coef_

        # choose features that correspond to largest absolute coefficients
        idx_coef = np.argsort(np.absolute(coef))
        if sum(coef != 0) > n_features:
            # use top 5 features
            idx_coef = idx_coef[-1*n_features:]
        else:
            # use <5 non-zero features
            idx_coef = np.intersect1d(idx_coef, np.where(coef != 0)[0])

        # scale time series with coefficients
        data = data[:, idx_coef]
        if data.size > 0:
            data = data @ np.diag(coef[idx_coef])
            # data = np.sum(
            #     data @ np.diag(coef[idx_coef]), axis=1).reshape(-1, 1)

    # select given features
    if idx_features is not None:
        data = data[:, idx_features]

    # scale features
    if scaler is not None:
        if scaler == 'l1-norm':
            scaler = norm(data, ord=1, axis=0)
        if scaler == 'l2-norm':
            scaler = norm(data, ord=2, axis=0)
        elif scaler == 'max':
            scaler = norm(data, ord=np.inf, axis=0)
        elif isinstance(scaler, int):
            scaler = np.array([int])
        data /= scaler

    return data


def plot_iodata(
    x, y, n_trials=7, title=None, show=True, savefig=False, fname=None,
    **kwargs
):
    """
    #TODO
    _summary_

    Parameters
    ----------
    x : _type_
        _description_
    y : _type_
        _description_
    title : _type_, optional
        _description_, by default None
    show : bool, optional
        _description_, by default True
    savefig : bool, optional
        _description_, by default False
    fname : _type_, optional
        _description_, by default None

    """

    # get end points for trials to plot trial separators
    if isinstance(x, list):
        n_trials = np.min([len(x), 10])
        x = x[:n_trials]
        y = y[:n_trials]

        tf, end_points = 0, []
        for _, trial in enumerate(x):
            tf += len(trial)
            end_points.append(tf)
    else:
        end_points = None

    # check X and y are arrays
    x, y = _check_xy_type(x, y)

    # check X and y dimensions
    x = _check_x_dims(x)
    y = _check_y_dims(y)

    # set plotting theme
    sns.set(style="ticks", font_scale=1.0)
    fig, ax = plt.subplots(1, 1, figsize=(12, 2))  # 12, 4.5

    # set color palette
    palette = kwargs.pop('palette', None)

    # plot inputs (x) and outputs (y)
    sns.lineplot(
        data=x, palette=palette, dashes=False, legend=False, ax=ax,
        linewidth=1.0, **kwargs
    )
    sns.lineplot(
        data=y, palette=palette, dashes=False, legend=False, ax=ax,
        linewidth=1.5, **kwargs
    )

    # set axis labels
    ax.set_xlabel('time steps', fontsize=11)
    ax.set_ylabel('signal amplitude', fontsize=11)

    # xlabels, ylabels
    if x.ndim == 1:
        x_labels = ['x']
    else:
        x_labels = [f'x{n+1}' for n in range(x.shape[1])]
    if y.ndim == 1:
        y_labels = ['y']
    else:
        y_labels = [f'y{n+1}' for n in range(y.shape[1])]

    # set legend
    new_labels = x_labels + y_labels
    ax.legend(handles=ax.lines, labels=new_labels, loc='best',
              fontsize=8)

    # plot trial line separators
    if end_points is not None:
        min_y = np.min(y).astype(int)
        max_y = np.max(y).astype(int)
        for tf in end_points:
            ax.plot(
                tf * np.ones((2)), np.array([min_y, max_y]), c='black',
                linestyle='--', linewidth=1.0
            )

    # set title
    if title is not None:
        plt.title(title, fontsize=12)

    sns.despine(offset=10, trim=True,
                top=True, bottom=False,
                right=True, left=False)

    if show:
        plt.show()

    if savefig:
        if fname is None:
            fname = 'io_data'

        fig.savefig(fname=os.path.join(FIG_DIR, f'{fname}.png'),
                    transparent=True, bbox_inches='tight', dpi=300)

    plt.close()


def plot_reservoir_states(
    x, reservoir_states, n_trials=7, title=None, show=True, savefig=False, fname=None,
    **kwargs
):

    # get end points for trials to plot trial separators
    if isinstance(reservoir_states, list):
        n_trials = np.min([len(x), 10])
        x = x[-n_trials:]
        reservoir_states = reservoir_states[-n_trials:]

        tf, end_points = 0, []
        for _, trial in enumerate(x):
            tf += len(trial)
            end_points.append(tf)
    else:
        end_points = None

    # check X is array
    x, _ = _check_xy_type(x, None)

    # check reservoir_states is array
    if isinstance(reservoir_states, (list, tuple)):
        reservoir_states = concat(reservoir_states)

    # check X dimensions
    x = _check_x_dims(x)

    # set plotting theme
    sns.set(style="ticks", font_scale=1.0)
    fig, axs = plt.subplots(
        2, 1, figsize=(12, 4), sharex=True, tight_layout=True
    )
    axs = axs.ravel()

    plt.subplots_adjust(wspace=0.1)

    # set color palette
    palette = kwargs.pop('palette', None)

    # plot inputs (x) and reservoir states
    sns.lineplot(
        data=x, palette=palette, dashes=False, legend=False, ax=axs[0],
        linewidth=1.0, **kwargs
    )

    palette = sns.color_palette("tab10", reservoir_states.shape[1])
    reservoir_states = minmax_scale(scale(reservoir_states, with_std=False), feature_range=(-1, 1))
    sns.lineplot(
        data=reservoir_states, palette=palette, dashes=False, legend=False,
        linewidth=0.5, ax=axs[1], **kwargs
    )

    # set axis labels
    axs[0].set_ylabel('x signal \namplitude', fontsize=11)
    axs[1].set_ylabel('reservoir \nstates', fontsize=11)
    axs[1].set_xlabel('time steps', fontsize=11)

    # xlabels, ylabels
    if x.ndim == 1:
        x_labels = ['x']
    else:
        x_labels = [f'x{n+1}' for n in range(x.shape[1])]

    # set legend
    axs[0].legend(
        handles=axs[0].lines, labels=x_labels, loc='best', fontsize=8
        )

    # plot trial line separators
    if end_points is not None:
        min_x = np.min(x).astype(int)
        max_x = np.max(x).astype(int)
        min_res_states = np.min(reservoir_states).astype(int)
        max_res_states = np.max(reservoir_states).astype(int)
        for tf in end_points:
            axs[0].plot(
                tf * np.ones((2)), np.array([min_x, max_x]), c='black',
                linestyle='--', linewidth=1.0
            )
            axs[1].plot(
                tf * np.ones((2)), np.array([min_res_states, max_res_states]),
                c='black', linestyle='--', linewidth=1.0
            )

    # set title
    if title is not None:
        plt.suptitle(title, fontsize=12)

    sns.despine(offset=10, trim=True,
                top=True, bottom=False,
                right=True, left=False)

    if show:
        plt.show()

    if savefig:
        if fname is None:
            fname = 'io_data'

        fig.savefig(fname=os.path.join(FIG_DIR, f'{fname}.png'),
                    transparent=True, bbox_inches='tight', dpi=300)

    plt.close()


def plot_diagnostics(
    x, y, reservoir_states, trained_model,
    idx_features=None, n_features=None, scaler=None,
    title=None, show=True, savefig=False, fname=None, **kwargs
):
    """
    #TODO
    _summary_

    Parameters
    ----------
    x : _type_
        _description_
    y : _type_
        _description_
    reservoir_states : _type_
        _description_
    trained_model : _type_
        _description_
    idx_features : _type_, optional
        _description_, by default None
    n_features : _type_, optional
        _description_, by default None
    scaler : _type_, optional
        _description_, by default None
    title : _type_, optional
        _description_, by default None
    show : bool, optional
        _description_, by default True
    savefig : bool, optional
        _description_, by default False
    fname : _type_, optional
        _description_, by default None
    """
    # check X and y are arrays
    x, y = _check_xy_type(x, y)

    # check reservoir_states is an array
    if isinstance(reservoir_states, (list, tuple)):
        reservoir_states = concat(reservoir_states)

    # check X and y dimensions
    x = _check_x_dims(x)
    y = _check_y_dims(y)

    # transform data
    x_trans = transform_data(
        x, feature_set='data', idx_features=idx_features,
        n_features=n_features, scaler=scaler, **kwargs
    )

    dec_func = transform_data(
        reservoir_states, feature_set='decfun', idx_features=idx_features,
        n_features=n_features, scaler=scaler, model=trained_model, **kwargs
    )

    y_trans = transform_data(
        y, feature_set='data', idx_features=idx_features,
        n_features=n_features, scaler=scaler, **kwargs
    )

    y_pred = transform_data(
        reservoir_states, feature_set='pred', idx_features=idx_features,
        n_features=n_features, scaler=scaler, model=trained_model, **kwargs
    )

    # set plotting theme
    sns.set(style="ticks", font_scale=1.0)
    fig, axs = plt.subplots(
        3, 1, figsize=(12, 6), sharex=True, tight_layout=True
    )
    axs = axs.ravel()

    plt.subplots_adjust(wspace=0.1)

    # set color palette
    palette = kwargs.pop('palette', None)

    # plot
    sns.lineplot(
        data=x_trans[:160], palette=palette,
        dashes=False, legend=False, linewidth=1.0, ax=axs[0])
    sns.lineplot(
        data=dec_func[:160], palette=palette,
        dashes=False, legend=False, linewidth=1.0, ax=axs[1])
    sns.lineplot(
        data=y_trans[:160], palette=palette,
        dashes=False, legend=False, linewidth=1.0, ax=axs[2])

    if y_pred.ndim:
        n_colors = 1
    else:
        n_colors = y_pred.shape[1]
    palette = sns.color_palette("tab10", n_colors+1)[1:]
    sns.lineplot(
        data=y_pred[:160], palette=palette,
        dashes=False, legend=False, ax=axs[2], linewidth=1.5)

    # set axis labels
    axs[0].set_ylabel('x signal \namplitude', fontsize=11)
    axs[1].set_ylabel('decision \nfunction', fontsize=11)
    axs[2].set_xlabel('time steps', fontsize=11)
    axs[2].set_ylabel('y signal \namplitude', fontsize=11)

    # set axis limits
    for ax in axs:
        ax.set_xlim(0, 160)

    # create legend labels
    if x.ndim == 1:
        x_labels = ['x']
    else:
        x_labels = [f'x{n+1}' for n in range(x.shape[1])]

    if dec_func.ndim == 1:
        dec_func_labels = ['decision function']
    else:
        dec_func_labels = [f'decision function {n+1}' for n in range(dec_func.shape[1])]

    # set legend
    axs[0].legend(handles=axs[0].lines, labels=x_labels,
                  loc='upper right', fontsize=8)
    axs[1].legend(handles=axs[1].lines, labels=dec_func_labels,
                  loc='upper right', fontsize=8)
    axs[2].legend(handles=axs[2].lines, labels=['target', 'predicted target'],
                  loc='upper right', fontsize=8)

    # set title
    if title is not None:
        plt.suptitle(title, fontsize=12)

    sns.despine(offset=10, trim=False,
                top=True, bottom=False,
                right=True, left=False)

    if show:
        plt.show()

    if savefig:
        if fname is None:
            fname = 'diagnostics_curve'

        fig.savefig(fname=os.path.join(FIG_DIR, f'{fname}.png'),
                    transparent=True, bbox_inches='tight', dpi=300)

    plt.close()


def plot_performance(
    df, x='alpha', y='score', normalize=False,
    title=None, show=True, savefig=False, fname=None, **kwargs
):

    if normalize:
        df[y] = df[y] / max(df[y])

    # set plotting theme
    sns.set(style="ticks", font_scale=1.0)
    fig, ax = plt.subplots(1, 1, figsize=(6, 2))

    # set color palette
    hue = kwargs.pop('hue', None)
    if hue is not None:
        n_hues = len(np.unique(df[hue]))
        palette = sns.color_palette('husl', n_hues+1)[:n_hues]
    else:
        palette = sns.color_palette('husl')

    # plot
    sns.lineplot(
        data=df, x=x, y=y, hue=hue, palette=palette, markers=True,
        legend=True, ax=ax, **kwargs)

    # set axis labels
    ax.set_xlabel('alpha', fontsize=11)
    y_label = ' '.join(y.split('_'))
    ax.set_ylabel(y_label, fontsize=11)

    # set legend
    plt.legend(loc='upper right', fontsize=8)

    # set title
    if title is not None:
        plt.title(title, fontsize=12)

    sns.despine(offset=10, trim=True,
                top=True, bottom=False,
                right=True, left=False)

    if show:
        plt.show()

    if savefig:
        if fname is None:
            fname = 'performance_curve'

        fig.savefig(fname=os.path.join(FIG_DIR, f'{fname}.png'),
                    transparent=True, bbox_inches='tight', dpi=300)

    plt.close()


def plot_phase_space(x, y, sample=None, xlim=None, ylim=None, subplot=None, cmap=None,
    num=1, figsize=(13, 5), title=None, fname='phase_space', savefig=False, block=False
):
    #TODO
    # open figure and create subplot
    plt.figure(num=num, figsize=figsize)
    if subplot is None:
        subplot = (1, 1, 1)
    plt.subplot(*subplot)

    # plot data
    if sample is None:
        plt.plot(x)
    else:
        t = np.arange(*sample)
        if cmap is None:
            plt.plot(t, x[t])
        else:
            for i, _ in enumerate(t[:-1]):
                plt.plot(x[t[i:i+2]], y[t[i:i+2]],
                         color=getattr(plt.cm, cmap)(255*i//np.diff(sample)))

    # add x and y limits
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.xlim(ylim)

    # set xtick/ythick fontsize
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    # add title
    if title is not None:
        plt.title(f'{title} phase space', fontsize=22)

    # set tight layout in case there are different subplots
    plt.tight_layout()

    if savefig:
        plt.savefig(fname=os.path.join(FIG_DIR, f'{fname}.png'),
                    transparent=True, bbox_inches='tight', dpi=300)
    plt.show(block=block)
