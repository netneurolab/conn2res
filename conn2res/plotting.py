# -*- coding: utf-8 -*-
"""
Plotting functions

@author: Estefany Suarez
"""
import os
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, minmax_scale
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

from .utils import *
from .readout import _check_xy_type, _check_x_dims, _check_y_dims


PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJ_DIR, 'figs')
if not os.path.isdir(FIG_DIR):
    os.makedirs(FIG_DIR)


def transform_data(
    data, feature_set, idx_features=None, n_features=None, scaler=None,
    model=None, **kwargs
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
    x, y, n_trials=7, palette=None,
    rc_params={}, fig_params={}, ax_params={}, lg_params={},
    title=None, show=True, savefig=False, fname='io_data',
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
    n_trials : _type_, optional
        _description_, by default 7
    palette : _type_, optional
        _description_, by default None
    rc_params : dict
        dictionary of matplotlib rc parameters, by default {}
    fig_params : dict
        dictionary of figure properties, by default {}
    ax_params : dict
        dictionary of axes properties, by default {}
    lg_params : dict
        dictionary of legend settings, by default {}
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
    rc_defaults = {'figure.titlesize': 12, 'axes.labelsize': 11,
                   'xtick.labelsize': 11, 'ytick.labelsize': 11,
                   'legend.fontsize': 8, 'legend.loc': 'best',
                   'lines.linewidth': 1, 'savefig.format': 'png'}
    rc_defaults.update(rc_params)
    sns.set_theme(style='ticks', rc=rc_defaults)
    
    # open figure and axes
    fig_defaults = {'figsize': (12, 2)}  # 12, 4.5
    fig_defaults.update(fig_params)
    fig = plt.figure(**fig_defaults)
    ax = fig.subplots(1, 1)

    # plot inputs (x) and outputs (y)
    sns.lineplot(
        data=x, palette=palette, dashes=False, legend=False, ax=ax,
        **kwargs
    )
    sns.lineplot(
        data=y, palette=palette, dashes=False, legend=False, ax=ax,
        linewidth=1.5, **kwargs
    )

    # set legend
    x_labels = ['x'] if x.ndim == 1 else [f'x{n+1}' for n in range(x.shape[1])]
    y_labels = ['y'] if y.ndim == 1 else [f'y{n+1}' for n in range(y.shape[1])]
    lg_defaults = {'labels': x_labels + y_labels}
    lg_defaults.update(**lg_params)
    ax.legend(handles=ax.lines, **lg_defaults)

    # set axes properties
    ax_defaults = {'xlabel': 'time steps', 'ylabel': 'signal amplitude',
                   'xlim': [0, 200]}
    ax_defaults.update(**ax_params)
    ax.set(**ax_defaults)

    # plot trial line separators
    if end_points is not None:
        min_y = np.min(y).astype(int)
        max_y = np.max(y).astype(int)
        for tf in end_points:
            ax.plot(
                tf * np.ones((2)), np.array([min_y, max_y]), c='black',
                linestyle='--'
            )

    # set title
    if title is not None:
        fig.suptitle(title)

    sns.despine(offset=10, trim=True,
                top=True, bottom=False,
                right=True, left=False)

    if show:
        plt.show(block=True)

    if savefig:
        fig.savefig(fname + '.' + mpl.rcParams['savefig.format'],
                    transparent=True, bbox_inches='tight', dpi=300)

    plt.close()

    # reset rc defaults
    mpl.rcdefaults()


def plot_reservoir_states(
    x, reservoir_states, n_trials=7, palette=None,
    rc_params={}, fig_params={}, ax_params=[{}] * 2, lg_params={},
    title=None, show=True, savefig=False, fname='res_states', **kwargs
):
    """
    _summary_

    Parameters
    ----------
    x : _type_
        _description_
    reservoir_states : _type_
        _description_
    n_trials : int, optional
        _description_, by default 7
    palette : _type_, optional
        _description_, by default None
    rc_params : dict
        dictionary of matplotlib rc parameters, by default {}
    fig_params : dict
        dictionary of figure properties, by default {}
    ax_params : list of dict
        list of dictionaries setting axes properties, by default [{}] * 2
    lg_params : dict
        dictionary of legend settings for first axis, by default {}
    title : _type_, optional
        _description_, by default None
    show : bool, optional
        _description_, by default True
    savefig : bool, optional
        _description_, by default False
    fname : _type_, optional
        _description_, by default 'res_states'
    """
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
    rc_defaults = {'figure.titlesize': 12, 'axes.labelsize': 11,
                   'xtick.labelsize': 11, 'ytick.labelsize': 11,
                   'legend.fontsize': 8, 'legend.loc': 'best',
                   'lines.linewidth': 1, 'savefig.format': 'png'}
    rc_defaults.update(rc_params)
    sns.set_theme(style='ticks', rc=rc_defaults)
    
    # open figure and axes
    fig_defaults = {'figsize': (12, 4), 'layout': 'tight'}
    fig_defaults.update(fig_params)
    fig = plt.figure(**fig_defaults)
    axs = fig.subplots(2, 1, sharex=True)
    axs = axs.ravel()

    fig.subplots_adjust(wspace=0.1)

    # plot inputs (x) and reservoir states
    sns.lineplot(
        data=x, palette=palette, dashes=False, legend=False, ax=axs[0],
        **kwargs
    )

    palette = sns.color_palette("tab10", reservoir_states.shape[1])
    reservoir_states = minmax_scale(
        scale(reservoir_states, with_std=False), feature_range=(-1, 1))
    sns.lineplot(
        data=reservoir_states, palette=palette, dashes=False, legend=False,
        linewidth=0.5, ax=axs[1], **kwargs
    )

    # set legend
    x_labels = ['x'] if x.ndim == 1 else [f'x{n+1}' for n in range(x.shape[1])]
    lg_defaults = {'labels': x_labels}
    lg_defaults.update(**lg_params)
    axs[0].legend(handles=axs[0].lines, **lg_defaults)

    # set axes properties
    xlabel = ['', 'time steps']
    ylabel = ['x signal \namplitude', 'reservoir \nstates']
    for i, ax in enumerate(axs):
        ax_defaults = {'xlim': [0, 200], 'xlabel': xlabel[i],
                       'ylabel': ylabel[i]}
        ax_defaults.update(**ax_params[i])
        ax.set(**ax_defaults)

    # plot trial line separators
    if end_points is not None:
        min_x = np.min(x).astype(int)
        max_x = np.max(x).astype(int)
        min_res_states = np.min(reservoir_states).astype(int)
        max_res_states = np.max(reservoir_states).astype(int)
        for tf in end_points:
            axs[0].plot(
                tf * np.ones((2)), np.array([min_x, max_x]), c='black',
                linestyle='--',
            )
            axs[1].plot(
                tf * np.ones((2)), np.array([min_res_states, max_res_states]),
                c='black', linestyle='--',
            )

    # set title
    if title is not None:
        fig.suptitle(title)

    sns.despine(offset=10, trim=True,
                top=True, bottom=False,
                right=True, left=False)

    if show:
        plt.show(block=True)

    if savefig:
        fig.savefig(fname + '.' + mpl.rcParams['savefig.format'],
                    transparent=True, bbox_inches='tight', dpi=300)

    plt.close()

    # reset rc defaults
    mpl.rcdefaults()


def plot_diagnostics(
    x, y, reservoir_states, trained_model, idx_features=None,
    n_features=None, scaler=None, palette=None,
    rc_params={}, fig_params={}, ax_params=[{}] * 3, lg_params=[{}] * 3,
    title=None, show=True, savefig=False, fname='diagnostics_curve', **kwargs
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
    palette : _type_, optional
        _description_, by default None
    rc_params : dict
        dictionary of matplotlib rc parameters
    fig_params : dict
        dictionary of figure properties
    ax_params : list of dict
        list of dictionaries setting axes properties, by default [{}] * 3
    lg_params : list of dict
        list of dictionaries setting legend, by default [{}] * 3
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
    rc_defaults = {'figure.titlesize': 12, 'axes.labelsize': 11,
                   'xtick.labelsize': 11, 'ytick.labelsize': 11,
                   'legend.fontsize': 8, 'legend.loc': 'upper right',
                   'lines.linewidth': 1, 'savefig.format': 'png'}
    rc_defaults.update(rc_params)
    sns.set_theme(style='ticks', rc=rc_defaults)
    
    # open figure and axes
    fig_defaults = {'figsize': (12, 6), 'layout': 'tight'}
    fig_defaults.update(fig_params)
    fig = plt.figure(**fig_defaults)
    axs = fig.subplots(3, 1, sharex=True)
    axs = axs.ravel()

    fig.subplots_adjust(wspace=0.1)

    # plot
    data = [x_trans, dec_func, y_trans]
    for i, ax in enumerate(axs):
        sns.lineplot(
            data=data[i][:160], palette=palette, dashes=False,
            legend=False, ax=ax)

    if y_pred.ndim:
        n_colors = 1
    else:
        n_colors = y_pred.shape[1]
    palette = sns.color_palette("tab10", n_colors+1)[1:]
    sns.lineplot(
        data=y_pred[:160], palette=palette, dashes=False, legend=False,
        ax=axs[2], linewidth=1.5)

    # set legend
    labels = [['x'] if x.ndim == 1 else [f'x{n+1}' for n in range(x.shape[1])],
              ['decision function'] if dec_func.ndim == 1 else [f'decision function {n+1}' for n in range(dec_func.shape[1])],
              ['target', 'predicted target']]
    for i, ax in enumerate(axs):
        lg_defaults = {'labels': labels[i]}
        lg_defaults.update(**lg_params[i])
        ax.legend(handles=ax.lines, **lg_defaults)
    
    # set axes properties
    xlabel = ['', '', 'time steps']
    ylabel = ['x signal \namplitude', 'decision \nfunction', 'y signal \namplitude']
    for i, ax in enumerate(axs):
        ax_defaults = {'xlim': [0, 160], 'xlabel': xlabel[i], 'ylabel': ylabel[i]}
        ax_defaults.update(**ax_params[i])
        ax.set(**ax_defaults)
    
    # set title
    if title is not None:
        fig.suptitle(title)

    sns.despine(offset=10, trim=False,
                top=True, bottom=False,
                right=True, left=False)

    if show:
        plt.show(block=True)

    if savefig:
        fig.savefig(fname=fname + '.' + mpl.rcParams['savefig.format'],
                    transparent=True, bbox_inches='tight', dpi=300)

    plt.close()

    # reset rc defaults
    mpl.rcdefaults()


def plot_performance(
    df, x='alpha', y='score', normalize=False, hue=None,
    rc_params={}, fig_params={}, ax_params={}, lg_params={},
    title=None, show=True, savefig=False, fname='performance_curve', **kwargs
):
    """
    _summary_

    Parameters
    ----------
    df : _type_
        _description_
    x : str, optional
        _description_, by default 'alpha'
    y : str, optional
        _description_, by default 'score'
    normalize : bool, optional
        _description_, by default False
    hue : optional
        _description_, by default None
    rc_params : dict
        dictionary of matplotlib rc parameters
    fig_params : dict
        dictionary of figure properties
    ax_params : dict
        dictionary of axes properties
    lg_params : dict
        dictionary of legend settings
    title : optional
        _description_, by default None
    show : bool, optional
        _description_, by default True
    savefig : bool, optional
        _description_, by default False
    fname : _type_, optional
        _description_, by default 'performance_curve'
    """
    if normalize:
        df[y] = df[y] / max(df[y])

    # set plotting theme
    rc_defaults = {'figure.titlesize': 12, 'axes.labelsize': 11,
                   'xtick.labelsize': 11, 'ytick.labelsize': 11,
                   'legend.fontsize': 8, 'legend.loc': 'upper right',
                   'savefig.format': 'png'}
    rc_defaults.update(rc_params)
    sns.set_theme(style='ticks', rc=rc_defaults)
    
    # open figure and axes
    fig_defaults = {'figsize': (6, 2)}
    fig_defaults.update(fig_params)
    fig = plt.figure(**fig_defaults)
    ax = fig.subplots(1, 1)

    # set color palette
    if hue is not None:
        n_hues = len(np.unique(df[hue]))
        palette = sns.color_palette('husl', n_hues+1)[:n_hues]
    else:
        palette = sns.color_palette('husl')

    # plot
    sns.lineplot(
        data=df, x=x, y=y, hue=hue, palette=palette, dashes=False,
        legend=False, markers=True, ax=ax, **kwargs)

    # set legend
    try:
        lg_defaults = {'labels': kwargs['hue_order']}
    except:
        lg_defaults = {'labels': list(pd.unique(df[hue]))}
    lg_defaults.update(**lg_params)
    ax.legend(handles=ax.lines, **lg_defaults)

    # set axis properties
    axes_defaults = {'xlabel': x, 'ylabel': ' '.join(y.split('_'))}
    axes_defaults.update(**ax_params)
    ax.set(**axes_defaults)

    # set title
    if title is not None:
        fig.suptitle(title)

    sns.despine(offset=10, trim=True,
                top=True, bottom=False,
                right=True, left=False)

    if show:
        plt.show(block=True)

    if savefig:
        fig.savefig(fname=fname + '.' + mpl.rcParams['savefig.format'],
                    transparent=True, bbox_inches='tight', dpi=300)

    plt.close()

    # reset rc defaults
    mpl.rcdefaults()
    

def plot_phase_space(
    x, y, sample=None, palette=None,
    fig_params={}, ax_params={}, rc_params={},
    title=None, show=False, savefig=False, fname='phase_space'
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
    sample : _type_, optional
        _description_, by default None
    palette : _type_, optional
        _description_, by default None
    fig_params : dict
        dictionary of figure properties
    ax_params : dict
        dictionary of axes properties
    rc_params : dict
        dictionary of matplotlib rc parameters
    title : _type_, optional
        _description_, by default None
    show : bool, optional
        _description_, by default True
    savefig : bool, optional
        _description_, by default False
    fname : _type_, optional
        _description_, by default 'phase_space'
    """    
    # set plotting theme
    rc_defaults = {'figure.titlesize': 12, 'axes.labelsize': 11,
                   'xtick.labelsize': 11, 'ytick.labelsize': 11,
                   'lines.linewidth': 1, 'savefig.format': 'png'}
    if palette is not None:
        # set cycler for color to change as a function of time step
        rc_defaults['axes.prop_cycle'] = cycler(color=sns.color_palette(palette, 256))
    rc_defaults.update(rc_params)
    sns.set_theme(style='ticks', rc=rc_defaults)
    
    # open figure and axes
    fig_defaults = {'figsize': (4, 4), 'layout': 'tight'}
    fig_defaults.update(fig_params)
    fig = plt.figure(**fig_defaults)
    ax = fig.subplots(1, 1)

    # plot data (these plots are easier with matplotlib)
    if sample is None:
        t = np.arange(x.shape[0])
    else:
        t = np.arange(*sample)
    if palette is None:
        ax.plot(x[t], y[t])
    else:
        for i in range(t.size-1):
            ax.plot(x[t[i:i+2]], y[t[i:i+2]])

    # set axis properties
    axes_defaults = {'xlim': [0.2, 1.4], 'ylim': [0.2, 1.4]}
    axes_defaults.update(**ax_params)
    ax.set(**axes_defaults)

    # set title
    if title is not None:
        fig.suptitle(title)

    sns.despine(offset=10, trim=False,
                top=True, bottom=False,
                right=True, left=False)
    
    if show:
        plt.show(block=True)

    if savefig:
        fig.savefig(fname=fname + '.' + mpl.rcParams['savefig.format'],
                    transparent=True, bbox_inches='tight', dpi=300)
        
        plt.close()
    
    # reset rc defaults
    mpl.rcdefaults()
