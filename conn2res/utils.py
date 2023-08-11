# -*- coding: utf-8 -*-
"""
Functions for basic matrix manipulations
"""
import numpy as np

# consensus
from sklearn.utils.validation import (check_consistent_length, check_array)
def _ecdf(data):
    """
    Estimates empirical cumulative distribution function of `data`

    Taken directly from StackOverflow. See original answer at
    https://stackoverflow.com/questions/33345780.

    Parameters
    ----------
    data : array_like

    Returns
    -------
    prob : numpy.ndarray
        Cumulative probability
    quantiles : numpy.darray
        Quantiles
    """

    sample = np.atleast_1d(data)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    prob = np.cumsum(counts).astype(float) / sample.size

    # match MATLAB
    prob, quantiles = np.append([0], prob), np.append(quantiles[0], quantiles)

    return prob, quantiles


def struct_consensus(data, distance, hemiid, weighted=False):
    """
    Calculates distance-dependent group consensus structural connectivity graph

    Takes as input a weighted stack of connectivity matrices with dimensions
    (N, N, S) where `N` is the number of nodes and `S` is the number of
    matrices or subjects. The matrices must be weighted, and ideally with
    continuous weights (e.g. fractional anisotropy rather than streamline
    count). The second input is a pairwise distance matrix, where distance(i,j)
    is the Euclidean distance between nodes i and j. The final input is an
    (N, 1) vector which labels nodes as belonging to the right (`hemiid==0`) or
    left (`hemiid=1`) hemisphere (note that these values can be flipped as long
    as `hemiid` contains only values of 0 and 1).

    This function estimates the average edge length distribution and builds
    a group-averaged connectivity matrix that approximates this distribution
    with density equal to the mean density across subjects.

    The algorithm works as follows:

    1. Estimate the cumulative edge length distribution,
    2. Divide the distribution into M length bins, one for each edge that will
       be added to the group-average matrix, and
    3. Within each bin, select the edge that is most consistently expressed
       expressed across subjects, breaking ties according to average edge
       weight (which is why the input matrix `data` must be weighted).

    The algorithm works separately on within/between hemisphere links.

    Parameters
    ----------
    data : (N, N, S) array_like
        Weighted connectivity matrices (i.e., fractional anisotropy), where `N`
        is nodes and `S` is subjects
    distance : (N, N) array_like
        Array where `distance[i, j]` is the Euclidean distance between nodes
        `i` and `j`
    hemiid : (N, 1) array_like
        Hemisphere designation for `N` nodes where a value of 0/1 indicates
        node `N_{i}` is in the right/left hemisphere, respectively
    weighted : bool
        Flag indicating whether or not to return a weighted consensus map. If
        `True`, the consensus will be multiplied by the mean of `data`.

    Returns
    -------
    consensus : (N, N) numpy.ndarray
        Binary (default) or mean-weighted group-level connectivity matrix

    References
    ----------
    Betzel, R. F., Griffa, A., Hagmann, P., & Mišić, B. (2018). Distance-
    dependent consensus thresholds for generating group-representative
    structural brain networks. Network Neuroscience, 1-22.
    """

    # confirm input shapes are as expected
    check_consistent_length(data, distance, hemiid)
    try:
        hemiid = check_array(hemiid, ensure_2d=True)
    except ValueError:
        raise ValueError('Provided hemiid must be a 2D array. Reshape your '
                         'data using array.reshape(-1, 1) and try again.')

    num_node, _, num_sub = data.shape      # info on connectivity matrices
    pos_data = data > 0                    # location of + values in matrix
    pos_data_count = pos_data.sum(axis=2)  # num sub with + values at each node

    with np.errstate(divide='ignore', invalid='ignore'):
        average_weights = data.sum(axis=2) / pos_data_count

    # empty array to hold inter/intra hemispheric connections
    consensus = np.zeros((num_node, num_node, 2))

    for conn_type in range(2):  # iterate through inter/intra hemisphere conn
        if conn_type == 0:      # get inter hemisphere edges
            inter_hemi = (hemiid == 0) @ (hemiid == 1).T
            keep_conn = np.logical_or(inter_hemi, inter_hemi.T)
        else:                   # get intra hemisphere edges
            right_hemi = (hemiid == 0) @ (hemiid == 0).T
            left_hemi = (hemiid == 1) @ (hemiid == 1).T
            keep_conn = np.logical_or(right_hemi @ right_hemi.T,
                                      left_hemi @ left_hemi.T)

        # mask the distance array for only those edges we want to examine
        full_dist_conn = distance * keep_conn
        upper_dist_conn = np.atleast_3d(np.triu(full_dist_conn))

        # generate array of weighted (by distance), positive edges across subs
        pos_dist = pos_data * upper_dist_conn
        pos_dist = pos_dist[np.nonzero(pos_dist)]

        # determine average # of positive edges across subs
        # we will use this to bin the edge weights
        avg_conn_num = len(pos_dist) / num_sub

        # estimate empirical CDF of weighted, positive edges across subs
        cumprob, quantiles = _ecdf(pos_dist)
        cumprob = np.round(cumprob * avg_conn_num).astype(int)

        # empty array to hold group-average matrix for current connection type
        # (i.e., inter/intra hemispheric connections)
        group_conn_type = np.zeros((num_node, num_node))

        # iterate through bins (for edge weights)
        for n in range(1, int(avg_conn_num) + 1):
            # get current quantile of interest
            curr_quant = quantiles[np.logical_and(cumprob >= (n - 1),
                                                  cumprob < n)]
            if curr_quant.size == 0:
                continue

            # find edges in distance connectivity matrix w/i current quantile
            mask = np.logical_and(full_dist_conn >= curr_quant.min(),
                                  full_dist_conn <= curr_quant.max())
            i, j = np.where(np.triu(mask))  # indices of edges of interest

            c = pos_data_count[i, j]   # get num sub with + values at edges
            w = average_weights[i, j]  # get averaged weight of edges

            # find locations of edges most commonly represented across subs
            indmax = np.argwhere(c == c.max())

            # determine index of most frequent edge; break ties with higher
            # weighted edge
            if indmax.size == 1:  # only one edge found
                group_conn_type[i[indmax], j[indmax]] = 1
            else:                 # multiple edges found
                indmax = indmax[np.argmax(w[indmax])]
                group_conn_type[i[indmax], j[indmax]] = 1

        consensus[:, :, conn_type] = group_conn_type

    # collapse across hemispheric connections types and make symmetrical array
    consensus = consensus.sum(axis=2)
    consensus = np.logical_or(consensus, consensus.T).astype(int)

    if weighted:
        consensus = consensus * np.mean(data, axis=2)
    return consensus


# functions for connectivity matrix
def check_symmetric(a, tol=1e-16):
    """
    _summary_

    Parameters
    ----------
    a : _type_
        _description_
    tol : _type_, optional
        _description_, by default 1e-16

    Returns
    -------
    _type_
        _description_
    """
    return np.allclose(a, a.T, atol=tol)


def make_symmetric(a, copy_lower=True):
    """
    _summary_

    Parameters
    ----------
    a : _type_
        _description_
    copy_lower : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """
    if copy_lower:
        return np.tril(a, -1) + np.tril(a, -1).T
    else:
        return np.triu(a, 1) + np.triu(a, 1).T


def check_square(a):
    """
    _summary_

    Parameters
    ----------
    a : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    s = a.shape
    return s[0] == s[1]


# functions for X and y
def split(a, sections):
    """
    _summary_

    Parameters
    ----------
    a : _type_
        _description_
    sections : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    TypeError
        _description_
    """
    if isinstance(a, np.ndarray):
        return np.split(a, indices_or_sections=sections, axis=0)
    else:
        raise TypeError("")


def get_sections(a):
    """
    _summary_

    Parameters
    ----------
    a : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    TypeError
        _description_
    """
    if isinstance(a, (list, tuple)):
        lens = [len(i) for i in a]
        sections = np.array([np.sum(lens[:i]) for i in range(1, len(lens))])
    else:
        raise TypeError("")

    return sections


def concat(a):
    """
    _summary_

    Parameters
    ----------
    a : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    TypeError
        _description_
    """
    if isinstance(a, (list, tuple)):
        if a[0].ndim == 1:
            a = [a[:, np.newaxis] for a in a]
        return np.vstack(a).squeeze()
    elif isinstance(a, np.ndarray):
        return a
    else:
        raise TypeError("")


def check(*objs):
    def _all_bases(o):
        for b in o.__bases__:
            if b is not object:
                yield b
            yield from _all_bases(b)
    s = [(i.__class__, *_all_bases(i.__class__)) for i in objs]
    return len(set(*s[:1]).intersection(*s[1:])) > 0
