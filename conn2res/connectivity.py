# -*- coding: utf-8 -*-
"""
Functionality for connectivity matrix
"""
import os
import numpy as np
import warnings
from scipy.linalg import eigh
from bct import get_components, distance_bin

from .utils import *


PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_DIR, 'examples', 'data')


class Conn:
    """
    Class that represents a weighted or unweighted network using connectivity
    data

    Notes:
    1. The diagonal of the connectivity matrix is set to zero as well as 
    inf and nan values are replaced to zero.

    2. We makes sure that all nodes are connected to the rest of the network,
    otherwise the network is reduced to the largest component. Importantly,
    the original indexes of nodes is kept during this process, i.e., for
    instance, node 95 stays node 95 even if node 90 is removed.

    3. The input and output nodes should be set such that they belong to the
    largest component, otherwise the signal cannot propagate from/to them.

    4. Symmetric networks are checked for connectedness only in a weak sense,
    i.e., using a network where the directed edges are replaced with
    symmetric edges.

    Parameters
    ----------
    w : (N, N, M) numpy.ndarray, optional
        connectivity matrix (source, target) passed directly instead of
        being loaded from disc
        N: number of nodes in the network
        M: number of subjects
        subj_id is ignored if array is 2-dimensional (N, N)
    filename : str, optional
        filename of the connectivity matrix to be loaded from disc
        specified using full path (by default, the connectivity matrix is
        loaded from examples/data)
    subj_id : int, optional
        index of subject along axis=2 in the group level connectivity
        matrix
    modules : numpy.ndarray, optional
        array to store for each node which module it belongs to
    density : float, optional
        density to which the network should be set (note that
        connectedness in not checked during this process, so it should
        be used with care!)
    """

    def __init__(self, w=None, filename=None, subj_id=0, modules=None,
                 density=None):
        """
        Constructor method for class Conn
        """
        if w is not None:
            # assign provided connectivity data
            self.w = w
        else:
            # load connectivity data
            if filename is not None:
                self.w = np.load(filename)
            else:
                self.w = load_file('connectivity.npy')

            # select one subject from group connectivity data
            if subj_id is not None and self.w.ndim == 3:
                self.w = self.w[:, :, subj_id]

        # set zero diagonal
        np.fill_diagonal(self.w, 0)

        # remove inf and nan
        self.w[np.logical_or(np.isinf(self.w), np.isnan(self.w))] = 0

        # make sure weights are float
        self.w = self.w.astype(float)

        # number of all active nodes
        self.n_nodes = len(self.w)

        # number of edges (in symmetric networks edges are counted twice!)
        self.n_edges = np.sum(self.w != 0)

        # check if network is symmetric (needed e.g. for checking connectedness)
        self.symmetric = check_symmetric(self.w)

        # use fixed density if set
        if density is not None:
            if self.symmetric:
                nedges = int(self.n_nodes * (self.n_nodes - 1) * density // 2)
                id_ = np.argsort(np.triu(self.w, 1), axis=None)
                self.w[np.unravel_index(id_[:-nedges], self.w.shape)] = 0
                self.w = make_symmetric(self.w, copy_lower=False)
            else:
                nedges = int(self.n_nodes * (self.n_nodes - 1) * density)
                id_ = np.argsort(self.w, axis=None)
                self.w[np.unravel_index(id_[:-nedges], self.w.shape)] = 0

        # density of network
        self.density = self.n_edges / (self.n_nodes * (self.n_nodes - 1))

        # indexes of set of active nodes
        self.idx_node = np.full(self.n_nodes, True)

        # make sure that all nodes are connected to the rest of the network
        self.subset_nodes()

        # assign modules
        self.modules = modules

    def scale_and_normalize(self):
        """
        Scale the connectivity matrix between [0, 1] and divide by spectral
        radius
        """

        # scale connectivity matrix between [0, 1]
        self.scale()

        # divide connectivity matrix by spectral radius
        self.normalize()

    def scale(self):
        """
        Scale the connectivity matrix between [0, 1]
        """

        # scale connectivity matrix between [0, 1]
        self.w = (self.w - self.w.min()) / (self.w.max() - self.w.min())

    def normalize(self):
        """
        Normalize the connectivity matrix by spectral radius
        """

        # divide connectivity matrix by spectral radius
        ew, _ = eigh(self.w)
        self.w = self.w / np.abs(ew).max()

    def binarize(self):
        """
        Binarize the connectivity matrix
        """

        # binarize connectivity matrix
        self.w = self.w.astype(bool).astype(float)

    def add_weights(self, w, mask='triu', order='random'):
        """
        Add weights to either a binary or weighted connectivity matrix

        Parameters
        ----------
        w : numpy.ndarray
            the weights to be added to the connectivity matrix
        mask : str, optional
            mask to be used to replace weights in the connectivity matrix, by
            default triu (upper triangular matrix)
        order : str, optional
            it decides whether the weights should be added randomly to the
            connectivity matrix or for instance, the rank of the weights
            should be kept, by default random

        Raises
        ------
        ValueError
            number of elements in mask and w do not match
        ValueError
            symmetric connectivity matrix is needed for this method
        """

        if mask == 'full':
            if w.size != self.n_edges:
                raise ValueError(
                    'number of elements in mask and w do not match')

            # add weights to full matrix
            if order == 'random':
                self.w[self.w != 0] = w

            elif order == 'absrank':  # keep absolute rank of weights
                id_ = np.argsort(np.abs(w))
                w = w[id_[::-1]]
                id_ = np.argsort(np.abs(self.w), axis=None)
                self.w[np.unravel_index(id_[:-w.size-1:-1], self.w.shape)] = w

            elif order == 'rank':  # keep rank of weights
                id_ = np.argsort(w)
                w = w[id_[::-1]]
                id_ = np.argsort(self.w, axis=None)
                self.w[np.unravel_index(id_[:-w.size-1:-1], self.w.shape)] = w

        elif mask == 'triu':
            if not self.symmetric:
                raise ValueError(
                    'add_weight(w, mask=''triu'', order=''random'') needs a symmetric connectivity matrix')
            if w.size != np.sum(np.triu(self.w, 1) != 0):
                raise ValueError(
                    'number of elements in mask and w do not match')

            # add weights to upper diagonal matrix
            if order == 'random':
                self.w[np.triu(self.w, 1) != 0] = w

            elif order == 'absrank':  # keep absolute rank of weights
                id_ = np.argsort(np.abs(w))
                w = w[id_[::-1]]
                id_ = np.argsort(np.abs(np.triu(self.w, 1)), axis=None)
                self.w[np.unravel_index(id_[:-w.size-1:-1], self.w.shape)] = w

            elif order == 'rank':  # keep rank of weights
                id_ = np.argsort(w)
                w = w[id_[::-1]]
                id_ = np.argsort(np.triu(self.w, 1), axis=None)
                self.w[np.unravel_index(id_[:-w.size-1:-1], self.w.shape)] = w

            # copy weights to lower diagonal
            self.w = make_symmetric(self.w, copy_lower=False)

    def subset_nodes(self, node_set='all', idx_node=None, **kwargs):
        """
        Reduce the connectivity matrix to a subset of nodes

        By default, the connectivity matrix is reduced to the largest
        connected component

        Parameters
        ----------
        node_set : str, optional
            subset of nodes defined as string, by default all
        idx_node : numpy.ndarray (dtype=bool), optional
            boolean indexes of nodes to be used for subset of nodes
        """

        # get nodes
        if idx_node is None:
            idx_node = np.isin(np.arange(self.n_nodes),
                               self.get_nodes(node_set, **kwargs))

        # update class attributes
        self._update_attributes(idx_node)

        # update component
        if self.symmetric:
            self._get_largest_component(self.w)
        else:
            self._get_largest_component(np.logical_or(self.w, self.w.T))
            warnings.warn("Asymmetric connectivity matrix is only weakly checked for connectedness.")

    def get_nodes(self, node_set, nodes_from=None, nodes_without=None,
                  filename=None, n_nodes=1, seed=None, **kwargs):
        """
        Get a set of nodes from the connectivity matrix

        Parameters
        ----------
        node_set : str
            subset of nodes defined as string
        nodes_from : numpy.ndarray, optional
            nodes from which the subset should be selected from
        nodes_without : numpy.ndarray, optional
            nodes that should be excluded from the subset
        filename : str, optional
            filename of the data to be loaded from disc specified using full
            path (by default, the data is loaded from examples/data), which
            contains information about the nodes (e.g., which modules they
            belong to or whether they belong to cortex or not)
        n_nodes : int, optional
            number of nodes in the subset, by default 1
        seed : int, array_like[ints], SeedSequence, BitGenerator, Generator, optional
            seed to initialize the random number generator, by default None
            for details, see numpy.random.default_rng()

        Raises
        ------
        ValueError
            number of nodes do not exist with given shortest path
        ValueError
            given node set does not exist in modules
        """

        # initialize fuller set of nodes we want to select from
        if nodes_from is None:
            nodes_from = np.arange(self.n_nodes)

        if node_set == 'all':
            # select all nodes without the ones we do not want to select from
            selected_nodes = np.setdiff1d(nodes_from, nodes_without)

        elif node_set in ['ctx', 'subctx']:
            # load cortex and filter to active nodes
            if filename is not None:
                ctx = np.load(filename)
            else:
                ctx = load_file('cortical.npy')
            ctx = ctx[self.idx_node]

            if node_set == 'ctx':
                # select all nodes in cortex we want to select from
                selected_nodes = np.where(ctx[nodes_from] == 1)[0]
            elif node_set == 'subctx':
                # select all nodes in subcortex we want to select from
                selected_nodes = np.where(ctx[nodes_from] == 0)[0]

            # remove nodes we do not want to select from
            selected_nodes = np.setdiff1d(selected_nodes, nodes_without)

        elif node_set == 'random':
            # nodes we want to select from
            nodes_from = np.setdiff1d(nodes_from, nodes_without)

            # use random number generator for reproducibility
            rng = np.random.default_rng(seed=seed)

            # select random nodes
            selected_nodes = rng.choice(nodes_from, size=n_nodes,
                                        replace=False)

        elif node_set == 'shortest_path':
            # calculate shortest paths between all nodes
            D = distance_bin(self.w)
            D = np.triu(D)  # remove repetitions

            # nodes we want to select from
            nodes_from = np.setdiff1d(nodes_from, nodes_without)

            # shortest paths between all nodes of interest
            D = D[np.ix_(nodes_from, nodes_from)]

            # select all node pairs with requested shortest path from each
            # other
            if isinstance(kwargs['shortest_path'], str):
                node_pairs = np.argwhere(D == np.amax(D))
            elif isinstance(kwargs['shortest_path'], int):
                node_pairs = np.argwhere(D == kwargs['shortest_path'])

            # select requested number of nodes from the set above
            if len(np.unique(node_pairs)) >= n_nodes:
                i = 1
                while len(np.unique(node_pairs[:i, :])) < n_nodes:
                    i += 1
                selected_nodes = nodes_from[np.unique(node_pairs[:i, :])]
            else:
                raise ValueError(
                    'n_nodes do not exist with given shortest_path')

        else:
            # nodes we want to select from
            nodes_from = np.setdiff1d(nodes_from, nodes_without)

            # load resting-state networks and filter to active nodes
            if filename is not None:
                rsn_mapping = np.load(filename)
            else:
                rsn_mapping = load_file('rsn_mapping.npy')
            rsn_mapping = rsn_mapping[self.idx_node]

            # get modules
            module_ids, modules = get_modules(rsn_mapping)

            if node_set in module_ids:
                # select all nodes in the requested module
                selected_nodes = [e for i, e in enumerate(
                    modules) if (module_ids == node_set)[i]][0]

                # intersection of nodes we want to select from
                selected_nodes = np.intersect1d(selected_nodes, nodes_from)
            else:
                raise ValueError('given node_set does not exist in modules')

        return selected_nodes

    def _get_largest_component(self, w):
        """
        Update a set of nodes so that they belong to the largest connected
        component
        """

        # get all components of the connectivity matrix
        comps, comp_sizes = get_components(w)

        # get indexes pertaining to the largest component
        idx_node = comps == np.argmax(comp_sizes) + 1

        # update class attributes
        self._update_attributes(idx_node)

    def _update_attributes(self, idx_node):
        """
        Update network attributes

        Parameters
        ----------
        idx_node : numpy.ndarray (dtype=bool)
            boolean indexes of nodes which we want to changes the attributes of

        Raises
        ------
        ValueError
            boolean indexing should be used for nodes
        """

        if isinstance(idx_node, np.ndarray) and idx_node.dtype == np.bool:
            # update node attributes
            self.n_nodes = sum(idx_node)
            self.idx_node[self.idx_node] = idx_node

            # update edge attributes
            self.w = self.w[np.ix_(idx_node, idx_node)]
            self.n_edges = np.sum(self.w != 0)

            # update density
            self.density = self.n_edges / (self.n_nodes * (self.n_nodes - 1))
        else:
            raise ValueError('Boolean indexing should be used for nodes')


def load_file(filename):
    """
    Load data from disc

    Parameters
    ----------
    filename : str, optional
        filename of the data to be loaded from DATA_DIR in disc

    Returns
    -------
    result : np.ndarray
        data stored in the file
    """
    return np.load(os.path.join(DATA_DIR, filename))


def get_modules(module_assignment):
    """
    Get module assignment of nodes

    Parameters
    ----------
    module_assignment : np.ndarray
        array of modules the nodes belong to

    Returns
    -------
    module_ids : np.ndarray
        array of unique modules
    readout_modules : np.ndarray
        indexes of unique modules the nodes belong to
    """
    # get module ids
    module_ids = np.unique(module_assignment)
    readout_modules = [np.where(module_assignment == i)[0] for i in module_ids]

    return module_ids, readout_modules


def get_readout_nodes(readout_modules):
    """
    Return a list with the set(s) of nodes in each module in
    'readout_modules', plus a set of module ids

    Parameters
    ----------
    readout_modules : (N,) list, tuple, numpy.ndarray or dict
        Can be a 1D array-like that assigns modules to each node. Can
        be a list of lists, where each sublist corresponds to the
        indexes of subsets of nodes. Can be a dictionary key:val pairs,
        where the keys correspond to modules and the values correspond
        to list/tuple that contains the subset of nodes in each module.

    Returns
    -------
    readout_nodes : list
        list that contains lists with indexes of subsets of nodes in
        'readout_modules'
    ids : list
        list that contains lists with indexes of subsets of nodes in
        'readout_modules'

    Raises
    ------
    TypeError
        _description_
    """
    if isinstance(readout_modules, (list, tuple, np.ndarray)):
        if all(isinstance(i, (list, tuple, np.ndarray)) for i in readout_modules):
            ids = list(range(len(readout_modules)))
            readout_nodes = list(module for module in readout_modules)
        else:
            ids = list(set(readout_modules))
            readout_nodes = list(
                np.where(np.array(readout_modules) == i)[0] for i in ids
            )
    elif isinstance(readout_modules, dict):
        ids = list(readout_modules.keys())
        readout_nodes = list(readout_modules.values())
    else:
        raise TypeError("")

    return readout_nodes, ids
