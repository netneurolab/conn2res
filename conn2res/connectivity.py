"""
Functionality for connectivity matrix
"""
import os
import numpy as np
from scipy.linalg import eigh
from bct.algorithms.clustering import get_components
from bct.algorithms.distance import distance_bin

from .utils import *


PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_DIR, 'examples', 'data')


class Conn:
    """
    Class that represents a connectivity matrix representing either weighted
    or unweighted connectivity data

    Attributes
    ----------
    # TODO

    Methods
    ----------
    # TODO
    """

    def __init__(self, filename=None, subj_id=0, w=None, modules=None):
        if w is not None:
            # assign provided connectivity data
            self.w = w
        else:
            # load connectivity data
            if filename is not None:
                self.w = np.load(filename)
            else:
                self.w = load_file('connectivity.npy')

            # select one subject
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

        # density of network
        self.density = self.n_edges / (self.n_nodes * (self.n_nodes - 1))

        # indexes of set of active nodes
        self.idx_node = np.full(self.n_nodes, True)

        # make sure that all nodes are connected to the rest of the network
        self.subset_nodes(idx_node=np.logical_or(
            np.any(self.w != 0, axis=0), np.any(self.w != 0, axis=1)))

        self.symmetric = check_symmetric(self.w)

        self.modules = modules

    def scale_and_normalize(self):
        """
        Scales the connectivity matrix between [0, 1] and divides by spectral
        radius

        # TODO
        """

        # scale connectivity matrix between [0, 1]
        self.scale()

        # divide connectivity matrix by spectral radius
        self.normalize()

    def scale(self):
        """
        Scales the connectivity matrix between [0, 1]

        # TODO
        """

        # scale connectivity matrix between [0, 1]
        self.w = (self.w - self.w.min()) / (self.w.max() - self.w.min())

    def normalize(self):
        """
        Normalizes the connectivity matrix with spectral radius

        # TODO
        """

        # divide connectivity matrix by spectral radius
        ew, _ = eigh(self.w)
        self.w = self.w / np.abs(ew).max()

    def binarize(self):
        """
        Binarizes the connectivity matrix

        # TODO
        """

        # binarize connectivity matrix
        self.w = self.w.astype(bool).astype(float)

    def add_weights(self, w, mask='triu', order='random'):
        """
        Add weights to either a binary or weighted connecivity matrix

        # TODO
        """

        if mask == 'full':
            if w.size != np.sum(self.w == 1):
                raise ValueError(
                    'number of elements in mask and w do not match')

            # add weights to full matrix
            if order == 'random':
                self.w[self.w != 0] = w

        elif mask == 'triu':
            if not check_symmetric(self.w):
                raise ValueError(
                    'add_weight(w, mask=''triu'') needs a symmetric connectivity matrix')
            if w.size != np.sum(np.triu(self.w, 1) != 0):
                raise ValueError(
                    'number of elements in mask and w do not match')

            # add weights to upper diagonal matrix
            if order == 'random':
                self.w[np.triu(self.w, 1) != 0] = w

            elif order == 'rank':  # keep absolute rank of weights
                id_ = np.argsort(np.abs(w))
                w = w[id_[::-1]]
                id_ = np.argsort(np.abs(np.triu(self.w, 1)), axis=None)
                self.w[np.unravel_index(id_[:-w.size-1:-1], self.w.shape)] = w

            # copy weights to lower diagonal
            self.w = make_symmetric(self.w, copy_lower=False)

    def subset_nodes(self, node_set='all', idx_node=None, **kwargs):
        """
        Defines subset of nodes of the connectivity matrix and reduces
        the connectivity matrix to this subset

        # TODO
        """

        # get nodes
        if idx_node is None:
            idx_node = np.isin(np.arange(self.n_nodes),
                               self.get_nodes(node_set, **kwargs))

        # update class attributes
        self._update_attributes(idx_node)

        # update component
        self._get_largest_component()

    def get_nodes(self, node_set, nodes_from=None, nodes_without=None, filename=None, n_nodes=1, **kwargs):
        """
        Gets a set of nodes of the connectivity matrix without changing 
        the connectivity matrix itself

        # TODO
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

            # select random nodes
            selected_nodes = np.random.choice(nodes_from, size=n_nodes,
                                              replace=False)

        elif node_set == 'shortest_path':
            # calculate shortest paths between all nodes
            D = distance_bin(self.w)
            D = np.triu(D)  # remove repetitions

            # nodes we want to select from
            nodes_from = np.setdiff1d(nodes_from, nodes_without)

            # shortest paths between all nodes of interest
            D = D[np.ix_(nodes_from, nodes_from)]

            # select all node pairs with requested shortest path from each other
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
                raise ValueError('node_set does not exist with given value')

        return selected_nodes

    def _get_largest_component(self):
        """
        Updates a set of nodes so that they belong to one connected component

        #TODO
        """

        # get all components of the connectivity matrix
        comps, comp_sizes = get_components(self.w)

        # get indexes pertaining to the largest component
        idx_node = comps == np.argmax(comp_sizes) + 1

        # update class attributes
        self._update_attributes(idx_node)

    def _update_attributes(self, idx_node):
        """
        Updates network attributes

        #TODO
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
            raise NotImplementedError


def load_file(filename):
    """
    #TODO
    _summary_

    Parameters
    ----------
    filename : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return np.load(os.path.join(DATA_DIR, filename))


def get_modules(module_assignment):
    """
    # TODO
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
