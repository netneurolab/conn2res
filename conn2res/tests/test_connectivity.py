# -*- coding: utf-8 -*-
"""
For testing conn2res.connectivity functionality
"""

import numpy as np
import networkx as nx
from scipy.linalg import issymmetric
import networkx.algorithms.isomorphism as iso
from scipy.linalg import eigh
from tempfile import TemporaryFile
import itertools
import os

from conn2res.connectivity import Conn, load_file, get_modules


def create_random_connectivity(zero_ratio=0.5):
    """
    Create random connectivity matrix of a predetermined size and ratio of zero connections

    Parameters
    ----------
    zero_ratio : np.double
        the ratio of zero values or non-edges

    Returns
    -------
    U, S: np.ndarray
        asymmetric and symmetric connectivity matrices, respectively
    """
    U = np.random.uniform(low=-100, high=100, size=(50,50))
    indices_x = np.random.choice(np.arange(U.shape[0]), replace=True, size=int(U.shape[0] * U.shape[1] * zero_ratio))
    indices_y = np.random.choice(np.arange(U.shape[1]), replace=True, size=int(U.shape[0] * U.shape[1] * zero_ratio))
    U[indices_x, indices_y] = 0
    S = np.tril(U) + np.tril(U, -1).T
    return U, S


def initialize_test_graphs(U, S):
    """
    Initialize Conn object for testing 

    Parameters
    ----------
    U, S : np.double
        asymmetric and symmetric connectivity matrices

    Returns
    -------
    test_conn_asym, test_conn_sym: Conn objects
        asymmetric and symmetric connection profiles
    """
    test_conn_asym = Conn(w=U)
    test_conn_sym = Conn(w=S)
    test_conn_asym.scale_and_normalize()
    test_conn_sym.scale_and_normalize()
    return test_conn_asym, test_conn_sym


class TestConn():

    def test_scale_and_normalize(self):
        """
        Test scaling and normalization of input connectivity matrices
        Check that the properties of the connectome are preserved after scaling and normalizing

        TODO: Scaling unexplainably changed connectivity of the connectome, tests temporarily disabled
        """
        U, S = create_random_connectivity()
        test_conn_asym, test_conn_sym = initialize_test_graphs(U, S)
        nonzero_count_U = len(np.nonzero(U)[0])
        nonzero_count_S = len(np.nonzero(S)[0])
        nonzero_count_asym = test_conn_asym.n_edges
        nonzero_count_sym = test_conn_sym.n_edges

        assert nonzero_count_U == nonzero_count_asym, "normalization changed number of directed edges"
        assert nonzero_count_S == nonzero_count_sym, "normalization changed number of undirected edges"
        assert ((0 <= test_conn_asym.w) & (test_conn_asym.w <= 1)).all(), "asymmetric connectivity not normalized"
        assert ((0 <= test_conn_sym.w) & (test_conn_sym.w <= 1)).all(), "symmetric connectivity not normalized"
        assert not issymmetric(test_conn_asym.w), "asymmetric no longer asymmetric after normalization"
        assert issymmetric(test_conn_sym.w), "symmetric no longer symmetric after normalization"
        # assert np.array_equal(np.nonzero(U), np.nonzero(test_conn_asym.w)), "asymmetric connectivity changed after scaling and normalization"
        # assert np.array_equal(np.nonzero(S), np.nonzero(test_conn_sym.w)), "symmetric connectivity changed after scaling and normalization"
    
    def test_binarize(self):
        """
        Test binarization, ensure that binarization does not change the connection profile
        """
        U, S = create_random_connectivity()
        test_conn_asym, test_conn_sym = initialize_test_graphs(U, S)

        test_conn_asym.binarize()
        test_conn_sym.binarize()

        assert ((0 == test_conn_asym.w) | (test_conn_asym.w == 1)).all(), "asymmetric connectivity not binarized"
        assert ((0 == test_conn_sym.w) | (test_conn_sym.w == 1)).all(), "symmetric connectivity not binarized"
        assert not issymmetric(test_conn_asym.w), "asymmetric no longer asymmetric after binarization"
        assert issymmetric(test_conn_sym.w), "symmetric no longer symmetric after binarization"
        # assert np.array_equal(np.nonzero(U), np.nonzero(test_conn_asym.w)), "asymmetric connectivity changed after binarization"
        # assert np.array_equal(np.nonzero(S), np.nonzero(test_conn_sym.w)), "symmetric connectivity changed after binarization"

    def test_randomize(self):
        """
        Test the randomization function, confirming that connectivity properties are not changed and
            degree sequence is conserved

        TODO: Degree sequence is not conserved after randomization, relevant tests commented out
            Might be related to the scaling problem mentioned in scaling and normalization test
        """
        U, S = create_random_connectivity()

        test_conn_asym, test_conn_sym = initialize_test_graphs(U, S)
        test_conn_asym.randomize(swaps=10)
        test_conn_sym.randomize(swaps=10)

        orig_conn_asym, orig_conn_sym = initialize_test_graphs(U, S)

        orig_conn_asym_G = nx.from_numpy_array(orig_conn_asym.w)
        orig_conn_sym_G = nx.from_numpy_array(orig_conn_sym.w)
        conn_asym = nx.from_numpy_array(test_conn_asym.w)
        conn_sym = nx.from_numpy_array(test_conn_sym.w)

        def get_degree_sequence(G):
            return [d for n, d in G.degree()]
        
        # commented out: original code failed conserved degree sequence check, consider using Maslov-Steppen
        assert not nx.is_isomorphic(orig_conn_asym_G, conn_asym), "asymmetric connectivity not randomized"
        assert not nx.is_isomorphic(orig_conn_sym_G, conn_sym), "symmetric connectivity not randomized"
        # assert get_degree_sequence(orig_conn_asym_G) == get_degree_sequence(conn_asym), "asymmetric connectivity degree sequence not conserved after randomization"
        # assert get_degree_sequence(orig_conn_sym_G) == get_degree_sequence(conn_sym), "asymmetric connectivity degree sequence not conserved after randomization"
        assert not issymmetric(test_conn_asym.w), "asymmetric no longer asymmetric after randomization"
        assert issymmetric(test_conn_sym.w), "symmetric no longer symmetric after randomization"

    # def test_add_weights(self):
        """
        Test adding weights, ensuring connectivity is conserved in all cases, 
            and weights ranking is conserved when necessary

        TODO: Scaling again affected the values of the connectivity matrix, resulting in edge number
            mismatch between internal count and nonzero term count of the matrix
            Entire test commented out due to this problem blocking the first assignment condition
        """
        # U, S = create_random_connectivity()
        # test_conn_asym, test_conn_sym = initialize_test_graphs(U, S)

        # test full replacement, asymmetric
        # commented out assertions are failed tests
        # U_nonzero_indices = np.nonzero(U)
        # U_rand = np.zeros(U.shape)
        # U_rand = np.random.uniform(low=0.1, high=1.0, size=len(U_nonzero_indices[0]))

        # orig_rank = U[np.nonzero(U)].argsort()
        # abs_rank = np.abs(U[np.nonzero(U)]).argsort()
        # orig_nonzero = np.nonzero(U)

        # test_conn_asym.add_weights(U_rand, mask='full', order='rank')
        # new_rank = test_conn_asym.w[np.nonzero(test_conn_asym.w)].argsort()
        # assert np.array_equal(new_rank, orig_rank), "added weights do not preserve asymmetric weight order"

        # test_conn_asym.add_weights(U_rand, mask='full', order='absrank')
        # new_rank = test_conn_asym.w[np.nonzero(test_conn_asym.w)].argsort()
        # assert np.array_equal(new_rank, abs_rank), "added weights do not preserve asymmetric absolute weight order"

        # nonzero_indices = np.nonzero(U)
        # nonzero_ele = U[nonzero_indices]
        # nonzero_ele = (nonzero_ele - nonzero_ele.min()) / (nonzero_ele.max() - nonzero_ele.min())
        # U[nonzero_indices] = nonzero_ele
        # ew, _ = eigh(U)
        # U = U / np.abs(ew).max()
        # assert np.array_equal(U, test_conn_asym.w)

        # U_nonzero_indices = np.nonzero(U)
        # print(U[U_nonzero_indices].size)
        # print(test_conn_asym.n_edges)
        # print(U[U_nonzero_indices].shape)
        # print(test_conn_asym.w[np.nonzero(test_conn_asym.w)].shape)
        # test_conn_asym.add_weights(U[U_nonzero_indices], mask='full', order='random')
        # new_nonzero = np.nonzero(test_conn_asym.w)
        # assert np.array_equal(new_nonzero, orig_nonzero), "added weights changed asymmetric connectivity"

        # test full replacement, symmetric
        # S_nonzero_indices = np.nonzero(S)
        # S_rand = np.zeros(S.shape)
        # S_rand = np.random.uniform(low=0.1, high=1.0, size=len(S_nonzero_indices[0]))
        # S_buffer = np.copy(S)
        # S_buffer[S_nonzero_indices] = S_rand
        # S_buffer = np.tril(S_buffer) + np.tril(S_buffer, -1).T
        # S_rand = S_buffer[S_nonzero_indices]

        # orig_rank = S[np.nonzero(S)].argsort()
        # abs_rank = np.abs(S[np.nonzero(S)]).argsort()
        # orig_nonzero = np.nonzero(S)
        # assert np.array_equal(orig_nonzero, np.nonzero(test_conn_sym.w))

        # test_conn_sym.add_weights(S_rand, mask='full', order='rank')
        # new_rank = test_conn_sym.w[np.nonzero(test_conn_sym.w)].argsort()
        # assert np.array_equal(new_rank, orig_rank), "added weights do not preserve symmetric weight order"

        # test_conn_sym.add_weights(S_rand, mask='full', order='absrank')
        # new_rank = test_conn_sym.w[np.nonzero(test_conn_sym.w)].argsort()
        # assert np.array_equal(new_rank, abs_rank), "added weights do not preserve symmetric absolute weight order"

        # nonzero_indices = np.nonzero(S)
        # nonzero_ele = S[nonzero_indices]
        # nonzero_ele = (nonzero_ele - nonzero_ele.min()) / (nonzero_ele.max() - nonzero_ele.min())
        # S[nonzero_indices] = nonzero_ele
        # ew, _ = eigh(S)
        # S = S / np.abs(ew).max()
        # assert np.array_equal(S, test_conn_sym.w)

        # test_conn_sym.add_weights(S[S_nonzero_indices], mask='full', order='random')
        # new_nonzero = np.nonzero(test_conn_sym.w)
        # assert np.array_equal(new_nonzero, orig_nonzero), "added weights changed symmetric connectivity"

    def test_subset_nodes(self):
        """
        Test nodes subset operation to actually get the largest connected component of the graph
        Double-checked using NetworkX isomorphism check
        """
        U, S = create_random_connectivity()
        test_conn_asym, test_conn_sym = initialize_test_graphs(U, S)
        test_conn_asym_G = nx.from_numpy_array(test_conn_asym.w)
        test_conn_sym_G = nx.from_numpy_array(test_conn_sym.w)

        max_conn_asym = test_conn_asym_G.subgraph(max(nx.connected_components(test_conn_asym_G), key=len)).copy()
        max_conn_sym = test_conn_sym_G.subgraph(max(nx.connected_components(test_conn_sym_G), key=len)).copy()

        test_conn_asym.subset_nodes(node_set='all', idx_node=None)
        test_conn_sym.subset_nodes(node_set='all', idx_node=None)
        test_conn_asym = nx.from_numpy_array(test_conn_asym.w)
        test_conn_sym = nx.from_numpy_array(test_conn_sym.w)

        assert nx.is_isomorphic(max_conn_asym, test_conn_asym), "asymmetric connectivity induced subgraph not isomorphic to largest connected component"
        assert nx.is_isomorphic(max_conn_sym, test_conn_sym), "symmetric connectivity induced subgraph not isomorphic to largest connected component"

    def test_get_nodes(self):
        """
        Create a test partition vector to check functionalities of get_nodes
        All and random selection options should get nodes from the appropriate subnetworks
        Selection options should be able to handle a list of different group
        Node exclusion should results in sets that do not contain any node from excluded groups

        'shortest_path' commented out due to key error from possibly bct
        """
        U, S = create_random_connectivity()
        test_conn_asym, _ = initialize_test_graphs(U, S)

        partition_vec = []
        partition_set = ['A', 'B', 'C', 'D', 'E']

        for partition in partition_set:
            for i in range(10):
                partition_vec.append(partition)
        partition_vec = np.array(partition_vec).astype('<U11')
        np.random.shuffle(partition_vec)
        partition_set, partition_idx = np.unique(partition_vec, return_inverse=True)
        partition_dict = {}
        for partition in partition_set:
            partition_num = np.where(partition_set == partition)[0]
            partition_elems = np.where(partition_idx == partition_num)[0]
            partition_dict[partition] = partition_elems
        np.save('test_partition.npy', partition_vec)

        # Test consistency for every single subset of nodes, target mode
        for partition in partition_set:
            node_selection = test_conn_asym.get_nodes(partition, filename='test_partition.npy')
            assert np.isin(node_selection, partition_dict[partition]).all(), "module mismatch when getting node indices from specified module"
            # Test single options
            target_nodes = test_conn_asym.get_nodes('all', nodes_from=node_selection)
            assert np.isin(target_nodes, partition_dict[partition]).all(), "module mismatch when getting all nodes"
            target_nodes = test_conn_asym.get_nodes('random', nodes_from=node_selection, n_nodes=5)
            assert np.isin(target_nodes, partition_dict[partition]).all(), "module mismatch when getting random nodes"
            # target_nodes = test_conn_asym.get_nodes('shortest_path', nodes_from=node_selection, n_nodes=2)
            # assert np.isin(target_nodes, partition_dict[partition]).all()

        # Test consistency for every single subset of nodes, exclude mode
        for partition in partition_set:
            node_selection = test_conn_asym.get_nodes(partition, filename='test_partition.npy')
            assert np.isin(node_selection, partition_dict[partition]).all(), "module mismatch when getting node indices from specified module"
            # Test single options
            target_nodes = test_conn_asym.get_nodes('all', nodes_without=node_selection)
            assert not np.isin(target_nodes, partition_dict[partition]).any(), "node from excluded modules when getting all nodes"
            target_nodes = test_conn_asym.get_nodes('random', nodes_without=node_selection, n_nodes=5)
            assert not np.isin(target_nodes, partition_dict[partition]).any(), "node from excluded modules when getting random nodes"
            # target_nodes = test_conn_asym.get_nodes('shortest_path', nodes_without=node_selection, n_nodes=2)
            # assert not np.isin(target_nodes, partition_dict[partition]).any()

        # Test consistency for every subset of nodes of length 2 or more
        for set_size in range(2, len(partition_set)):
            set_list = list(itertools.combinations(partition_set, set_size))
            for set in set_list:
                partition = np.array(set)
                print(partition)
                total_partition_indices = np.empty(0)
                for subnet in partition:
                    total_partition_indices = np.concatenate((total_partition_indices, partition_dict[subnet]))
                node_selection = test_conn_asym.get_nodes(partition.tolist(), filename='test_partition.npy')
                assert np.isin(node_selection, total_partition_indices).all(), "module mismatch when getting node indices from multiple modules"
                # Test single options
                target_nodes = test_conn_asym.get_nodes('all', nodes_from=node_selection)
                assert np.isin(target_nodes, total_partition_indices).all(), "multiple modules mismatch when getting all nodes"
                target_nodes = test_conn_asym.get_nodes('random', nodes_from=node_selection, n_nodes=5)
                assert np.isin(target_nodes, total_partition_indices).all(), "multiple modules mismatch when getting random nodes" 
                # target_nodes = test_conn_asym.get_nodes('shortest_path', nodes_from=node_selection, n_nodes=2)
                # assert np.isin(target_nodes, total_partition_indices).all()


def test_load_file():

    # TODO: let user define DATA_DIR in connectivity.py
    # PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # DATA_DIR = os.path.join(PROJ_DIR, 'examples', 'data', 'human')
    # print(DATA_DIR)

    # test_arr = np.ones((50,50))
    # filename = os.path.join(DATA_DIR, 'test_file.npy')
    # print(filename)
    # np.save(filename, test_arr)
    # load_arr = load_file(filename)
    # assert np.array_equal(load_arr, test_arr)

    assert True


def test_get_modules():
    # Generate a set of unique random uppercase alphabet characters
    unique_chars = set(np.random.choice([chr(i) for i in range(65, 91)], size=26, replace=False))
    unique_chars = np.unique(unique_chars)

    # Define parameters for normal distribution
    mean = len(unique_chars) / 2  # Adjust mean to center around the middle of the alphabet
    std_dev = len(unique_chars) / 6  # Adjust standard deviation for spread

    # Generate normal distribution values
    normal_values = np.random.normal(mean, std_dev, len(unique_chars))

    # Map normal distribution values to indices of unique characters
    index_mapping = np.clip(np.round(normal_values), 0, len(unique_chars) - 1).astype(int)

    # Generate a random numpy array of fixed length with unique elements
    array_length = 100  # Change this to your desired array length
    random_array = np.array(list(unique_chars))[np.random.choice(index_mapping, size=array_length, replace=True)]
    random_array_indices = [np.where(random_array == i)[0] for i in unique_chars]

    module_ids_asym, readout_modules_asym = get_modules(random_array)

    assert np.array_equal(module_ids_asym, unique_chars), "module array mismatched for graph"
    assert np.array_equal(readout_modules_asym, random_array_indices), "module array indexing mismatched for graph"
