# -*- coding: utf-8 -*-
"""
Reservoir classes

@author: Estefany Suarez
"""

import itertools as itr
import numpy as np
import numpy.ma as ma
from numpy.linalg import (pinv, matrix_rank)

import matplotlib.pyplot as plt

class Reservoir:
    """
    Class that represents a general Reservoir object

    ...

    Attributes
    ----------
    w_ih : numpy.ndarray
        input connectivity matrix (source, target)
    w_hh : numpy.ndarray
        reservoir connectivity matrix (source, target)
    _state : numpy.ndarray
        reservoir activation states
    input_size : int
        dimension of feature space
    hidden_size : int
        dimension of the reservoir

    Methods
    ----------

    """
    def __init__(self, w_ih, w_hh):
        """
        Constructor class for general Reservoir Networks

        Parameters
        ----------
        w_ih: (N_inputs, N) numpy.ndarray
            input connectivity matrix (source, target)
            N_inputs: number of external input nodes
            N: number of nodes in the network
        w_hh : (N, N) numpy.ndarray
            reservoir connectivity matrix (source, target)
            N: number of nodes in the network. If w_hh is directed, then rows
            (columns) should correspond to source (target) nodes.
        """

        self.w_ih = w_ih
        self.w_hh = w_hh
        self._state = None
        self.input_size, self.hidden_size = w_ih.shape


class EchoStateNetwork(Reservoir):
    """
    Class that represents an Echo State Network

    ...

    Attributes
    ----------
    w_ih : numpy.ndarray
        input connectivity matrix (source, target)
    w_hh : numpy.ndarray
        reservoir connectivity matrix (source, target)
    _state : numpy.ndarray
        reservoir activation states
    input_size : int
        dimension of feature space
    hidden_size : int
        dimension of the reservoir
    activation_function : {'tanh', 'piecewise'}
        type of activation function

    Methods
    -------

    """

    def __init__(self, activation_function='tanh', *args, **kwargs):
        """
        Constructor class for Echo State Networks

        Parameters
        ----------
        w_ih: (N_inputs, N) numpy.ndarray
            Input connectivity matrix (source, target)
            N_inputs: number of external input nodes
            N: number of nodes in the network
        w_hh : (N, N) numpy.ndarray
            Reservoir connectivity matrix (source, target)
            N: number of nodes in the network. If w_hh is directed, then rows
            (columns) should correspond to source (target) nodes.
        nonlinearity : str {'tanh', 'relu'}, default 'tanh'
            Activation function
        """
        super().__init__(*args, **kwargs)
        self.activation_function = self.set_activation_function(activation_function)


    def simulate(self, ext_input, ic=None, threshold=0.5):
        """
        Simulates reservoir dynamics given an external input signal
        'ext_input'

        Parameters
        ----------
        ext_input: (time, N_inputs) numpy.ndarray
            External input signal
            N_inputs: number of external input nodes
        ic: (N,) numpy.ndarray
            Initial conditions
            N: number of nodes in the network. If w_hh is directed, then rows
            (columns) should correspond to source (target) nodes.
        threshold : float
            Threshold for piecewise nonlinearity. Ignored for the others.

        Returns
        -------
        self._state : numpy.ndarray
            activation states of the reservoir; includes all the nodes
        """

        print('\n GENERATING RESERVOIR STATES ...')

        # check data type for ext_input. If list convert to numpy.ndarray
        if isinstance(ext_input, list): ext_input = np.asarray(ext_input)

        # initialize reservoir states
        timesteps = range(1, len(ext_input))
        self._state = np.zeros((len(timesteps)+1, self.hidden_size))

        # set initial conditions
        if ic is not None: self._state[0,:] = ic

        # simulation of the dynamics
        for t in timesteps:
           
            if (t>0) and (t%100 == 0): print(f'\t ----- timestep = {t}')
            
            synap_input = np.dot(self._state[t-1,:], self.w_hh) + np.dot(ext_input[t-1,:], self.w_ih)
            self._state[t,:] = self.activation_function(synap_input)

        return self._state


    def set_activation_function(self, function):
        
        def linear(x, m=1):
            return m * x

        def elu(x, alpha=0.5):
            x[x <= 0] = alpha*(np.exp(x[x <= 0]) -1)
            return x 

        def relu(x):
            return np.maximum(0, x)
        
        def leaky_relu(x, alpha=0.5):
            return np.maximum(alpha * x, x)

        def sigmoid(x):
            return 1.0 / (1 + np.exp(-x))
        
        def tanh(x):
            return np.tanh(x)
        
        def step(x, thr=0.5, vmin=0, vmax=1):
            return np.piecewise(x, [x<thr, x>=thr], [vmin, vmax]).astype(int)

        if function == 'linear':
            return linear
        elif function == 'elu':
            return elu
        elif function == 'relu':
            return relu
        elif function == 'leaky_relu':
            return leaky_relu       
        elif function == 'sigmoid':
            return sigmoid
        elif function == 'tanh':
            return tanh
        elif function == 'step':
            return step


class MemristiveReservoir:
    """
    Class that represents a general Memristive Reservoir

    ...

    Attributes
    ----------
    W : numpy.ndarray
        reservoir's binary connectivity matrix
    I : numpy.ndarray
        set of internal nodes
    E : numpy.ndarray
        set of external nodes
    GR : numpy.ndarray
        set of grounded nodes
    n_internal_nodes : int
        number of internal nodes
    n_external_nodes : int
        number of external nodes
    n_grounded_nodes : int
        number of gorunded nodes
    G : numpy.ndarray
        matrix of conductances

    Methods
    ----------
    #TODO

    References
    ----------
    #TODO

    """
    def __init__(self, w, i_nodes, e_nodes, gr_nodes, *args, **kwargs):
        """
        Constructor class for Memristive Networks. Memristive networks are an
        abstraction for physical networks of memristive elements.

        Parameters
        ----------
        w : (N, N) numpy.ndarray
            reservoir's binary connectivity matrix
            N: total number of nodes in the network (internal + external
            + grounded nodes)
        i_nodes : (n_internal_nodes,) numpy.ndarray
            indexes of internal nodes
            n_internal_nodes: number of internal nodes
        e_nodes : (n_external_nodes,) numpy.ndarray
            indexes of external nodes
            n_external_nodes: number of external nodes
        gr_nodes : (n_grounded_nodes,) numpy.ndarray
            indexes of grounded nodes
            n_grounded_nodes: number of grounded nodes
        """
        super().__init__(*args, **kwargs)
        self._W  = self.setW(w)
        self._I  = np.asarray(i_nodes)
        self._E  = np.asarray(e_nodes)
        self._GR = np.asarray(gr_nodes)

        self._n_internal_nodes = len(self._I)
        self._n_external_nodes = len(self._E)
        self._n_grounded_nodes = len(self._GR)
        self._n_nodes = len(self._W)

        self._G = None

    
    def setW(self, w):
        """
        #TODO
        This function guarantees that W is binary and symmetric. Converts
        directed connectivity matrices in undirected.

        Parameters
        ----------

        """

        # convert to binary
        w = w.astype(bool).astype(int)

        # make sure the diagonal is zero
        np.fill_diagonal(w, 0)

        # make symmetric if w is directed
        if not check_symmetric(w):

            # connections in upper diagonal
            upper_diag = w[np.triu_indices_from(w, 1)]

            # connections in lower diagonal
            lower_diag = w.T[np.triu_indices_from(w, 1)]

            # matrix of undirected connections
            W = np.zeros_like(w).astype(int)
            W[np.triu_indices_from(w,1)] = np.logical_or(upper_diag,
                                                         lower_diag
                                                        ).astype(int)

            return make_symmetric(W, copy_lower=False)

        else:
            return w

    
    def init_property(self, mean, std=0.1):
        """
        This function initializes property matrices following a normal
        distribution with mean = 'mean' and standard deviation = 'mean' * 'std'

        Parameters
        ----------
        #TODO

        Returns
        -------
        #TODO

        """

        p = np.random.normal(mean, std*mean, size=self._W.shape)
        p = make_symmetric(p)

        return p * self._W # ma.masked_array(p, mask=np.logical_not(self._W))

    
    def solveVi(self, Ve, Vgr=None, G=None, **kwargs):
        """
        This function uses Kirchhoff's law to estimate voltage at the internal
        nodes based on the current state of the conductance matrix G, a given
        external input voltage 'V_E' and the ground level voltage 'V_GR'

        Parameters
        ----------
        #TODO

        Returns
        -------
        #TODO

        """

        if Vgr is None: Vgr = np.zeros((self._n_grounded_nodes))
        if G is None: G = self._G

        # TODO: verify that the axis along which the sum is performed is correct
        # matrix N
        N = np.zeros((self._n_nodes, self._n_nodes))
        np.fill_diagonal(N, np.sum(G, axis=1))

        # matrix A
        A = N - G

        # inverse matrix A_II
        A_II = A[np.ix_(self._I, self._I)]
        # print(matrix_rank(A_II, hermitian=check_symmetric(A_II)))
        A_II_inv = pinv(A_II)

        # matrix HI
        H_IE  = np.dot(G[np.ix_(self._I, self._E)], Ve)
        H_IGR = np.dot(G[np.ix_(self._I, self._GR)], Vgr)

        H_I = H_IE + H_IGR

        # return voltage at internal nodes
        return np.dot(A_II_inv, H_I)

    
    def getV(self, Vi, Ve, Vgr=None):
        """
        Given the nodal voltage at the internal, external and grounded
        nodes, this function estimates the voltage across all existent
        memristive elements

        Parameters
        ----------
        #TODO

        Returns
        -------
        #TODO

        """

        # set voltage at grounded nodes
        if Vgr is None: Vgr = np.zeros((self._n_grounded_nodes))

        # set of all nodal voltages
        voltage = np.concatenate([Vi[:, np.newaxis], Ve[:, np.newaxis], Vgr[:, np.newaxis]]).squeeze()

        # set of all nodes (internal + external + grounded)
        nodes   = np.concatenate([self._I[:, np.newaxis], self._E[:, np.newaxis], self._GR[:, np.newaxis]]).squeeze()

        # dictionary that groups pairs of voltage
        nv_dict = {n:v for n,v in zip(nodes,voltage)}

        # voltage across memristors
        V = np.zeros_like(self._W).astype(float)
        for i,j in list(zip(*np.where(self._W != 0))):
            if j > i:
                V[i,j] = nv_dict[i] - nv_dict[j]
            else:
                V[i,j] = nv_dict[j] - nv_dict[i]

        return mask(self, V) 

    
    def simulate(self, Vext, ic=None, mode='forward'):
        """
        Simulates the dynamics of a memristive reservoir given an external
        voltage signal V_E

        Parameters
        ----------
        Vext : (time, N_external_nodes) numpy.ndarray
            External input signal
            N_external_nodes: number of external (input) nodes
        ic : (N_internal_nodes,) numpy.ndarray
            Initial conditions
            N_internal_nodes: number of internal (output) nodes
        mode : {'forward', 'backward'}
            Refers to the method used to solve the system of equations. 
            Use 'forward' for explicit Euler method, and 'backward' for 
            implicit Euler method.

        #TODO
        """

        print('\n GENERATING RESERVOIR STATES ...')

        Vi = []
        if mode == 'forward':
            for t, Ve in enumerate(Vext):

                if (t>0) and (t%100 == 0): print(f'\t ----- timestep = {t}')

                # get voltage at internal nodes
                Vi.append(self.solveVi(Ve))

                # update matrix of voltages across memristors
                V = self.getV(Vi[-1], Ve)

                # update conductance
                self.updateG(V=V, update=True)
        
        elif mode == 'backward':
            for t,Ve in enumerate(Vext):
                
                if (t>0) and (t%100 == 0): print(f'\t ----- timestep = {t}')

                # get voltage at internal nodes
                Vi.append(self.iterate(Ve))
        
        V = np.zeros((len(Vi), self._n_nodes))
        V[:, self._I] = np.asarray(Vi)
        V[:, self._E] = Vext
        return V


    def iterate(self, Ve, tol=5e-2, iters=100): 
        """
        #TODO

        """

        # initial guess for voltage at internal nodes 
        Vi = [self.solveVi(Ve=Ve, G=self._G)]

        # initial guess for conductance
        G  = [self._G] 

        convergence = False
        n_iters = 0
        while not convergence: 

            assert n_iters < iters, 'There is no convergence !!!' 
                
            # get voltage across memristors 
            # and update conductance
            V = self.getV(Vi[-1].copy(), Ve)
            G_tmp = self.updateG(V, G[-1], update=False)

            # solve voltage at internal nodes Vi
            Vi.append(self.solveVi(Ve=Ve, G=G_tmp))

            # update conductance with G_tmp
            G.append(self.updateG(V, G_tmp, update=False)) # supposedly correct 
            # G.append(self.updateG(self.getV(Vi[-1].copy(), Ve), G[-1], update=False))
            # G.append(self.updateG(self.getV(Vi[-1].copy(), Ve), G_tmp, update=False))

            # estimate error 
            err_Vi = self.getErr(Vi[-2], Vi[-1])
            err_G  = self.getErr(G[-2], G[-1])

            max_err = np.max((np.max(err_Vi), np.max(err_G)))
            if max_err < tol:
                self.updateG(self.getV(Vi[-1].copy(), Ve), G[-1], update=True)
                convergence = True
            else:
                del G[0]
                del Vi[0]
            
            n_iters += 1

            # print(f'\t\t n_iter = {n_iters}')
            # print(f'\t\t\t max error = {max_err}')
        
        return Vi[-1]

        
    def getErr(self, x_0, x_1):
        """
        # TODO
        """
        
        err = 2 * np.abs(x_1-x_0)/(np.abs(x_1)+np.abs(x_0))
        err[np.isnan(err)] = 0.0
        
        return err


class MSSNetwork(MemristiveReservoir):
    """
    Class that represents an Metastable Switch Memristive network
    (see Nugent and Molter, 2014 for details)

    ...

    Attributes
    ----------
    W : numpy.ndarray
        reservoir's binary connectivity matrix
    I : numpy.ndarray
        set of internal nodes
    E : numpy.ndarray
        set of external nodes
    GR : numpy.ndarray
        set of grounded nodes
    n_internal_nodes : int
        number of internal nodes
    n_external_nodes : int
        number of external nodes
    n_grounded_nodes : int
        number of gorunded nodes
    G : numpy.ndarray
        matrix of conductances
    vA : numpy.ndarray of floats

    vB : numpy.ndarray of floats

    tc : numpy.ndarray of floats

    NMSS : numpy.ndarray of ints

    Woff : numpy.ndarray of floats

    Won : numpy.ndarray of floats

    Ga : numpy.ndarray of floats

    Gb : numpy.ndarray of floats

    Nb : numpy.ndarray of ints

    

    Methods
    ----------
    #TODO

    """

    # physical parameters of the MMS model
    k = 1.3806503e-23   # Boltzman's constant
    Q = 1.60217646e-19  # electron charge
    Temp = 298          # temperature
    b = Q/(k*Temp)
    VT = 1/b

    def __init__(self, vA=0.17, vB=0.22, tc=0.32e-3, NMSS=10000,\
                 Woff=0.91e-3, Won=0.87e-2, Nb=2000, noise=0.1, *args, **kwargs):
        """
        Constructor class for Memristive Networks following the Generalized
        Memristive Switch Model proposed in Nugent and Molter, 2014. Default
        parameter values correspond to an Ag-chalcogenide memristive device
        taken from Nugent and Molter, 2014.

        Parameters
        ----------
        w : (N, N) numpy.ndarray
            reservoir's binary connectivity matrix
            N: total number of nodes in the network (internal + external
            + grounded nodes)
        i_nodes : (n_internal_nodes,) numpy.ndarray
            indexes of internal nodes
            n_internal_nodes: number of internal nodes
        e_nodes : (n_external_nodes,) numpy.ndarray
            indexes of external nodes
            n_external_nodes: number of external nodes
        gr_nodes : (n_grounded_nodes,) numpy.ndarray
            indexes of grounded nodes
            n_grounded_nodes: number of grounded nodes
        vA : float. Default: 0.17

        vB : float. Default: 0.22

        tc : float. Default: 0.32e-3

        NMSS : int. Default: 10000

        Woff : float. Default: 0.91e-3

        Won : float. Default: 0.87e-2

        Nb : int. Default: 2000

        noise : float. Default: 0.1

        #TODO
        """
        super().__init__(*args, **kwargs)

        self.vA     = self.init_property(vA, noise)      # constant
        self.vB     = self.init_property(vB, noise)      # constant
        self.tc     = self.init_property(tc, noise)      # constant
        self.NMSS   = self.init_property(NMSS, noise)    # constant
        self.Woff   = self.init_property(Woff, noise)    # constant
        self.Won    = self.init_property(Won, noise)     # constant
        self._Ga = mask(self, np.divide(self.Woff, self.NMSS,
                                        where=self.NMSS != 0))  # constant
        self._Gb = mask(self, np.divide(self.Won, self.NMSS,
                                        where=self.NMSS != 0))  # constant

        self._Nb     = self.init_property(Nb, noise)
        self._G      = self._Nb * (self._Gb - self._Ga) + self.NMSS * self._Ga


    def dG(self, V, G=None, dt=1e-4):
        """
        #TODO
        This function updates the conductance matrix G given V

        Parameters
        ----------
        V : (N,N) numpy.ndarray
            matrix of voltages accross memristors

        Returns
        -------

        References
        ----------

        """
        
        # set Nb values
        if G is not None: 
            Gdiff1 = G - self.NMSS * self._Ga
            Gdiff2 = self._Gb - self._Ga
            Nb = mask(self, np.divide(Gdiff1, Gdiff2, where=Gdiff2 != 0))

        else: 
            Nb = self._Nb

        # ratio of dt to characterictic time of the device tc
        alpha = np.divide(dt, self.tc, where=self.tc != 0)

        # compute Pa
        exponent = -1 * (V - self.vA) / self.VT
        Pa = alpha / (1 + np.exp(exponent))

        # compute Pb
        exponent = -1 * (V + self.vB) / self.VT
        Pb = alpha * (1 - 1 / (1 + np.exp(exponent)))

        # compute dNb
        Na = self.NMSS - Nb
        
        Gab = np.random.binomial(Na.astype(int), mask(self, Pa))
        Gba = np.random.binomial(Nb.astype(int), mask(self, Pb))

        if check_symmetric(self._W):
            Gab = make_symmetric(Gab)
            Gba = make_symmetric(Gba)

        dNb = (Gab-Gba)

        return dNb

    
    def updateG(self, V, G=None, update=False):
        """
        # TODO
        """
        
        if G is None: G = self._G

        # compute dG
        dNb = self.dG(V=V, G=G)     
        dG  = dNb * (self._Gb-self._Ga)

        if update:  
            # update Nb
            self._Nb += dNb

            # update G
            self._G = self._Nb * (self._Gb - self._Ga) + self.NMSS * self._Ga        
            # self._G += dG   

        else:
            return self._G.copy() + dG # updated conductance


def mask(reservoir, a):
    """
    This functions converts to zero all entries in matrix 'a' for which there is
    no existent connection

    #TODO
    """

    a[np.where(reservoir._W == 0)] = 0
    return a


def check_symmetric(a, tol=1e-16):
    """
    This functions checks whether matrix 'a' is symmetric

    #TODO
    """
    return np.allclose(a, a.T, atol=tol)


def make_symmetric(a, copy_lower=True):
    if copy_lower:
        return np.tril(a,-1) + np.tril(a,-1).T
    else:
        return np.triu(a,1) + np.triu(a,1).T


def check_square(a):
    """
    This functions checks whether matrix 'a' is square

    #TODO
    """

    s = a.shape
    if s[0] == s[1]: return True
    else: return False


def reservoir(name, **kwargs):
    if name == 'EchoStateNetwork':
        return EchoStateNetwork(**kwargs)
    if name == 'MSSNetwork':
        return MSSNetwork(**kwargs)
