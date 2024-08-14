# -*- coding: utf-8 -*-
"""
Functionality for simulating reservoirs
"""
from abc import ABCMeta, abstractmethod
import time
import numpy as np
from numpy.linalg import pinv
from . import utils
from abc import ABC,abstractmethod
import cupy as cp
from joblib import Parallel, delayed
from abc import ABC,abstractmethod
from joblib import Parallel, delayed

class Reservoir(metaclass=ABCMeta):
    """
    Class that represents a general Reservoir object

    ...

    Attributes
    ----------
    w : numpy.ndarray
        reservoir connectivity matrix (source, target)
    _state : numpy.ndarray
        reservoir activation states
    n_nodes : int
        dimension of the reservoir

    Methods
    ----------
    # TODO

    simulate

    add_washout_time

    """

    def __init__(self, w):
        """
        Constructor class for general Reservoir Networks

        Parameters
        ----------
        w : (N, N) numpy.ndarray
            reservoir connectivity matrix (source, target)
            N: number of nodes in the network. If w is directed, then rows
            (columns) should correspond to source (target) nodes.
        """
        self.w = w
        self._state = None
        self.n_nodes = len(self.w)

    @abstractmethod
    def simulate(self, *args, **kwargs):
        pass

    def add_washout_time(self, *args, idx_washout=0):
        """
        Add washout time to reservoir states and corresponding arrays (e.g., label, sample weight)
        'ext_input'

        Parameters
        ----------
        idx_washout: int
            index up to which the values of arrays should be deleted
        args: numpy.ndarray
            reservoir states and any additional arrays where washout is to be applied

        Returns
        -------
        args: numpy.ndarray
            same arrays as in args but after washout is applied
        """

        # delete initial indexes of arrays in args
        argout = tuple(a[idx_washout:] for a in args)

        return argout


class EchoStateNetwork(Reservoir):
    """
    Class that represents an Echo State Network

    ...

    Attributes
    ----------
    w : numpy.ndarray
        reservoir connectivity matrix (source, target)
    _state : numpy.ndarray
        reservoir activation states
    n_nodes : int
        dimension of the reservoir
    activation_function : fct
    activation_function : fct
        type of activation function
    activation_function_derivative : fct
        derivative of activation function
    LE : numpy.ndarray
        Lyapunov exponents of the reservoir
    LE_trajectory : numpy.ndarray
        Lyapunov exponents of the reservoir over time
    activation_function_derivative : fct
        derivative of activation function
    LE : numpy.ndarray
        Lyapunov exponents of the reservoir
    LE_trajectory : numpy.ndarray
        Lyapunov exponents of the reservoir over time

    Methods
    -------
    # TODO

    simulate

    set_activation_function

    """

    def __init__(self, *args, activation_function='tanh', **kwargs):
        """
        Constructor class for Echo State Networks

        Parameters
        ----------
        w: (N, N) numpy.ndarray
            Reservoir connectivity matrix (source, target)
            N: number of nodes in the network. If w is directed, then rows
            (columns) should correspond to source (target) nodes.
        activation_function: str {'linear', 'elu', 'relu', 'leaky_relu',
            'sigmoid', 'tanh', 'step'}, default 'tanh'
            Activation function (nonlinearity of the system's units)
        """

        super().__init__(*args, **kwargs)

        # activation function
        self.activation_function = self.set_activation_function(
            activation_function, **kwargs)
        self.activation_function_derivative = self.get_derivative(
            activation_function, **kwargs)
        self.activation_function_derivative = self.get_derivative(
            activation_function, **kwargs)

    def simulate(
        self, ext_input, w_in, input_gain=None, ic=None, output_nodes=None,
        return_states=True, compute_LE = False, warmup = 0, **kwargs
    ):
        """
        Simulates reservoir dynamics given an external input signal
        'ext_input' and an input connectivity matrix 'w_in'

        Parameters
        ----------
        ext_input : (time, N_inputs) numpy.ndarray
            External input signal
            N_inputs: number of external input signals
        w_in : (N_inputs, N) numpy.ndarray
            Input connectivity matrix (source, target)
            N_inputs: number of external input signals
            N: number of nodes in the network
        ic : (N,) numpy.ndarray, optional
            Initial conditions
            N: number of nodes in the network. If w is directed, then rows
            (columns) should correspond to source (target) nodes.
        output_nodes : list or numpy.ndarray, optional
            List of nodes for which reservoir states will be returned if
            'return_states' is True.
        return_states : bool, optional
            If True, simulated resrvoir states are returned. True by default.
        compute_LE : bool, optional
            If True, compute the Lyapunov spectrum of the reservoir. False by default.
        compute_LE : bool, optional
            If True, compute the Lyapunov spectrum of the reservoir. False by default.
        kwargs:
            Other keyword arguments are passed to self.activation_function

        Returns
        -------
        self._state : (time, N) numpy.ndarray
            Activation states of the reservoir.
            N: number of nodes in the network if output_nodes is None, else
            number of output_nodes
        """
        # print('\n GENERATING RESERVOIR STATES ...')

        # if ext_input is list or tuple convert to numpy.ndarray
        if isinstance(ext_input, (list, tuple)):
            sections = utils.get_sections(ext_input)
            ext_input = utils.concat(ext_input)
            convert_to_list = True
        else:
            convert_to_list = False

        # initialize reservoir states
        timesteps = range(1, len(ext_input) + 1)
        self._state = np.zeros((len(timesteps) + 1, self.n_nodes))

        # set initial conditions
        if ic is not None:
            self._state[0, :] = ic

        # scale input connectivity matrix
        if input_gain is not None:
            w_in = input_gain * w_in

        if compute_LE and self.activation_function_derivative is not None:
            Q = np.eye(self.n_nodes)
            y = np.zeros(self.n_nodes)
            self.LE_trajectory = np.zeros((len(timesteps), self.n_nodes))

        # scale input connectivity matrix
        if input_gain is not None:
            w_in = input_gain * w_in

        if compute_LE and self.activation_function_derivative is not None:
            Q = np.eye(self.n_nodes)
            y = np.zeros(self.n_nodes)
            self.LE_trajectory = np.zeros((len(timesteps), self.n_nodes))

        # simulate dynamics
        for t in timesteps:
            # if (t > 0) and (t % 100 == 0):
            #     print(f'\t ----- timestep = {t}')
            synap_input = np.dot(
                self._state[t-1, :], self.w) + np.dot(ext_input[t-1, :], w_in)
            self._state[t, :] = self.activation_function(synap_input, **kwargs)

            if compute_LE and self.activation_function_derivative is not None:
                J = np.dot(np.diag(self.activation_function_derivative(synap_input)), self.w)
                Q = np.dot(J, Q)
                Q, R = np.linalg.qr(Q)
                yt = np.log2(np.abs(np.diag(R)))
                self.LE_trajectory[t-1, :] = yt
                if t > warmup:
                    y += yt
        
        if compute_LE and self.activation_function_derivative is not None:
            self.LE = y / (timesteps[-1] - warmup)

            if compute_LE and self.activation_function_derivative is not None:
                J = np.dot(np.diag(self.activation_function_derivative(synap_input)), self.w)
                Q = np.dot(J, Q)
                Q, R = np.linalg.qr(Q)
                yt = np.log2(np.abs(np.diag(R)))
                self.LE_trajectory[t-1, :] = yt
                if t > warmup:
                    y += yt
        
        if compute_LE and self.activation_function_derivative is not None:
            self.LE = y / (timesteps[-1] - warmup)

        # remove initial condition (to match the time index of _state
        # and ext_input)
        self._state = self._state[1:]

        # convert back to list or tuple
        if convert_to_list:
            self._state = utils.split(self._state, sections)

        # return the same type
        if return_states:
            if output_nodes is not None:
                if convert_to_list:
                    return [state[:, output_nodes] for state in self._state]
                else:
                    return self._state[:, output_nodes]
            else:
                return self._state

    def set_activation_function(self, function, **kwargs):

        def linear(x, **kwargs):
            m = kwargs.get('m', 1)
        def linear(x, **kwargs):
            m = kwargs.get('m', 1)
            return m * x

        def elu(x, **kwargs):
            alpha = kwargs.get('alpha', 0.5)
        def elu(x, **kwargs):
            alpha = kwargs.get('alpha', 0.5)
            x[x <= 0] = alpha*(np.exp(x[x <= 0]) - 1)
            return x

        def relu(x):
            return np.maximum(0, x)

        def leaky_relu(x, **kwargs):
            alpha = kwargs.get('alpha', 0.5)
        def leaky_relu(x, **kwargs):
            alpha = kwargs.get('alpha', 0.5)
            return np.maximum(alpha * x, x)

        def sigmoid(x):
            return 1.0 / (1 + np.exp(-x))

        def tanh(x):
            return np.tanh(x)

        def step(x, **kwargs):
            thr = kwargs.get('thr', 0.5)
            vmin = kwargs.get('vmin', 0)
            vmax = kwargs.get('vmax', 1)
        def step(x, **kwargs):
            thr = kwargs.get('thr', 0.5)
            vmin = kwargs.get('vmin', 0)
            vmax = kwargs.get('vmax', 1)
            return np.piecewise(x, [x < thr, x >= thr], [vmin, vmax]).astype(int)

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


class SpikingNeuralNetwork(Reservoir):
    """
    Class that represents a Spiking Neural Network
    Adapted from Kim et al., 2019 and Nicola & Clopath, 2017
    (https://github.com/rkim35/spikeRNN/blob/master/spiking/LIF_network_fnc.m)
    ...

    Attributes
    ----------
    w : (N, N) numpy.ndarray
        reservoir connectivity matrix (source, target)
        N: number of nodes in the network. If w is directed, then rows
        (columns) should correspond to source (target) nodes.
    _state : same as ext_input
        reservoir activation states
    n_nodes : int
        dimension of the reservoir
    inh : (N,) numpy.ndarray
        boolean array indicating whether a node is
        inhibitory (True) or excitatory (False)
        N: number of nodes in the network
    exc : (N,) numpy.ndarray
        boolean array indicating whether a node is
        excitatory (True) or inhibitory (False)
        N: number of nodes in the network
    som : (N,) numpy.ndarray
        boolean array indicating whether a node is
        somatostatin-expressing (True) or not (False)
        N: number of nodes in the network
    dt : float
        sampling rate (in s)
    T : float
        trial duration (in s)
    nt : int
        number of time steps
    td : float or numpy.ndarray
        decay time constants of the synaptic filter model (in s)
    REC : (nt, N) numpy.ndarray
        membrane voltage tracings (mV)
        nt: number of time steps
        N: number of nodes in the network
    Is : (N, nt) numpy.ndarray
        external input current
        nt: number of time steps
        N: number of nodes in the network
    IPSCs : (N, nt) numpy.ndarray
        post synaptic currents over time
        nt: number of time steps
        N: number of nodes in the network
    spk : (N, nt) numpy.ndarray
        spike raster
        nt: number of time steps
        N: number of nodes in the network
    rs : (N, nt/timescale) numpy.ndarray
        filtered firing rates over time
        nt: number of time steps
        N: number of nodes in the network
        timescale: number of internal time steps per external time steps
    hs : (N, nt) numpy.ndarray
        filtered firing rates over time (synaptic input accumulation)
        nt: number of time steps
        N: number of nodes in the network
    tspike : (N_spikes, 2) numpy.ndarray
        spike times (in s)
        N_spikes: number of spikes
        tspike[:, 0]: spike neuronal indices
        tspike[:, 1]: spike times (in s)
    inh_fr : (N_inh,) numpy.ndarray
        average firing rates of inhibitory neurons
        N_inh: number of inhibitory neurons
    exc_fr : (N_exc,) numpy.ndarray
        average firing rates of excitatory neurons
        N_exc: number of excitatory neurons
    all_fr : (N,) numpy.ndarray
        average firing rates of all neurons
        N: number of neurons in the network

    Methods
    -------
    # TODO

    simulate

    """

    def __init__(self, *args, inh = 0.2, som = 0., apply_Dale = True, **kwargs):
        """
        Constructor class for Spiking Neural Networks

        Parameters
        ----------
        w: (N, N) numpy.ndarray
            Reservoir connectivity matrix (source, target)
            N: number of nodes in the network. If w is directed,
            then rows (columns) should correspond to source (target) nodes.
        inh: float or (N,) numpy.ndarray, optional
            If float, inh should be in the range [0, 1] and
            indicates the proportion of inhibitory neurons in the network.
            If numpy.ndarray of shape (N,), then inh is a
            boolean array indicating whether a node is
            inhibitory (True) or excitatory (False).
            This parameter is used to apply Dale's principle, constraining
            the connectivity matrix such that a neuron can only be either
            excitatory or inhibitory, but not both.
            Default: 0.2
        som: float or (N,) numpy.ndarray, optional
            If float, som inh should be in the range [0, 1] and
            indicates the proportion of somatostatin-expressing interneurons
            in the network.
            If numpy.ndarray of shape (N,), then som is a
            boolean array indicating whether a node is
            somatostatin-expressing (True) or not (False).
            Importantly, somatostatin-expressing interneurons are a
            subset of the inhibitory neurons.
            This parameter is used to constrain a
            common cortical microcircuit motif where somatostatin-expressing
            inhibitory neurons do not receive inhibitory input.
            Default: 0
        """

        super().__init__(*args, **kwargs)

        if not(isinstance(inh, (float, np.ndarray))):
            raise TypeError('inh must be float or numpy.ndarray')

        if not(isinstance(som, (float, np.ndarray))):
            raise TypeError('som must be float or numpy.ndarray')

        if isinstance(inh, float) and isinstance(som, np.ndarray):
            raise TypeError('inh must be numpy.ndarray if som is numpy.ndarray')

        if isinstance(inh, np.ndarray) and isinstance(som, np.ndarray):
            if not np.all(som <= inh):
                raise ValueError('som must be a subset of inh')

        if isinstance(inh, np.ndarray):
            if not np.issubdtype(inh.dtype, np.bool_):
                raise TypeError('inh must be boolean')
            exc = ~inh
        else:
            if not (0 <= inh <= 1):
                raise ValueError('inh and exc must be in the range [0, 1]')
            inh = np.random.rand(self.n_nodes) < inh
            exc = ~inh

        if isinstance(som, np.ndarray):
            if not np.issubdtype(som.dtype, np.bool_):
                raise TypeError('som must be boolean')
            som_idx = np.where(som)[0]
        else:
            if not (0 <= som <= 1):
                raise ValueError('inh and exc must be in the range [0, 1]')
            if som > 0:
                som_size = int(np.round(som * np.sum(inh)))
                som = np.zeros(self.n_nodes, dtype=bool)
                som_idx = np.random.choice(np.where(inh)[0], som_size, replace=False)
                som[som_idx] = True
            else:
                som = np.zeros(self.n_nodes, dtype=bool)

        # apply Dale's principle if inhibitory neurons are specified
        if np.any(inh):
            w = np.abs(self.w)

            # mask matrix imposing Dale's principle
            mask = np.eye(self.n_nodes, dtype=np.float32)
            mask[np.where(inh)[0], np.where(inh)[0]] = -1

            # mask matrix imposing wiring motif mediated by
            # somatostatin-expressing interneurons
            som_mask = np.ones((self.n_nodes, self.n_nodes), dtype=np.float32)
            if np.any(som):
                for i in som_idx:
                    som_mask[i, np.where(inh)[0]] = 0

            self.w = np.multiply(np.matmul(w, mask), som_mask)

        self.inh = inh
        self.exc = exc
        self.som = som

    def simulate(
        self, ext_input, w_in,
        downsample = 1, taus = 35,
        tau_min = 20, tau_max = 50, sig_param = None,
        timescale = 100, dt = 0.05, tref = 2, tm = 10,
        vreset = -65, vpeak = -40, tr = 2,
        stim_mode = None, stim_dur = None, stim_units = None, stim_val = 0.5,
        input_gain=None, ic=None, output_nodes=None,
        return_states=True
    ):
        """
        Simulates the dynamics of a spiking neural network given
        an external input signal 'ext_input',
        an input connectivity matrix 'w_in', and
        synaptic decay time constants 'taus'

        Parameters
        ----------
        ext_input : (time, N_inputs) numpy.ndarray
            External input signal
            N_inputs: number of external input signals
        w_in : (N_inputs, N) numpy.ndarray
            Input connectivity matrix (source, target)
            N_inputs: number of external input signals
            N: number of nodes in the network
        taus : float or (2,) array_like, optional
            Parameter(s) that modify the decay time constants of the
            synaptic filter model.
            If float, then the same decay time constant is used
            for all neurons.
            If array_like, then:
            taus[0]: minimum
            taus[1]: maximum
        downsample : int, optional
            Downsamples external input signal by a factor of 'downsample'.
            Default: 1
        taus : float or (N,) numpy.ndarray, optional
            Decay time constants of the synaptic filter model (in ms).
            If float, then the same decay time constant tau
            is used for all neurons.
            If numpy.ndarray of shape (N,), then:
            taus[i]: decay time constant of neuron i
            N: number of nodes in the network
            Default: 35
        tau_min : float, optional
            Minimum decay time constant of the synaptic filter model (in ms).
            Default: 20
            Note: used in combination with tau_max and sig_param;
            overrides taus
        tau_max : float, optional
            Maximum decay time constant of the synaptic filter model (in ms).
            Default: 50
            Note: used in combination with tau_min and sig_param;
            overrides taus
        sig_param : float, (N,) numpy.ndarray, or string, optional
            Parameter(s) of the sigmoid function that constrains
            the decay time constants of the synaptic filter model.
            If float, then the same parameter is used for all neurons
            yielding a single decay time constant for all neurons.
            If numpy.ndarray of shape (N,), then:
            sig_param[i]: parameter of the sigmoid function of neuron i
            and a different decay time constant is used for each neuron.
            If 'normal', then N values are sampled from a normal distribution
            with mean = 0 and standard deviation = 1.
            N: number of nodes in the network.
            Default: None
            Note: used in combination with tau_min and tau_max;
            overrides taus
        dt : float, optional
            Sampling rate (in ms). Default: 0.05
        timescale : float, optional
            number of internal time steps per external time steps
            Default: 100
        tref : float, optional
            Refractory time constant (in ms). Default: 2
        tm : float, optional
            Membrane time constant (in ms). Default: 10
        vreset : float, optional
            Reset voltage (in mV). Default: -65
        vpeak : float, optional
            Peak voltage (in mV). Default: -40
        tr : float, optional
            Rise time constant (in ms). Default: 2
        stim_mode : {'exc', 'inh'}, optional
            Indicates whether to apply artificial
            depolarizing ('exc') or hyperpolarizing ('inh')
            stimulation (modelling optogenetic stimulation).
            Default: None
        stim_dur : (2,) numpy.ndarray, optional
            Time interval (in timesteps) during which
            artificial stimulation or inhibition is applied.
            stim_dur[0]: stimulus onset
            stim_dur[1]: stimulus offset
            Default: None
        stim_units : (N,) numpy.ndarray, optional
            Indices of neurons that will be stimulated or inhibited.
            Default: None
        stim_val : float, optional
            Value of the artificial stimulation or inhibition (in mV).
            Default: 0.5
        input_gain : float, optional
            Constant gain that scales w_in. Default: None
        ic : (N,) numpy.ndarray, optional
            Initial voltage conditions
            N: number of nodes in the network.
            Default: None
        output_nodes : array_like, optional
            List of nodes for which reservoir states will be returned if
            'return_states' is True. Default: None
        return_states : bool, optional
            If True, simulated reservoir states are returned.
            Default: True

        Returns
        -------
        self._state : (time, N) numpy.ndarray
            Activation states of the reservoir.
            N: number of nodes in the network if output_nodes is None, else
            number of output_nodes
        """

        # inhibitory and excitatory neuron indices
        inh_ind = np.where(self.inh)[0]
        exc_ind = np.where(self.exc)[0]

        # if ext_input is list or tuple convert to numpy.ndarray
        if isinstance(ext_input, (list, tuple)):
            sections = utils.get_sections(ext_input)
            ext_input = utils.concat(ext_input)
            convert_to_list = True
        else:
            convert_to_list = False

        # scale input connectivity matrix
        if input_gain is not None:
            w_in = input_gain * w_in
        w_in = w_in.T

        # Downsample input stimulus
        ext_input = ext_input.T
        ext_input = ext_input[:, ::downsample]
        ext_stim = np.dot(w_in, ext_input)

        # Set simulation parameters
        # sampling rate (s)
        dt = dt/1000 * downsample
        # trial duration (s)
        T = (ext_input.shape[1]) * dt * timescale
        # number of time steps
        nt = int(np.round(T / dt))
        # refractory time constant (s)
        tref = tref/1000
        # membrane time constant (s)
        tm = tm/1000
        # rise time constant (s)
        tr = tr/1000

        # Synaptic decay time constants (in sec)
        # for the synaptic filter
        # td: decay time constants
        if sig_param is not None:
            if sig_param == 'normal':
                sig_param = np.random.randn(self.n_nodes)
            td = (1 / (1 + np.exp(-sig_param)) * (tau_max - tau_min)
                  + tau_min) / 1000
        else:
            td = taus/1000

        # Initialize variables for LIF neurons simulation
        # post synaptic current
        IPSC = np.zeros(self.n_nodes)
        # filtered firing rates (synaptic input accumulation)
        h = np.zeros(self.n_nodes)
        # filtered firing rates
        r = np.zeros(self.n_nodes)
        # filtered firing rates (rising phase)
        hr = np.zeros(self.n_nodes)
        # contribution of each neuron to IPSC
        JD = np.zeros(self.n_nodes)
        # number of spikes
        ns = 0

        # Initialize voltage
        if ic is not None:
            v = ic
        else:
            v = vreset + np.random.rand(self.n_nodes) * (30 - vreset)

        # Initialize storage arrays for recording results
        # membrane voltage tracings (mV)
        REC = np.zeros((nt, self.n_nodes))
        # external input current
        Is = np.zeros((self.n_nodes, nt))
        # post synaptic currents over time
        IPSCs = np.zeros((self.n_nodes, nt))
        # spike raster
        spk = np.zeros((self.n_nodes, nt))
        # filtered firing rates over time
        rs = np.zeros((self.n_nodes, nt))
        # filtered firing rates over time (synaptic input accumulation)
        hs = np.zeros((self.n_nodes, nt))

        tlast = np.zeros(self.n_nodes) # last spike time

        BIAS = vpeak # bias current

        # Start the simulation loop
        for i in range(nt):
            # Record IPSC over time
            IPSCs[:, i] = IPSC

            # Calculate synaptic current
            I = IPSC + BIAS
            I = I + ext_stim[:, i // timescale]
            Is[:, i] = ext_stim[:, i // timescale]

            # Compute voltage change according to LIF equation
            dv = (dt * i > tlast + tref) * (-v + I) / tm
            v = v + dt * dv + np.random.randn(self.n_nodes) / 10

            # Apply artificial stimulation/inhibition
            if stim_mode == 'exc':
                if stim_dur is None:
                    raise ValueError('stim_dur not specified')
                elif stim_dur[0] <= i < stim_dur[1]:
                    if stim_units is None:
                        raise ValueError('stim_units not specified')
                    elif np.random.rand() < 0.5:
                        v[stim_units] = v[stim_units] + stim_val
            elif stim_mode == 'inh':
                if stim_dur is None:
                    raise ValueError('stim_dur not specified')
                elif stim_dur[0] <= i < stim_dur[1]:
                    if stim_units is None:
                        raise ValueError('stim_units not specified')
                    elif np.random.rand() < 0.5:
                        v[stim_units] = v[stim_units] - stim_val

            # Indices of neurons that have fired
            index = np.where(v >= vpeak)[0]

            # Store spike times and compute weighted contributions to IPSC
            if len(index) > 0:
                JD = np.sum(self.w[:, index], axis=1)
                curr_ts = np.column_stack((index, np.zeros(len(index)) + dt * i))
                if ns == 0:
                    tspike = curr_ts
                else:
                    tspike = np.append(tspike, curr_ts, axis=0)
                ns = ns + len(index)

            # Set refractory period
            tlast = tlast + (dt * i - tlast) * (v >= vpeak)

            # Compute IPSC and filtered firing rates
            # If the rise time is 0, then use the single synaptic filter,
            # otherwise (i.e. if the rise time is positive)
            # use the double-exponential filter
            if tr == 0:
                IPSC = IPSC * np.exp(-dt / td) + JD * (len(index) > 0) / td
                r = r * np.exp(-dt / td) + (v >= vpeak) / td
                rs[:, i] = r
            else:
                IPSC = IPSC * np.exp(-dt / td) + h * dt
                h = h * np.exp(-dt / tr) + JD * (len(index) > 0) / (tr * td)
                hs[:, i] = h

                r = r * np.exp(-dt / td) + hr * dt
                hr = hr * np.exp(-dt / tr) + (v >= vpeak) / (tr * td)
                rs[:, i] = r

            # Record spikes
            spk[:, i] = v >= vpeak

            # Cap depolarization
            v = v + (30 - v) * (v >= vpeak)

            # Record membrane voltage
            REC[i, :] = v

            # Reset voltage after spike
            v = v + (vreset - v) * (v >= vpeak)

        # Compute average firing rates for different populations
        inh_fr = np.zeros(len(inh_ind))
        for i in range(len(inh_ind)):
            inh_fr[i] = np.sum(spk[inh_ind[i], :] > 0) / T

        exc_fr = np.zeros(len(exc_ind))
        for i in range(len(exc_ind)):
            exc_fr[i] = np.sum(spk[exc_ind[i], :] > 0) / T

        all_fr = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            all_fr[i] = np.sum(spk[i, 10:] > 0) / T

        # Average over every 'timescale' time steps
        rs = rs.reshape(rs.shape[0], int(rs.shape[1]/timescale), timescale).mean(axis = -1)
        self._state = rs.T

        # Convert back to list or tuple
        if convert_to_list:
            self._state = utils.split(self._state, sections)

        self.dt = dt
        self.T = T
        self.nt = nt
        self.td = td
        self.REC = REC
        self.Is = Is
        self.IPSCs = IPSCs
        self.spk = spk
        self.rs = rs
        self.hs = hs
        self.tspike = tspike
        self.inh_fr = inh_fr
        self.exc_fr = exc_fr
        self.all_fr = all_fr

        # Return the same type
        if return_states:
            if output_nodes is not None:
                if convert_to_list:
                    return [state[:, output_nodes] for state in self._state]
                else:
                    return self._state[:, output_nodes]
            else:
                return self._state

class MemristiveReservoir(ABC):
    """
    Class that represents a general Memristive Reservoir

    ...

    Attributes
    ----------
    _W : numpy.ndarray
        reservoir's binary connectivity matrix
    _I : numpy.ndarray
        indices of internal nodes
    _E : numpy.ndarray
        indices of external nodes
    _GR : numpy.ndarray
        indices of grounded nodes
    n_internal_nodes : int
        number of internal nodes
    n_external_nodes : int
        number of external nodes
    n_grounded_nodes : int
        number of gorunded nodes
    n_nodes : int
        total number of nodes (internal, external, and ground)
    G : numpy.ndarray
        matrix of conductances
    save_conductance : bool
        Indicates whether to save conductance state after each simulation
        step. If True, then will be stored in self._G_history. This will
        increase memory demands.
    _state : numpy.ndarray
        reservoir activation states

    Methods
    ----------
    # TODO

    References
    ----------
    # TODO

    """

    def __init__(self, w, int_nodes, ext_nodes, gr_nodes, save_conductance=False, *args, **kwargs):
        """
        Constructor class for Memristive Networks. Memristive networks are an
        abstraction for physical networks of memristive elements.

        Parameters
        ----------
        w : (N, N) numpy.ndarray
            reservoir's binary connectivity matrix
            N: total number of nodes in the network (internal + external
            + grounded nodes)
        int_nodes : (n_internal_nodes,) numpy.ndarray
            indexes of internal nodes
            n_internal_nodes: number of internal nodes
        ext_nodes : (n_external_nodes,) numpy.ndarray
            indexes of external nodes
            n_external_nodes: number of external nodes
        gr_nodes : (n_grounded_nodes,) numpy.ndarray
            indexes of grounded nodes
            n_grounded_nodes: number of grounded nodes
        save_conductance : bool, optional
            Indicates whether to save conductance state after each simulation
            step. If True, then will be stored in self._G_history. This will
            increase memory demands. Default: False
        """
        # super().__init__(*args, **kwargs)
        self._W = self.setW(w)
        self._I = np.asarray(int_nodes)
        self._E = np.asarray(ext_nodes)
        self._GR = np.asarray(gr_nodes)

        self._n_internal_nodes = len(self._I)
        self._n_external_nodes = len(self._E)
        self._n_grounded_nodes = len(self._GR)
        self._n_nodes = len(self._W)

        self._G = None

        self.save_conductance = save_conductance
        self._G_history = None

        self._state = None

    def setW(self, w):
        """
        # TODO
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
        if not utils.check_symmetric(w):

            # connections in upper diagonal
            upper_diag = w[np.triu_indices_from(w, 1)]

            # connections in lower diagonal
            lower_diag = w.T[np.triu_indices_from(w, 1)]

            # matrix of undirected connections
            W = np.zeros_like(w).astype(int)
            W[np.triu_indices_from(w, 1)] = np.logical_or(upper_diag,
                                                          lower_diag
                                                          ).astype(int)

            return utils.make_symmetric(W, copy_lower=False)

        else:
            return w

    def init_property(self, mean, std=0.1, seed=None):
        """
        This function initializes property matrices following a normal
        distribution with mean = 'mean' and standard deviation = 'mean' * 'std'

        Parameters
        ----------
        # TODO
        seed : int, array_like[ints], SeedSequence, BitGenerator, Generator, optional
            seed to initialize the random number generator, by default None
            for details, see numpy.random.default_rng()

        Returns
        -------
        # TODO

        """

        # use random number generator for reproducibility
        rng = np.random.default_rng(seed=seed)

        p = rng.normal(mean, std*mean, size=self._W.shape)
        p = utils.make_symmetric(p)

        return p * self._W.astype(np.float64)  # ma.masked_array(p, mask=np.logical_not(self._W))
    
    def solveVi(self, Ve, Vgr=None, G=None, **kwargs):
        """
        This function uses Kirchhoff's law to estimate voltage at the internal
        nodes based on the current state of the conductance matrix G, a given
        external input voltage 'V_E' and the ground level voltage 'V_GR'

        Parameters
        ----------
        # TODO

        Returns
        -------
        # TODO

        """

        if Vgr is None:
            Vgr = np.zeros((self._n_grounded_nodes))
        if G is None:
            G = self._G

        # TODO: verify that the axis along which the sum is performed is correct
        # matrix N
        N = np.zeros((self._n_nodes, self._n_nodes))
        np.fill_diagonal(N, np.sum(G, axis=1))
            
            
        # matrix A
        A = N - G

        # inverse matrix A_II
        A_II = A[np.ix_(self._I, self._I)]
        # print(matrix_rank(A_II, hermitian=utils.check_symmetric(A_II)))
        A_II_inv = pinv(A_II)

        # matrix HI
        H_IE = np.dot(G[np.ix_(self._I, self._E)], Ve)
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
        # TODO

        Returns
        -------
        # TODO

        """

        # set voltage at grounded nodes
        if Vgr is None:
            Vgr = np.zeros((self._n_grounded_nodes))

        # set of all nodal voltages
        #How does it make sure the correct voltages are paired with each correct node
        #ANSWER: In the below nodes concat, the order of nodes is the same as voltages, thus the nv_dict
        #        is a pairing of (node_index[in W] , voltage)
        #How does it make sure the correct voltages are paired with each correct node
        #ANSWER: In the below nodes concat, the order of nodes is the same as voltages, thus the nv_dict
        #        is a pairing of (node_index[in W] , voltage)
        voltage = np.concatenate(
            [Vi[:, np.newaxis], Ve[:, np.newaxis], Vgr[:, np.newaxis]]).squeeze()

        # set of all nodes (internal + external + grounded)
        nodes = np.concatenate(
            [self._I[:, np.newaxis], self._E[:, np.newaxis], self._GR[:, np.newaxis]]).squeeze()

        # dictionary that groups pairs of voltage
        nv_dict = {n:v for n, v in zip(nodes, voltage)}
        nv_dict = {n:v for n, v in zip(nodes, voltage)}

        # voltage across memristors
        #Goes across each connection (non-zero in W) and find the voltage using kirchoffs law 
        #loops through all non-zero (i,j) positions in W and sets that "Connection Voltage" as the difference
        #between individual voltages, stored in the nv_dict -> (node_index, voltage)
        #This difference in voltage (Voltage Drop) at (i,j) is stored in V at position (i,j)
        
        
        # V = np.zeros_like(self._W).astype(float)

        # def pairwise(i, j):
        #     if j > i:
        #         V[i, j] = nv_dict[i] - nv_dict[j]
        #     else:
        #         V[i, j] = nv_dict[j] - nv_dict[i]

        # for i, j in list(zip(*np.where(self._W != 0))):
        #     if j > i:
        #         V[i, j] = nv_dict[i] - nv_dict[j]
        #     else:
        #         V[i, j] = nv_dict[j] - nv_dict[i]
        #Goes across each connection (non-zero in W) and find the voltage using kirchoffs law 
        #loops through all non-zero (i,j) positions in W and sets that "Connection Voltage" as the difference
        #between individual voltages, stored in the nv_dict -> (node_index, voltage)
        #This difference in voltage (Voltage Drop) at (i,j) is stored in V at position (i,j)
        
        
        V = np.zeros_like(self._W).astype(float)

        # def pairwise(i, j):
        #     if j > i:
        #         V[i, j] = nv_dict[i] - nv_dict[j]
        #     else:
        #         V[i, j] = nv_dict[j] - nv_dict[i]

        for i, j in list(zip(*np.where(self._W != 0))):
            if j > i:
                V[i, j] = nv_dict[i] - nv_dict[j]
            else:
                V[i, j] = nv_dict[j] - nv_dict[i]

        # Parallel(n_jobs=8,prefer='processes')(delayed(pairwise)(i,j) for i, j in list(zip(*np.where(self._W != 0))))

        #removes voltage values from V in positions where there is no connection in W 
        # Parallel(n_jobs=8,prefer='processes')(delayed(pairwise)(i,j) for i, j in list(zip(*np.where(self._W != 0))))

        # vec = np.zeros(len(self._W), dtype=np.float32)
        # for k,v in nv_dict.items():
        #     vec[k] = v

        # V = np.abs(np.subtract.outer(vec,vec))

        #removes voltage values from V in positions where there is no connection in W 
        return self.mask(V)

    def simulate(self, Vext, ic=None, mode='forward'):
        """
        Simulates the dynamics of a memristive reservoir given an external
        voltage signal V_E

        Parameters
        ----------
        Vext : (time, N_external_nodes) numpy.ndarray
            External voltage signal
            N_external_nodes: number of external (input) nodes
        ic : (N_internal_nodes,) numpy.ndarray
            Initial conditions
            N_internal_nodes: number of internal (output) nodes
        mode : {'forward', 'backward'}
            Refers to the method used to solve the system of equations.
            Use 'forward' for explicit Euler method, and 'backward' for
            implicit Euler method.

        Returns
        -------
        self._state : (time, N) numpy.ndarray
            activation states of the reservoir; includes all the nodes
            N: number of nodes in the network
        """
        #Forward is explicit WORKING, backward is implicit 
        #Forward is explicit WORKING, backward is implicit 

        # print('\n GENERATING RESERVOIR STATES ...')
        # print(f'\n SIMULATING STATES IN {mode.upper()} MODE ...')

        # initialize reservoir states
        self._state = np.zeros((len(Vext), self._n_nodes))

        # initialize array for storing conductance history if needed
        if self.save_conductance:
            self._G_history = np.zeros((len(Vext), self._n_nodes,
                                        self._n_nodes))

        for t, Ve in enumerate(Vext):
            if mode == 'forward':

                # if (t>0) and (t%10 == 0): print(f'\t ----- timestep = {t}')

                # store external voltages
                #self._state[t, self._E] = Ve
                # if (t>0) and (t%10 == 0): print(f'\t ----- timestep = {t}')

                # store external voltages
                #self._state[t, self._E] = Ve

                # get voltage at internal nodes
                Vi = self.solveVi(Ve)

                # update matrix of voltages across memristors
                V = self.getV(Vi, Ve)

                # update conductance
                self.updateG(V=V, update=True)

            elif mode == 'backward':

                if (t > 0) and (t % 100 == 0):
                    print(f'\t ----- timestep = {t}')

                # get voltage at internal nodes
                Vi = self.iterate(Ve)

            # store activation states
            self._state[t, self._E] = Ve
            self._state[t, self._I] = Vi

            # store conductance
            if self.save_conductance:
                self._G_history[t] = self._G

        #self._state is a (t x N_nodes) matrix which keeps track of node voltage across time
        #self._state is a (t x N_nodes) matrix which keeps track of node voltage across time
        return self._state

    def iterate(self, Ve, tol=5e-2, iters=100):
        """
        # TODO

        """

        # initial guess for voltage at internal nodes
        Vi = [self.solveVi(Ve=Ve, G=self._G)]

        # initial guess for conductance
        G = [self._G]

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
            # supposedly correct
            G.append(self.updateG(V, G_tmp, update=False))
            # G.append(self.updateG(self.getV(Vi[-1].copy(), Ve), G[-1], update=False))
            # G.append(self.updateG(self.getV(Vi[-1].copy(), Ve), G_tmp, update=False))

            # estimate error
            err_Vi = self.getErr(Vi[-2], Vi[-1])
            err_G = self.getErr(G[-2], G[-1])

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

    @abstractmethod
    def dG(self, V, G=None, dt=1e-4, seed=None):
        pass

    @abstractmethod
    def updateG(self, V, G=None, update=False):
        pass

    @abstractmethod
    def dG(self, V, G=None, dt=1e-4, seed=None):
        pass

    @abstractmethod
    def updateG(self, V, G=None, update=False):
        pass

    def getErr(self, x_0, x_1):
        """
        # TODO
        """

        err = 2 * np.abs(x_1-x_0)/(np.abs(x_1)+np.abs(x_0))
        err[np.isnan(err)] = 0.0

        return err

    def mask(self, a):
        """
        This functions converts to zero all entries in matrix 'a' for which there is
        no existent connection
        # TODO
        """

        a[np.where(self._W == 0)] = 0
        return a


class MSSNetwork(MemristiveReservoir):
    """
    Class that represents a Metastable Switch Memristive network
    (see Nugent and Molter, 2014 for details)

    ...

    Attributes
    ----------
    w : numpy.ndarray
        reservoir's binary connectivity matrix
    I : numpy.ndarray
        indices of internal nodes
    E : numpy.ndarray
        indices of external nodes
    GR : numpy.ndarray
        indices of grounded nodes
    n_internal_nodes : int
        number of internal nodes
    n_external_nodes : int
        number of external nodes
    n_grounded_nodes : int
        number of gorunded nodes
    n_nodes : int
        total number of nodes (internal, external, and ground)
    G : numpy.ndarray
        matrix of conductances
    save_conductance : bool
        Indicates whether to save conductance state after each simulation
        step. If True, then will be stored in self._G_history. This will
        increase memory demands.
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
    # TODO

    """

    # physical parameters of the MMS model
    k = 1.3806503e-23   # Boltzman's constant
    Q = 1.60217646e-19  # electron charge
    Temp = 298          # temperature
    b = Q/(k*Temp)
    VT = 1/b

    def __init__(self, vA=0.17, vB=0.22, tc=0.32e-3, NMSS=10000,
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
        int_nodes : (n_internal_nodes,) numpy.ndarray
            indexes of internal nodes
            n_internal_nodes: number of internal nodes
        ext_nodes : (n_external_nodes,) numpy.ndarray
            indexes of external nodes
            n_external_nodes: number of external nodes
        gr_nodes : (n_grounded_nodes,) numpy.ndarray
            indexes of grounded nodes
            n_grounded_nodes: number of grounded nodes
        save_conductance : bool, optional
            Indicates whether to save conductance state after each simulation
            step. If True, then will be stored in self._G_history. This will
            increase memory demands. Default: False
        vA : float. Default: 0.17

        vB : float. Default: 0.22

        tc : float. Default: 0.32e-3

        NMSS : int. Default: 10000

        Woff : float. Default: 0.91e-3

        Won : float. Default: 0.87e-2

        Nb : int. Default: 2000

        noise : float. Default: 0.1

        # TODO
        """
        super().__init__(*args, **kwargs)

        self.vA = self.init_property(vA, noise)      # constant
        self.vB = self.init_property(vB, noise)      # constant
        self.tc = self.init_property(tc, noise)      # constant
        self.NMSS = self.init_property(NMSS, noise)    # constant Note: This sets a different number of switches per memristor 
        self.Woff = self.init_property(Woff, noise)    # constant
        self.Won = self.init_property(Won, noise)     # constant
        self._Ga = self.mask(np.divide(self.Woff,self.NMSS)) # constant
        self._Gb = self.mask(np.divide(self.Won,self.NMSS))   # constant

        self._Nb = self.init_property(Nb, noise)
        self._G = self._Nb * (self._Gb - self._Ga) + self.NMSS * self._Ga

    def dG(self, V, G=None, dt=1e-4, seed=None):
        """
        # TODO
        This function updates the conductance matrix G given V

        Parameters
        ----------
        V : (N,N) numpy.ndarray
            matrix of voltages accross memristors
        seed : int, array_like[ints], SeedSequence, BitGenerator, Generator, optional
            seed to initialize the random number generator, by default None
            for details, see numpy.random.default_rng()

        Returns
        -------

        References
        ----------

        """

        # set Nb values
        if G is not None:
            Gdiff1 = G - self.NMSS * self._Ga
            Gdiff2 = self._Gb - self._Ga
            Nb = self.mask(np.divide(Gdiff1,Gdiff2))

        else:
            Nb = self._Nb

        # ratio of dt to characterictic time of the device tc
        alpha = np.divide(dt, self.tc, where=self.tc != 0)
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

        # use random number generator for reproducibility
        rng = np.random.default_rng(seed=seed)

        Gab = rng.binomial(Na.astype(int), self.mask(Pa))
        Gba = rng.binomial(Nb.astype(int), self.mask(Pb))

        if utils.check_symmetric(self._W):
            Gab = utils.make_symmetric(Gab)
            Gba = utils.make_symmetric(Gba)

        dNb = (Gab-Gba)

        return dNb

    def updateG(self, V, G=None, update=False):
        """
        # TODO
        """

        if G is None:
            G = self._G

        # compute dG
        dNb = self.dG(V=V, G=G)
        dG = dNb * (self._Gb-self._Ga)

        if update:
            # update Nb
            self._Nb += dNb

            # update G
            self._G = self._Nb * (self._Gb - self._Ga) + self.NMSS * self._Ga
            # self._G += dG

        else:
            return self._G.copy() + dG  # updated conductance


class MemristiveReservoirCupy(ABC):
    """
    Class that represents a general Memristive Reservoir

    This is a copy of the above Memristive reservoir, with the changes to incorporate CuPy for computation speedup
    ...

    Attributes
    ----------
    _W : numpy.ndarray
        reservoir's binary connectivity matrix
    _I : numpy.ndarray
        indices of internal nodes
    _E : numpy.ndarray
        indices of external nodes
    _GR : numpy.ndarray
        indices of grounded nodes
    n_internal_nodes : int
        number of internal nodes
    n_external_nodes : int
        number of external nodes
    n_grounded_nodes : int
        number of gorunded nodes
    n_nodes : int
        total number of nodes (internal, external, and ground)
    G : numpy.ndarray
        matrix of conductances
    save_conductance : bool
        Indicates whether to save conductance state after each simulation
        step. If True, then will be stored in self._G_history. This will
        increase memory demands.
    _state : numpy.ndarray
        reservoir activation states

    Methods
    ----------
    # TODO

    References
    ----------
    # TODO

    """

    def __init__(self, w, int_nodes, ext_nodes, gr_nodes, save_conductance=False, save_dissipated=False ,save_voltage=False,*args, **kwargs):
        """
        Constructor class for Memristive Networks. Memristive networks are an
        abstraction for physical networks of memristive elements.

        Parameters
        ----------
        w : (N, N) numpy.ndarray
            reservoir's binary connectivity matrix
            N: total number of nodes in the network (internal + external
            + grounded nodes)
        int_nodes : (n_internal_nodes,) numpy.ndarray
            indexes of internal nodes
            n_internal_nodes: number of internal nodes
        ext_nodes : (n_external_nodes,) numpy.ndarray
            indexes of external nodes
            n_external_nodes: number of external nodes
        gr_nodes : (n_grounded_nodes,) numpy.ndarray
            indexes of grounded nodes
            n_grounded_nodes: number of grounded nodes
        save_conductance : bool, optional
            Indicates whether to save conductance state after each simulation
            step. If True, then will be stored in self._G_history. This will
            increase memory demands. Default: False
        save_dissipated : bool, optional
            Indicates whether to save energy dissipated between timesteps. 
            If True, then the energy dissipated will be stored in self.energy_dissipated. 
            This will increase memory demands. Default: False
        """
        #setW now returns a CuPy array
        self._W = self.setW(w)
        #moves all of the arrays to the Device (GPU)
        self._I = cp.asarray(int_nodes)
        self._E = cp.asarray(ext_nodes)
        self._GR = cp.asarray(gr_nodes)

        self._n_internal_nodes = len(self._I)
        self._n_external_nodes = len(self._E)
        self._n_grounded_nodes = len(self._GR)
        self._n_nodes = len(self._W)

        self._G = None

        self.save_conductance = save_conductance
        self.save_dissipated = save_dissipated
        self.save_voltage = save_voltage
        self._G_history = None

        self._state = None
        self.energy_dissipated = None
        self.memristor_voltage = None
        

    def setW(self, w):
        """
        # TODO
        This function guarantees that W is binary and symmetric. Converts
        directed connectivity matrices in undirected.
        Returns a CuPy array

        Parameters
        ----------
        w : (N,N) ndarray (Numpy or CuPy)
            Transformed W matrix
        """

        # convert to binary
        w = w.astype(bool).astype(int)
        w = cp.asarray(w)
        # make sure the diagonal is zero
        cp.fill_diagonal(w, 0)

        # make symmetric if w is directed
        if not utils.check_symmetric(w):
            #Numpy functions are interchangeable with CuPy, thus some of the 
            #missing functions from CuPy are left implemented in Numpy

            # connections in upper diagonal
            upper_diag = w[np.triu_indices_from(w, 1)]

            # connections in lower diagonal
            lower_diag = w.T[np.triu_indices_from(w, 1)]

            # matrix of undirected connections
            W = cp.zeros_like(w).astype(int)
            W[np.triu_indices_from(w, 1)] = cp.logical_or(upper_diag,
                                                          lower_diag
                                                          ).astype(int)

            return utils.make_symmetric(W, copy_lower=False)

        else:
            return w

    def init_property(self, mean, std=0.1, seed=None):
        """
        This function initializes property matrices following a normal
        distribution with mean = 'mean' and standard deviation = 'mean' * 'std'

        Parameters
        ----------
        # TODO
        mean : float
            The mean to be used in the Gaussian random generation
        std : float
            The standard deviation to be used in the Gaussian random generation.
            Default: 0.1
        seed : int, array_like[ints], SeedSequence, BitGenerator, Generator, optional
            seed to initialize the random number generator, by default None
            for details, see numpy.random.default_rng()

        Returns
        -------
        __name__ : cupy.ndarray
            (N,N) Matrix of size W of random values drawn from Gaussian using mean=mean and std=0.1

        """

        # use random number generator for reproducibility
        rng = np.random.default_rng(seed=seed)

        p = cp.asarray(rng.normal(mean, std*mean, size=self._W.shape))
        p = utils.make_symmetric(p)

        return cp.multiply(p , self._W).astype(cp.float64)  # ma.masked_array(p, mask=np.logical_not(self._W))   

    def solveVi(self, Ve, Vgr=None, G=None, **kwargs):
        """
        This function uses Kirchhoff's law to estimate voltage at the internal
        nodes based on the current state of the conductance matrix G, a given
        external input voltage 'V_E' and the ground level voltage 'V_GR'
        Returns a CuPy array
        
        Parameters
        ----------
        Ve : cupy.ndarray
            An array of the voltages applied to the external nodes
        Vgr : cupy.ndarray
            An array of the voltages applied to the external nodes
            Default: None
        G : cupy.ndarray
            A matrix of conductance values for each memristor in the network        

        Returns
        -------
        x : cupy.ndarray
            An array of the voltages at each of the internal nodes (NOT MEMRISTORS)

        """

        #Sets the ground voltage to 0 if None was passed
        if Vgr is None:
            Vgr = cp.zeros((self._n_grounded_nodes))
        if G is None:
            G = self._G

        # TODO: verify that the axis along which the sum is performed is correct
        # matrix N
        N = cp.zeros((self._n_nodes, self._n_nodes))
        cp.fill_diagonal(N, cp.sum(G, axis=1))
            
        # matrix A
        A = N - G

        # inverse matrix A_II
        #LOGIC IS that A is the conductance at each node, thus inverse is Resistance
        A_II = A[cp.ix_(self._I, self._I)]
        A_II_inv = pinv(A_II)

        # matrix HI* finds the current at each node
        H_IE = cp.dot(G[cp.ix_(self._I, self._E)], Ve)
        H_IGR = cp.dot(G[cp.ix_(self._I, self._GR)], Vgr)

        H_I = H_IE + H_IGR

        # return voltage at internal nodes (RESISTANCE * CURRENT)
        return cp.dot(A_II_inv, H_I)

    def getV(self, Vi, Ve, Vgr=None,reverse = False):
        """
        Given the nodal voltage at the internal, external and grounded
        nodes, this function estimates the voltage across all existent
        memristive elements
        Returns a CuPy array

        Parameters
        ----------
        Vi : cupy.ndarray
            Array of voltages across all internal nodes. Calculated 
            with solveVi

        Ve : cupy.ndarray
            Array of voltages passed through all the external nodes

        Vgr : cupy.ndarray
            Array of voltages across all ground nodes.
            Default: None 

        reverse : bool
            boolean controls whether singla flows left to right or right to left
            Default:False (Left to right)   

        Returns
        -------
        V : (N,N) cupy.ndarray
            Matrix of voltage across every memristor, ie voltage across
            every connection in the connection matrix W.

        """

        # set voltage at grounded nodes
        if Vgr is None:
            Vgr = cp.zeros((self._n_grounded_nodes))

        # set of all nodal voltages
        #How does it make sure the correct voltages are paired with each correct node
        #ANSWER: In the below nodes concat, the order of nodes is the same as voltages, thus the nv_dict
        #        is a pairing of (node_index[in W] , voltage)
        voltage = cp.concatenate(
            [Vi[:, cp.newaxis], Ve[:, cp.newaxis], Vgr[:, cp.newaxis]]).squeeze()

        # set of all nodes (internal + external + grounded)
        nodes = cp.concatenate(
            [self._I[:, cp.newaxis], self._E[:, cp.newaxis], self._GR[:, cp.newaxis]]).squeeze()

        # voltage across memristors
        #Goes across each connection (non-zero in W) and find the voltage using kirchoffs law 
        #loops through all non-zero (i,j) positions in W and sets that "Connection Voltage" as the difference
        #between individual voltages, stored in the nv_dict -> (node_index, voltage)
        #This difference in voltage (Voltage Drop) at (i,j) is stored in V at position (i,j)

        V = cp.zeros_like(self._W).astype(cp.float64)
        i ,j = cp.where(self._W!=0)

        nv_i = cp.empty(len(i))
        i_u = cp.unique(i)
        nv_j = cp.empty(len(j))
        j_u = cp.unique(j)

        for x in i_u:
            nv_i[cp.where(i==x)] = voltage[cp.where(nodes==x)]

        for y in j_u:
            nv_j[cp.where(j==y)] = voltage[cp.where(nodes==y)]

        if reverse:
            V[i,j] = cp.where(j>i,nv_j-nv_i,nv_i-nv_j)
        else:
            V[i,j] = cp.where(j>i,nv_i-nv_j,nv_j-nv_i)

        #removes voltage values from V in positions where there is no connection in W 
        return self.mask(V)  
  

    def simulate(self, Vext, ic=None, mode='forward',ret_int_only=False,return_nodes=None,reverse=False, dt = 1e-4):
        """
        Simulates the dynamics of a memristive reservoir given an external
        voltage signal V_E

        Parameters
        ----------
        Vext : (time, N_external_nodes) numpy.ndarray
            External voltage signal
            N_external_nodes: number of external (input) nodes
        ic : (N_internal_nodes,) numpy.ndarray
            Initial conditions
            N_internal_nodes: number of internal (output) nodes
        mode : {'forward', 'backward'}
            Refers to the method used to solve the system of equations.
            Use 'forward' for explicit Euler method, and 'backward' for
            implicit Euler method.
        ret_int_only : boolean
            Flag that tells simulate to return either all of the node 
            states, or only the states of the internal nodes
            Default: False
        return_nodes : numpy.ndarray
            Contains the indices of the nodes of which to return the voltage values.
            Default = None
        dt : float
            This is the simulated time for which the signal is propagated into the network.
            i.e voltage x is input at internal nodes for dt amount of time
            Default: 1e-4    

        Returns
        -------
        self._state : (time, N) numpy.ndarray
            activation states of the reservoir; includes all the nodes
            N: number of nodes in the network
        """
        #Forward is explicit WORKING, backward is implicit 
        # initialize reservoir states
        self._state = cp.zeros((len(Vext), self._n_nodes))

        if self.save_dissipated:
            self.energy_dissipated = np.zeros((len(Vext)-1, self._n_nodes, self._n_nodes))

        if self.save_voltage:
            self.memristor_voltage = np.zeros((len(Vext), self._n_nodes,self._n_nodes))

        # initialize array for storing conductance history if needed
        if self.save_conductance:
            self._G_history = np.zeros((len(Vext), self._n_nodes,self._n_nodes))
            power = []

        #moves input to gpu and stacks to match number of input nodes
        Vext = cp.asarray(Vext)
        Vext = cp.hstack([Vext for _ in range(self._n_external_nodes)])
        
        for t, Ve in enumerate(Vext):
            if mode == 'forward':

                # if (t>0) and (t%10 == 0): print(f'\t ----- timestep = {t}')
                # get voltage at internal nodes
                Vi = self.solveVi(Ve)

                # update matrix of voltages across memristors
                V = self.getV(Vi, Ve, reverse=reverse)

                if self.save_voltage:
                    self.memristor_voltage[t] = cp.asnumpy(V)

                # update conductance
                self.updateG(V=V, update=True)

                #calculate energy dissipated between this and previous timestep using power at each step using Trapezoid method
                if(self.save_dissipated):
                    square = cp.square(V)
                    power.append(cp.multiply(square,self._G))
                    if(len(power)>1):
                        self.energy_dissipated[t-1] = cp.multiply(cp.divide(cp.add(power[0],power[1]),2),dt).get()
                        power.pop(0)
                
            elif mode == 'backward':

                if (t > 0) and (t % 100 == 0):
                    print(f'\t ----- timestep = {t}')

                # get voltage at internal nodes
                Vi = self.iterate(Ve)

            self._state[t, self._E] = Ve
            self._state[t, self._I] = Vi
            # store conductance
            if self.save_conductance:
                self._G_history[t] = cp.asnumpy(self._G)

        #self._state is a (t x N_nodes) matrix which keeps track of node voltage across time
        #This checks if the flag for returning only the internal nodes is set
        if ret_int_only and return_nodes is None: 
            return cp.asnumpy(self._state[:,self._I])
        elif ret_int_only and return_nodes is not None:
            raise ValueError(
                "Only ret_int_only boolean can be true or return_nodes not None, not both"
            )
        elif not ret_int_only and return_nodes is not None:
            return cp.asnumpy(self._state[:,return_nodes])
        else:
            return cp.asnumpy(self._state)

    def iterate(self, Ve, tol=5e-2, iters=100):
        """
        # TODO

        """

        # initial guess for voltage at internal nodes
        Vi = [self.solveVi(Ve=Ve, G=self._G)]

        # initial guess for conductance
        G = [self._G]

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
            # supposedly correct
            G.append(self.updateG(V, G_tmp, update=False))
            # G.append(self.updateG(self.getV(Vi[-1].copy(), Ve), G[-1], update=False))
            # G.append(self.updateG(self.getV(Vi[-1].copy(), Ve), G_tmp, update=False))

            # estimate error
            err_Vi = self.getErr(Vi[-2], Vi[-1])
            err_G = self.getErr(G[-2], G[-1])

            max_err = cp.max((cp.max(err_Vi), cp.max(err_G)))
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

    @abstractmethod
    def dG(self, V, G=None, dt=1e-4, seed=None):
        pass

    @abstractmethod
    def updateG(self, V, G=None, update=False):
        pass

    def updateNodes(self, int_nodes=None, ext_nodes=None, gr_nodes=None):
        if int_nodes is not None:
            self._I = cp.asarray(int_nodes)
            self._n_internal_nodes = len(int_nodes)
        if ext_nodes is not None:
            self._E = cp.asarray(ext_nodes)
            self._n_external_nodes = len(ext_nodes)
        if gr_nodes is not None:
            self._GR = cp.asarray(gr_nodes)
            self._n_grounded_nodes = len(gr_nodes)   

    def getErr(self, x_0, x_1):
        """
        # TODO
        """

        err = 2 * cp.abs(x_1-x_0)/(cp.abs(x_1)+cp.abs(x_0))
        err[np.isnan(err)] = 0.0

        return err

    def mask(self, a):
        """
        This functions converts to zero all entries in matrix 'a' for which there is
        no existent connection in W
        
        Parameters
        ----------
        a : (N,N)  cupy.ndarray, numpy.ndarray
            Matrix to be changed according to connections in W.

        Returns
        -------
        a : cupy.ndarray
            Masked matrix 'a'
        """
        # W = self._W.get()
        a[np.where(self._W == 0)] = 0
        return a

class MSSNetworkCupy(MemristiveReservoirCupy):
    """
    Class that represents a Metastable Switch Memristive network
    (see Nugent and Molter, 2014 for details)

    ...

    Attributes
    ----------
    w : numpy.ndarray
        reservoir's binary connectivity matrix
    I : numpy.ndarray
        indices of internal nodes
    E : numpy.ndarray
        indices of external nodes
    GR : numpy.ndarray
        indices of grounded nodes
    n_internal_nodes : int
        number of internal nodes
    n_external_nodes : int
        number of external nodes
    n_grounded_nodes : int
        number of gorunded nodes
    n_nodes : int
        total number of nodes (internal, external, and ground)
    G : numpy.ndarray
        matrix of conductances
    save_conductance : bool
        Indicates whether to save conductance state after each simulation
        step. If True, then will be stored in self._G_history. This will
        increase memory demands.
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
    # TODO

    """

    # physical parameters of the MMS model
    k = 1.3806503e-23   # Boltzman's constant
    Q = 1.60217646e-19  # electron charge
    Temp = 298.0          # temperature
    b = Q/(k*Temp)
    VT = 1.0/b

    def __init__(self, vA=0.17, vB=0.22, tc=0.32e-3, NMSS=1000000,
                 Woff=0.91e-3, Won=0.87e-2, Nb=200000, noise=0.1, *args, **kwargs):
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
        int_nodes : (n_internal_nodes,) numpy.ndarray
            indexes of internal nodes
            n_internal_nodes: number of internal nodes
        ext_nodes : (n_external_nodes,) numpy.ndarray
            indexes of external nodes
            n_external_nodes: number of external nodes
        gr_nodes : (n_grounded_nodes,) numpy.ndarray
            indexes of grounded nodes
            n_grounded_nodes: number of grounded nodes
        save_conductance : bool, optional
            Indicates whether to save conductance state after each simulation
            step. If True, then will be stored in self._G_history. This will
            increase memory demands. Default: False

        vA : float. Default: 0.17
            vA controls the voltage needed for metastable switch(es) to transition
            from B state to A state

        vB : float. Default: 0.22
            vB controls the voltage needed for metastable switch(es) to transition
            from A state to B state

        tc : float. Default: 0.32e-3

        NMSS : int. Default: 10000
            Number of Metastable switches switches per memristor
                - More switches means memristor is more sensitive to change

        Woff : float. Default: 0.91e-3
            Minimum conductance of memristor

        Won : float. Default: 0.87e-2
            Maximum conductance of memristor

        Nb : int. Default: 2000
            Note: Nb values changed to be uniform * NMSS, since having
            normally distributed values for Nb around mean 2000000 made it so
            that with any voltage passed, Nb for each memristor was consistently
            increasing at each iteration. i.e original starting Nb was too low for
            NMSS 

        noise : float. Default: 0.1

        # TODO
        """
        super().__init__(*args, **kwargs)
        #NOTE: init_property initializes properties from normal distribution vs mask initializes constants


        # self.vA = self.init_property(vA, noise)      # constant
        # self.vB = self.init_property(vB, noise)      # constant
        # self.tc = self.init_property(tc, noise)      # constant
        self.vA = self.mask(cp.full(self._W.shape,vA))
        self.vB = self.mask(cp.full(self._W.shape,vB))
        self.tc = self.mask(cp.full(self._W.shape,tc))
        # self.NMSS = cp.round(self.init_property(NMSS, noise)).astype(cp.float64)    # constant Note: This sets a different number of switches per memristor 
        self.NMSS = self.mask(cp.full(self._W.shape,NMSS))
        # self.Woff = self.init_property(Woff, noise)    # constant
        # self.Won = self.init_property(Won, noise)     # constant
        self.Woff = self.mask(cp.full(self._W.shape,Woff))    # constant
        self.Won = self.mask(cp.full(self._W.shape,Won))     # constant
        self._Ga = self.mask(cp.divide(self.Woff,self.NMSS)) # constant
        self._Gb = self.mask(cp.divide(self.Won,self.NMSS))   # constant

        # self._Nb = cp.round(self.init_property(Nb, noise)).astype(cp.float64)
        self._Nb = (cp.asarray(np.random.default_rng().uniform(low=0.1,high=0.9,size=self._W.shape)) * self.NMSS).astype(int).astype(cp.float64)
        self._G = cp.asarray(self._Nb * (self._Gb - self._Ga) + self.NMSS * self._Ga)

    #Note: dt was prev 1e-4 changed to match AgChalc Memristor 
    def dG(self, V, G=None, dt=1e-4, seed=None):
        """
        # TODO
        This function updates the conductance matrix G given V.
        G represents the conductance for each memristor in the network
        (One per connection in W)

        Parameters
        ----------
        V : (N,N) cupy.ndarray
            matrix of voltages accross memristors
        G : (N,N) cupy.ndarray
            Matrix of conductances across each memristor in the network
        seed : int, array_like[ints], SeedSequence, BitGenerator, Generator, optional
            seed to initialize the random number generator, by default None
            for details, see numpy.random.default_rng()

        Returns
        -------
        dNb : cupy.ndarray
            Matrix representing the number of switches change to or from 
            B state per memristor in the network.
            (+ more switches changed to B state,  - More switches changed to A state)

        References
        ----------

        """

        # set Nb values
        if G is not None:
            Gdiff1 = G - self.NMSS * self._Ga
            Gdiff2 = self._Gb - self._Ga
            Nb = self.mask(cp.divide(Gdiff1,Gdiff2))

        else:
            Nb = self._Nb

        tc_np = self.tc.get()
        # ratio of dt to characterictic time of the device tc
        alpha = cp.asarray(np.divide(dt, tc_np, where=tc_np != 0))

        # compute Pa
        exponent = -1 * (V - self.vA) / self.VT
        Pa = alpha / (1 + cp.exp(exponent))
        # compute Pb
        exponent = -1 * (V + self.vB) / self.VT
        Pb = alpha * (1 - (1 / (1 + cp.exp(exponent))))
        # compute dNb
        Na = self.NMSS - Nb
        Na = cp.asnumpy(Na)
        Nb = cp.asnumpy(Nb)
        # use random number generator for reproducibility
        rng = np.random.default_rng(seed=seed)
        
        Pa[cp.isnan(Pa)] = 0.0
        Pb[cp.isnan(Pb)] = 0.0
        Pa = cp.clip(Pa, 0.0, 1.0)
        Pb = cp.clip(Pb, 0.0, 1.0)

        #calculates number of switches that switch states in each memristor
        #state A to B
        Gab = cp.asarray(rng.binomial(Na.astype(int), Pa.get()))
        #state B to A
        Gba = cp.asarray(rng.binomial(Nb.astype(int), Pb.get()))

        if utils.check_symmetric(self._W):
            Gab = utils.make_symmetric(Gab)
            Gba = utils.make_symmetric(Gba)

        #finds the change in number of B state switches in each memristor
        dNb = (Gab-Gba).astype(cp.float64)

        return dNb

    def updateG(self, V, G=None, update=False):
        """
        This function updates the conductance matrix G
        which represents the conductance across each memristor in
        the network. Does this according to the change in number
        of B state switches in each memristor

        Parameters
        ----------
        V : (N,N) cupy.ndarray
            Matrix of voltages across all memristors in the network
        G : (N,N) cupy.ndarray
            Matrix of conductance across each memristor (connection)
            in the network
            Deafault: None
        update: boolean
            Boolean flag of whether the current conductance matrix should be 
            updated according to the calculated change of conductance or not

        Returns
        -------
        G + dG : (N,N) cupy.ndarray
            returns a copy of the updated conducatance matrix if the update
            flag is false
        """

        if G is None:
            G = self._G


        # compute dG
        dNb = self.dG(V=V, G=G)
        dG = dNb * (self._Gb-self._Ga)

        if update:
            # update Nb
            self._Nb = self._Nb + dNb
            self._Nb = cp.where(self._Nb>self.NMSS,self.NMSS,cp.where(self._Nb<0.0,0.0,self._Nb))
            # update G
            self._G = self._Nb * (self._Gb - self._Ga) + self.NMSS * self._Ga
            # self._G += dG

        else:
            return self._G.copy() + dG  # updated conductance


class MultiTaskMemeristiveReservoirCupy(MemristiveReservoirCupy):
    """
    Class that represents a general Memristive Reservoir

    This is a copy of the above Memristive reservoir, with the changes to incorporate CuPy for computation speedup

    The Idea behind simulating multitasking is to take both sets of input nodes and input signals, stack them, 
    then treat them as a single simulation. For example input_nodes = [nodes_x,nodes_y] input_singal = [x1y1,x2y2,...]
    ...

    Attributes
    ----------
    _W : numpy.ndarray
        reservoir's binary connectivity matrix
    _I : numpy.ndarray
        indices of internal nodes
    _E : numpy.ndarray
        indices of external nodes
    _GR : numpy.ndarray
        indices of grounded nodes
    n_internal_nodes : int
        number of internal nodes
    n_external_nodes : int
        number of external nodes
    n_grounded_nodes : int
        number of gorunded nodes
    n_nodes : int
        total number of nodes (internal, external, and ground)
    G : numpy.ndarray
        matrix of conductances
    save_conductance : bool
        Indicates whether to save conductance state after each simulation
        step. If True, then will be stored in self._G_history. This will
        increase memory demands.
    _state : numpy.ndarray
        reservoir activation states

    Methods
    ----------
    # TODO

    References
    ----------
    # TODO

    """

    def __init__(self, w, int_nodes, ext_nodes, gr_nodes, save_conductance=False, *args, **kwargs):
        """
        Constructor class for Memristive Networks. Memristive networks are an
        abstraction for physical networks of memristive elements.

        Parameters
        ----------
        w : (N, N) numpy.ndarray
            reservoir's binary connectivity matrix
            N: total number of nodes in the network (internal + external
            + grounded nodes)
        int_nodes : (n_internal_nodes,) numpy.ndarray
            indexes of internal nodes
            n_internal_nodes: number of internal nodes
        ext_nodes : (n_tasks,n_external_nodes) numpy.ndarray
            indexes of external nodes, per task (TASK x EXT INDECES)
            n_external_nodes: number of external nodes
        gr_nodes : (n_tasks, n_grounded_nodes) numpy.ndarray
            indexes of grounded nodes per task
            n_grounded_nodes: number of grounded nodes
        save_conductance : bool, optional
            Indicates whether to save conductance state after each simulation
            step. If True, then will be stored in self._G_history. This will
            increase memory demands. Default: False
        """
        #setW now returns a CuPy array
        self._W = self.setW(w)
        #moves all of the arrays to the Device (GPU)
        self._I = cp.asarray(int_nodes)
        self._E = cp.asarray(ext_nodes)
        self._GR = cp.asarray(gr_nodes)

        self._n_internal_nodes = len(self._I)
        self._n_external_nodes = len(self._E.flatten())
        self._n_grounded_nodes = len(self._GR.flatten())
        self._n_nodes = len(self._W)

        self._G = None

        self.save_conductance = save_conductance
        self._G_history = None

        self._state = None
        

    def adjust_matrices(self,Vext, target=None):
        if target is None:
            target = np.min(a.shape[0] for a in Vext)

        adjusted_Vext = []
        for v in Vext:
            t = v.shape[0]
            if t>target:
                adjusted = v[:t,:]
            else:
                adjusted = v

            adjusted_Vext.append(adjusted)

        return adjusted_Vext                

    def simulate(self, Vext, ic=None, mode='forward',ret_int_only=False,return_nodes=None):
        """
        Simulates the dynamics of a memristive reservoir given an external
        voltage signal V_E

        Parameters
        ----------
        Vext : (time, N_external_nodes) numpy.ndarray
            External voltage signal
            N_external_nodes: number of external (input) nodes
        ic : (N_internal_nodes,) numpy.ndarray
            Initial conditions
            N_internal_nodes: number of internal (output) nodes
        mode : {'forward', 'backward'}
            Refers to the method used to solve the system of equations.
            Use 'forward' for explicit Euler method, and 'backward' for
            implicit Euler method.
        ret_int_only : boolean
            Flag that tells simulate to return either all of the node 
            states, or only the states of the internal nodes
            Default: False    
        return_nodes : (n_tasks, output_nodes) numpy.ndarray
            a task x output_indeces array that specifies which nodes should be output for which task

        Returns
        -------
        self._state : (time, N) numpy.ndarray
            activation states of the reservoir; includes all the nodes
            N: number of nodes in the network
        """
        #Forward is explicit WORKING, backward is implicit 
        # Vext = self.adjust_matrices(Vext)
        # Vext = np.hstack([Vext[i] for i in range(len(Vext))])
        

        # initialize reservoir states
        self._state = cp.zeros((len(Vext[0]), self._n_nodes))
        # initialize array for storing conductance history if needed
        if self.save_conductance:
            self._G_history = cp.zeros((len(Vext), self._n_nodes,
                                        self._n_nodes))

        Vext = cp.asarray(Vext)

        #For each set of input data, stacks it n times for number of input nodes for said task
        temp = []
        for x in range(len(self._E)):
            temp.append(cp.hstack([Vext[x] for _ in range(len(self._E[x]))]))

        #stacks (concatenates vertically) the input signal to get a single inputxTime array
        Vext = cp.hstack(temp)

        hold_ext = self._E
        hold_gr = self._G

        #flattens ground and external nodes to treat as single set
        #NOTE: input array indices match with external node indices
        #ex: nodes = [node_x1,node_x2,node_x3,node_y1,node_y2,node_y3]
        #    input = [in_x   , in_x  , in_x  , in_y  , in_y  , in_y  ]
        #                       --- across time ---
        self._E = self._E.flatten()
        self._GR = self._GR.flatten()

        for t, Ve in enumerate(Vext):
            time.sleep(1)
            print(Ve)
            if mode == 'forward':

                # if (t>0) and (t%50 == 0): print(f'\t ----- timestep = {t}')
                print(t)

                # get voltage at internal nodes
                Vi = self.solveVi(Ve)

                # update matrix of voltages across memristors
                V = self.getV(Vi, Ve)

                # update conductance
                self.updateG(V=V, update=True)

            elif mode == 'backward':

                if (t > 0) and (t % 100 == 0):
                    print(f'\t ----- timestep = {t}')

                # get voltage at internal nodes
                Vi = self.iterate(Ve)

            # store activation states
            self._state[t, self._E] = Ve
            self._state[t, self._I] = Vi

            # store conductance
            if self.save_conductance:
                self._G_history[t] = self._G

        #self._state is a (t x N_nodes) matrix which keeps track of node voltage across time

        self._E = hold_ext
        self._G = hold_gr

        #This checks if the flag for returning only the internal nodes is set
        #Returns same format as single task, up to user to take what nodes they need per task
        if ret_int_only and return_nodes is None: 
            return cp.asnumpy(self._state[:,self._I])
        elif ret_int_only and return_nodes is not None:
            raise ValueError(
                "Only ret_int_only boolean can be true or return_nodes not None, not both"
            )
        elif not ret_int_only and return_nodes is not None:
            return [cp.asnumpy(self._state[:,ret]) for ret in return_nodes]
        else:
            return cp.asnumpy(self._state)
        
    def updateNodes(self, int_nodes=None, ext_nodes=None, gr_nodes=None):
        if int_nodes is not None:
            self._I = cp.asarray(int_nodes)
            self._n_internal_nodes = len(int_nodes.flatten())
        if ext_nodes is not None:
            self._E = cp.asarray(ext_nodes)
            self._n_external_nodes = len(ext_nodes.flatten())
        if gr_nodes is not None:
            self._GR = cp.asarray(gr_nodes)
            self._n_grounded_nodes = len(gr_nodes.flatten())     

class MultiTaskMSSNetworkCupy(MultiTaskMemeristiveReservoirCupy):
    k = 1.3806503e-23   # Boltzman's constant
    Q = 1.60217646e-19  # electron charge
    Temp = 298.0          # temperature
    b = Q/(k*Temp)
    VT = 1.0/b

    def __init__(self, vA=0.17, vB=0.22, tc=0.32e-3, NMSS=1000000,
                 Woff=0.91e-3, Won=0.87e-2, Nb=200000, noise=0.1, *args, **kwargs):
        """
        Constructor class for Memristive Networks following the Generalized
        Memristive Switch Model proposed in Nugent and Molter, 2014. Default
        parameter values correspond to an Ag-chalcogenide memristive device
        taken from Nugent and Molter, 2014.

        NOTE: No changes in this class since functions like single task when
        any of these functions are called, all pre-processing done by 
        MultiTaskMSSNetworkCupy

        Parameters
        ----------
        w : (N, N) numpy.ndarray
            reservoir's binary connectivity matrix
            N: total number of nodes in the network (internal + external
            + grounded nodes)
        int_nodes : (n_internal_nodes,) numpy.ndarray
            indexes of internal nodes
            n_internal_nodes: number of internal nodes
        ext_nodes : (n_external_nodes,) numpy.ndarray
            indexes of external nodes
            n_external_nodes: number of external nodes
        gr_nodes : (n_grounded_nodes,) numpy.ndarray
            indexes of grounded nodes
            n_grounded_nodes: number of grounded nodes
        save_conductance : bool, optional
            Indicates whether to save conductance state after each simulation
            step. If True, then will be stored in self._G_history. This will
            increase memory demands. Default: False
        vA : float. Default: 0.17

        vB : float. Default: 0.22

        tc : float. Default: 0.32e-3

        NMSS : int. Default: 10000

        Woff : float. Default: 0.91e-3

        Won : float. Default: 0.87e-2

        Nb : int. Default: 2000

        noise : float. Default: 0.1

        # TODO
        """
        super().__init__(*args, **kwargs)

        self.vA = self.init_property(vA, noise)      # constant
        self.vB = self.init_property(vB, noise)      # constant
        self.tc = self.init_property(tc, noise)      # constant
        self.NMSS = cp.round(self.init_property(NMSS, noise)).astype(cp.float64)    # constant Note: This sets a different number of switches per memristor 
        # self.NMSS = self.mask(self.NMSS)
        self.Woff = self.init_property(Woff, noise)    # constant
        self.Won = self.init_property(Won, noise)     # constant
        self._Ga = self.mask(cp.divide(self.Woff,self.NMSS)) # constant
        self._Gb = self.mask(cp.divide(self.Won,self.NMSS))   # constant

        self._Nb = cp.round(self.init_property(Nb, noise)).astype(cp.float64)
        # self._Nb = cp.where(self.NMSS < self._Nb, self.NMSS, self._Nb)
        self._G = cp.asarray(self._Nb * (self._Gb - self._Ga) + self.NMSS * self._Ga)

    def dG(self, V, G=None, dt=1e-4, seed=None):
        """
        # TODO
        This function updates the conductance matrix G given V.
        G represents the conductance for each memristor in the network
        (One per connection in W)

        Parameters
        ----------
        V : (N,N) cupy.ndarray
            matrix of voltages accross memristors
        G : (N,N) cupy.ndarray
            Matrix of conductances across each memristor in the network
        seed : int, array_like[ints], SeedSequence, BitGenerator, Generator, optional
            seed to initialize the random number generator, by default None
            for details, see numpy.random.default_rng()

        Returns
        -------
        dNb : cupy.ndarray
            Matrix representing the number of switches change to or from 
            B state per memristor in the network.
            (+ more switches changed to B state,  - More switches changed to A state)

        References
        ----------

        """

        # set Nb values
        if G is not None:
            Gdiff1 = G - self.NMSS * self._Ga
            Gdiff2 = self._Gb - self._Ga
            Nb = self.mask(cp.divide(Gdiff1,Gdiff2))

        else:
            Nb = self._Nb

        tc_np = self.tc.get()
        # ratio of dt to characterictic time of the device tc
        alpha = cp.asarray(np.divide(dt, tc_np, where=tc_np != 0))

        # compute Pa
        exponent = -1 * (V - self.vA) / self.VT
        Pa = alpha / (1 + cp.exp(exponent))

        # compute Pb
        exponent = -1 * (V + self.vB) / self.VT
        Pb = alpha * (1 - 1 / (1 + cp.exp(exponent)))

        # compute dNb
        Na = self.NMSS - Nb

        Na = cp.asnumpy(Na)
        Nb = cp.asnumpy(Nb)

        
        # use random number generator for reproducibility
        # rng = np.random.default_rng(seed=seed)
        rng = np.random.default_rng(seed=seed)
        
        Pa[cp.isnan(Pa)] = 0.0
        Pb[cp.isnan(Pb)] = 0.0
        Pa = cp.clip(Pa, 0.0, 1.0)
        Pb = cp.clip(Pb, 0.0, 1.0)
        

        Gab = cp.asarray(rng.binomial(Na.astype(int), self.mask(Pa).get()))
        Gba = cp.asarray(rng.binomial(Nb.astype(int), self.mask(Pb).get()))

        if utils.check_symmetric(self._W):
            Gab = utils.make_symmetric(Gab)
            Gba = utils.make_symmetric(Gba)

        dNb = (Gab-Gba).astype(cp.float64)

        return dNb

    def updateG(self, V, G=None, update=False):
        """
        This function updates the conductance matrix G
        which represents the conductance across each memristor in
        the network.

        Parameters
        ----------
        V : (N,N) cupy.ndarray
            Matrix of voltages across all memristors in the network
        G : (N,N) cupy.ndarray
            Matrix of conductance across each memristor (connection)
            in the network
            Deafault: None
        update: boolean
            Boolean flag of whether the current conductance matrix should be 
            updated according to the calculated change of conductance or not

        Returns
        -------
        G + dG : (N,N) cupy.ndarray
            returns a copy of the updated conducatance matrix if the update
            flag is false
        """

        if G is None:
            G = self._G


        # compute dG
        dNb = self.dG(V=V, G=G)
        dG = dNb * (self._Gb-self._Ga)

        if update:
            # update Nb
            self._Nb = self._Nb + dNb
            self._Nb = cp.where(self._Nb>self.NMSS,self.NMSS,cp.where(self._Nb<0.0,0.0,self._Nb))
            # update G
            self._G = self._Nb * (self._Gb - self._Ga) + self.NMSS * self._Ga
            # self._G += dG

        else:
            return self._G.copy() + dG  # updated conductance

def reservoir(name, **kwargs):
    if name == 'EchoStateNetwork':
        return EchoStateNetwork(**kwargs)
    if name == 'MSSNetwork':
        return MSSNetwork(**kwargs)
