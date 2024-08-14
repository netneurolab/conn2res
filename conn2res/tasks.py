# -*- coding: utf-8 -*-
"""
Functionality for fetching task datasets
"""
from abc import ABCMeta, abstractmethod
import numpy as np
import neurogym as ngym
from reservoirpy import datasets
from . import datasetsConn2Res

NEUROGYM_TASKS = [
    'AntiReach',
    # 'Bandit',  # *
    'ContextDecisionMaking',
    # 'DawTwoStep',  # *
    'DelayComparison',
    'DelayMatchCategory',
    'DelayMatchSample',
    'DelayMatchSampleDistractor1D',
    'DelayPairedAssociation',
    # 'Detection',  # *
    'DualDelayMatchSample',
    # 'EconomicDecisionMaking',  # *
    'GoNogo',
    'HierarchicalReasoning',
    'IntervalDiscrimination',
    'MotorTiming',
    'MultiSensoryIntegration',
    # 'Null',
    'OneTwoThreeGo',
    'PerceptualDecisionMaking',
    'PerceptualDecisionMakingDelayResponse',
    'PostDecisionWager',
    'ProbabilisticReasoning',
    'PulseDecisionMaking',
    'Reaching1D',
    'Reaching1DWithSelfDistraction',
    'ReachingDelayResponse',
    'ReadySetGo',
    'SingleContextDecisionMaking',
    'SpatialSuppressMotion',
    # 'ToneDetection'  # *
]

RESERVOIRPY_TASKS = [
    'henon_map',
    'logistic_map',
    'lorenz',
    'mackey_glass',
    'multiscroll',
    'doublescroll',
    'rabinovich_fabrikant',
    'narma',
    'lorenz96',
    'rossler'
]

CONN2RES_TASKS = [
    'MemoryCapacity',
    'memory_capacity',
    'non_linear_transformation',
]


class Task(metaclass=ABCMeta):
    """
    Class for generating task datasets

    Parameters
    ----------
    name : str
        name of the task
    n_trials : int, optional
        number of trials if task indicated by 'name' is a
        a trial-based task, by default 10
    """

    def __init__(self, name, n_trials=10):
        """
        Constructor method for class Task
        """
        self.name = name
        self.n_trials = n_trials
        self.n_targets = None
        self.n_features = None

    @property
    @abstractmethod
    def name(self):
        pass

    @name.setter
    @abstractmethod
    def name(self, name):
        pass

    @abstractmethod
    def fetch_data(self, n_trials=None, **kwargs):
        pass


class NeuroGymTask(Task):
    """
    Class for generating task datasets from the
    Neurogym repository

    Parameters
    ----------
    name : str
        name of the task
    n_trials : int, optional
        number of trials if task indicated by 'name' is a
        a trial-based task, by default 10
    """

    def __init__(self, *args, **kwargs):
        """
            Constructor method for class NeuroGymTask
        """
        super().__init__(*args, **kwargs)
        self.timing = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name not in NEUROGYM_TASKS:
            raise ValueError("Task not included in NeuroGym tasks")

        self._name = name

    def fetch_data(self, n_trials=None, input_gain=None, add_bias=False,
                   **kwargs):
        """
        Fetch data for Neurogym tasks

        Parameters
        ----------
        n_trials : int, optional
            number of trials to be generated, by default None
        input_gain : float, optional
            gain on the input signal, i.e., scalar multiplier, by default None
        add_bias : bool, optional
            decides whether bias is added to the input signal or not,
            by default False

        Returns
        -------
        x, y : numpy.ndarray, list
            input (x) and output (y) training data
        """
        if n_trials is not None:
            self.n_trials = n_trials

        # create a Dataset object from NeuroGym
        dataset = ngym.Dataset(self._name + '-v0', kwargs)

        # get environment object
        env = dataset.env

        # generate per trial dataset
        _ = env.reset()

        x, y = [], []
        for _ in range(self.n_trials):
            env.new_trial()
            ob, gt = env.ob, env.gt

            # reshape data if needed
            if ob.ndim == 1:
                ob = ob[:, np.newaxis]
            if gt.ndim == 1:
                gt = gt[:, np.newaxis]

            # scale input data
            if input_gain is not None:
                ob *= input_gain

            # add bias to input data if needed
            if add_bias:
                ob = np.hstack((np.ones((n_trials, 1)), ob))

            # store input and output
            x.append(ob)
            y.append(gt)

        # set attributes
        if x[0].squeeze().ndim == 1:
            self.n_features = 1
        elif x[0].squeeze().ndim == 2:
            self.n_features = x[0].shape[1]

        if y[0].squeeze().ndim == 1:
            self.n_targets = 1
        elif y[0].squeeze().ndim == 2:
            self.n_targets = y[0].shape[1]

        self.timing = env.timing
        # self._data = {'x': x, 'y': y}

        return x, y


class ReservoirPyTask(Task):
    """
    Class for generating task datasets from the
    ReservoirPy repository

    Parameters
    ----------
    name : str
        name of the task
    n_trials : int, optional
        number of trials if task indicated by 'name' is a
        a trial-based task, by default 10
    """

    def __init__(self, *args, **kwargs):
        """
            Constructor method for class NeuroGymTask
        """
        super().__init__(*args, **kwargs)
        self.horizon = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name not in RESERVOIRPY_TASKS:
            raise ValueError("Task not included in ReservoirPy tasks")

        self._name = name

    def fetch_data(self, n_trials=None, horizon=1, win=30,
                   input_gain=None, add_bias=False, **kwargs):
        """
        Fetch data for ReservoirPy tasks, which are defined as single-
        or multi-output tasks using chaotic time series as input signal
        and preceded or delayed output signal(s) with respect to the input

        Parameters
        ----------
        n_trials : int, optional
            number of time steps in input and output, by default None
        horizon : int, numpy.ndarray or list, optional
            shift between input and output, i.e., positive number for
            prediction and negative number for memory task, by default 1
            note that array/list is used for multi-output task
        win : int, optional
            initial window of the input signal to be used for generating the
            delayed output signal in case of memory tasks, by default 30
            note that no values in horizon should exceed this window (in
            absolute value), otherwise ValueError is thrown
        input_gain : float, optional
            gain on the input signal, i.e., scalar multiplier, by default None
        add_bias : bool, optional
            decides whether bias is added to the input signal or not,
            by default False

        Returns
        -------
        x, y : numpy.ndarray, list
            input (x) and output (y) training data

        Raises
        ------
        ValueError
            if horizon has elements with different sign
        ValueError
            if any horizon exceeds win (in absolute value)
        """
        if n_trials is not None:
            self.n_trials = n_trials

        # make sure horizon is a list. Exclude 0.
        if isinstance(horizon, (int, np.integer)):
            horizon = [horizon]

        # check that horizon has elements with same sign
        if np.unique(np.sign(horizon)).size != 1:
            raise ValueError("Horizon sohuld have elements with same sign")

        # calculate absolute maximum horizon
        abs_horizon_max = np.max(np.abs(horizon))
        if win < abs_horizon_max:
            raise ValueError("Absolute maximum horizon should be within window")

        # generate input data
        env = getattr(datasets, self._name)
        x = env(n_timesteps=self.n_trials + win + abs_horizon_max + 1, **kwargs)

        # output data
        y = np.hstack([x[win + h : -abs_horizon_max + h - 1] for h in horizon])

        # update input data
        z = x[:win]
        x = x[win : -abs_horizon_max - 1]

        # reshape data if needed
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if y.ndim == 1:
            y = y[:, np.newaxis]

        # scale input data
        if input_gain is not None:
            x *= input_gain

        # add bias to input data if needed
        if add_bias:
            x = np.hstack((np.ones((n_trials, 1)), x))

        # set attributes
        if x.squeeze().ndim == 1:
            self.n_features = 1
        elif x.squeeze().ndim == 2:
            self.n_features = x.shape[1]

        if y.squeeze().ndim == 1:
            self.n_targets = 1
        elif y.squeeze().ndim == 2:
            self.n_targets = y.shape[1]

        self.horizon = horizon
        # self._data = {'x': x, 'y': y}

        return x, y,z


class Conn2ResTask(Task):
    """
    Class for generating datasets for MemoryCapacity task

    Parameters
    ----------
    name : str
        name of the task
    n_trials : int, optional
        number of trials if task indicated by 'name' is a
        a trial-based task, by default 10
    """

    def __init__(self, *args, **kwargs):
        """
            Constructor method for class Conn2ResTask
        """
        super().__init__(*args, **kwargs)
        self.horizon_max = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name not in CONN2RES_TASKS:
            raise ValueError("Task not included in conn2res tasks")

        self._name = name

    def fetch_data(self, n_trials=None, horizon_max=-20, win=30, 
                   input_gain=None, add_bias=False,**kwargs):
        """
        Fetch data for MemoryCapacity, which is defined as a multi-output
        task using a uniformly distributed input signal and multiple
        delayed output signals

        Parameters
        ----------
        n_trials : int, optional
            number of time steps in input and output, by default None
        horizon_max : int, optional
            maximum shift between input and output, i.e., negative number 
            for memory capacity task, by default -20
            note that an array of horizons are generated from -1 to 
            inclusive of horizon_max using a step of -1, which 
            defines memory capacity task as a multi-output task, i.e., one 
            task per horizon
        win : int, optional
            initial window of the input signal to be used for generating the
            delayed output signal, by default 30
            note that horizon_max should exceed this window (in
            absolute value), otherwise ValueError is thrown
        low : float, optional
            lower boundary of the output interval of numpy.uniform(),
            by default -1
        high : float, optional
            upper boundary of the output interval of numpy.uniform(),
            by default 1
        input_gain : float, optional
            gain on the input signal, i.e., scalar multiplier, by default None
        add_bias : bool, optional
            decides whether bias is added to the input signal or not,
            by default False
        seed : int, array_like[ints], SeedSequence, BitGenerator, Generator, optional
            seed to initialize the random number generator, by default None
            for details, see numpy.random.default_rng()

        Returns
        -------
        x, y : numpy.ndarray, list
            input (x) and output (y) training data

        Raises
        ------
        ValueError
            if maximum horizon exceeds win (in absolute value)
        """
        if n_trials is not None:
            self.n_trials = n_trials

        # generate horizon as a list inclusive of horizon_max
        sign_ = np.sign(horizon_max)
        horizon = np.arange(
            sign_,
            sign_ + horizon_max,
            sign_,
        )

        # calculate absolute maximum horizon
        abs_horizon_max = np.abs(horizon_max)
        if win < abs_horizon_max:
            raise ValueError("Absolute maximum horizon should be within window")

        env = getattr(datasetsConn2Res, self._name)
        x,y,z = env(self.n_trials,input_gain=input_gain, add_bias=add_bias, **kwargs)

        # set attributes
        if x.squeeze().ndim == 1:
            self.n_features = 1
        elif x.squeeze().ndim == 2:
            self.n_features = x.shape[1]

        if y.squeeze().ndim == 1:
            self.n_targets = 1
        elif y.squeeze().ndim == 2:
            self.n_targets = y.shape[1]

        self.horizon_max = horizon_max
        # self._data = {'x': x, 'y': y}

        return x, y, z


def get_task_list(repository):
    """
    Returns list of tasks in repository

    Parameters
    ----------
    repository : str
        _description_

    Returns
    -------
    _type_
        _description_
    """
    repository = repository.lower()
    if repository == 'neurogym':
        return NEUROGYM_TASKS
    elif repository == 'reservoirpy':
        return RESERVOIRPY_TASKS
    elif repository == 'conn2res':
        return CONN2RES_TASKS