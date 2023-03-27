# -*- coding: utf-8 -*-
"""
Task class

@author: Estefany Suarez
"""

from abc import ABCMeta, abstractmethod
import numpy as np
import neurogym as ngym
from reservoirpy import datasets


NEUROGYM_TASKS = [
    'AntiReach',
    'Bandit',  # *
    'ContextDecisionMaking',
    'DawTwoStep',  # *
    'DelayComparison',
    'DelayMatchCategory',
    'DelayMatchSample',
    'DelayMatchSampleDistractor1D',
    'DelayPairedAssociation',
    'Detection',  # *
    'DualDelayMatchSample',
    'EconomicDecisionMaking',  # *
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
    'ToneDetection'  # *
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
    'MemoryCapacity'
]


class Task(metaclass=ABCMeta):

    def __init__(self, name, n_trials=10):
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

    def __init__(self, *args, **kwargs):
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

    def fetch_data(self, n_trials=None, **kwargs):
        """
        _summary_

        Parameters
        ----------
        n_trials : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
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

            # store inputs
            if ob.ndim == 1:
                x.append(ob[:, np.newaxis])
            else:
                x.append(ob)

            if gt.ndim == 1:
                y.append(gt[:, np.newaxis])
            else:
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

    def __init__(self, *args, **kwargs):
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

    def fetch_data(self, n_trials=None, horizon=1, **kwargs):
        """
        _summary_

        Parameters
        ----------
        n_trials : _type_, optional
            _description_, by default None
        horizon : int, optional
            _description_, by default 1

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        if n_trials is not None:
            self.n_trials = n_trials

        # make sure horizon is a list. Exclude 0.
        if isinstance(horizon, int):
            horizon = [horizon]

        # check that horizon has elements with same sign
        horizon_sign = np.unique(np.sign(horizon))
        if horizon_sign.size == 1:
            horizon_sign = horizon_sign[0]
        else:
            raise ValueError('horizon should have elements with same sign')

        # transform horizon elements to positive values (and nd.array)
        horizon = np.abs(horizon)

        # calculate maximum horizon
        horizon_max = np.max(horizon)

        env = getattr(datasets, self._name)
        self.n_trials += horizon_max
        x = env(n_timesteps=self.n_trials, **kwargs)

        y = np.hstack([x[horizon_max-h:-h] for h in horizon])
        x = x[horizon_max:]

        if horizon_sign == 1:
            if y.shape[1] == 1:
                x = y.copy()
                y = x.copy()
            else:
                raise ValueError('positive horizon should be integer not list')

        if x.ndim == 1:
            x = x[:, np.newaxis]

        if y.ndim == 1:
            y = y[:, np.newaxis]

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

        return x, y


class Conn2ResTask(Task):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.horizon = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name not in CONN2RES_TASKS:
            raise ValueError("Task not included in conn2res tasks")

        self._name = name

    def fetch_data(self, n_trials=None, horizon=None, **kwargs):
        """
        _summary_

        Parameters
        ----------
        n_trials : _type_, optional
            _description_, by default None
        horizon : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        if n_trials is not None:
            self.n_trials = n_trials

        # check that horizon has elements with same sign
        horizon_sign = np.unique(np.sign(horizon))
        if horizon_sign.size == 1:
            horizon_sign = horizon_sign[0]
        else:
            raise ValueError('horizon should have elements with same sign')

        # transform horizon elements to positive values (and nd.array)
        horizon = np.abs(horizon)

        # calculate maximum horizon
        horizon_max = np.max(horizon)

        # if task == 'MemoryCapacity':
        x = np.random.uniform(-1, 1, (self.n_trials+horizon_max))[:, np.newaxis]

        y = np.hstack([x[horizon_max-h:-h] for h in horizon])
        x = x[horizon_max:]

        if horizon_sign == 1:
            if y.shape[1] == 1:
                x = y.copy()
                y = x.copy()
            else:
                raise ValueError('positive horizon should be integer not list')

        if x.ndim == 1:
            x = x[:, np.newaxis]

        if y.ndim == 1:
            y = y[:, np.newaxis]

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

        return x, y
