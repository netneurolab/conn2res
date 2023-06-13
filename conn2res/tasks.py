print("----------------here----------------------")


class Task():
    """
    Class that represents a general Task

    Attributes
    ----------
    name : numpy.ndarray
        reservoir connectivity matrix (source, target)

    Methods
    -------
    get_last_name

    """

    def __init__(self, name):
        self.name = name

    def get_last_name(self, last_name):
        return last_name


class NeuroGymTask(Task):
    def __init__(self, name):
        pass


class Reservoirpy(Task):
    def __init__(self, name):
        pass


class Conn2ResTask(Task):
    def __init__(self, name):
        pass

# # -*- coding: utf-8 -*-
# """
# Task class

# @author: Estefany Suarez
# """

# from abc import ABCMeta, abstractmethod
# import numpy as np
# import neurogym as ngym
# from reservoirpy import datasets


# NEUROGYM_TASKS = [
#     'AntiReach',
#     'Bandit',  # *
#     'ContextDecisionMaking',
#     'DawTwoStep',  # *
#     'DelayComparison',
#     'DelayMatchCategory',
#     'DelayMatchSample',
#     'DelayMatchSampleDistractor1D',
#     'DelayPairedAssociation',
#     'Detection',  # *
#     'DualDelayMatchSample',
#     'EconomicDecisionMaking',  # *
#     'GoNogo',
#     'HierarchicalReasoning',
#     'IntervalDiscrimination',
#     'MotorTiming',
#     'MultiSensoryIntegration',
#     # 'Null',
#     'OneTwoThreeGo',
#     'PerceptualDecisionMaking',
#     'PerceptualDecisionMakingDelayResponse',
#     'PostDecisionWager',
#     'ProbabilisticReasoning',
#     'PulseDecisionMaking',
#     'Reaching1D',
#     'Reaching1DWithSelfDistraction',
#     'ReachingDelayResponse',
#     'ReadySetGo',
#     'SingleContextDecisionMaking',
#     'SpatialSuppressMotion',
#     'ToneDetection'  # *
# ]

# RESERVOIRPY_TASKS = [
#     'henon_map',
#     'logistic_map',
#     'lorenz',
#     'mackey_glass',
#     'multiscroll',
#     'doublescroll',
#     'rabinovich_fabrikant',
#     'narma',
#     'lorenz96',
#     'rossler'
# ]

# CONN2RES_TASKS = [
#     'MemoryCapacity'
# ]


# class Task(metaclass=ABCMeta):

#     def __init__(self, name, n_trials=10):
#         self.name = name
#         self.n_trials = n_trials
#         self.n_targets = None
#         self.n_features = None

#     @property
#     @abstractmethod
#     def name(self):
#         pass

#     @name.setter
#     @abstractmethod
#     def name(self, name):
#         pass

#     @abstractmethod
#     def fetch_data(self, n_trials=None, **kwargs):
#         pass


# class NeuroGymTask(Task):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.timing = None

#     @property
#     def name(self):
#         return self._name

#     @name.setter
#     def name(self, name):
#         if name not in NEUROGYM_TASKS:
#             raise ValueError("Task not included in NeuroGym tasks")

#         self._name = name

#     def fetch_data(self, n_trials=None, **kwargs):
#         """
#         _summary_

#         Parameters
#         ----------
#         n_trials : _type_, optional
#             _description_, by default None

#         Returns
#         -------
#         _type_
#             _description_
#         """
#         if n_trials is not None:
#             self.n_trials = n_trials

#         # create a Dataset object from NeuroGym
#         dataset = ngym.Dataset(self._name + '-v0', kwargs)

#         # get environment object
#         env = dataset.env

#         # generate per trial dataset
#         _ = env.reset()

#         x, y = [], []
#         for _ in range(self.n_trials):
#             env.new_trial()
#             ob, gt = env.ob, env.gt

#             # store inputs
#             if ob.ndim == 1:
#                 x.append(ob[:, np.newaxis])
#             else:
#                 x.append(ob)

#             if gt.ndim == 1:
#                 y.append(gt[:, np.newaxis])
#             else:
#                 y.append(gt)

#         # set attributes
#         if x[0].squeeze().ndim == 1:
#             self.n_features = 1
#         elif x[0].squeeze().ndim == 2:
#             self.n_features = x[0].shape[1]

#         if y[0].squeeze().ndim == 1:
#             self.n_targets = 1
#         elif y[0].squeeze().ndim == 2:
#             self.n_targets = y[0].shape[1]

#         self.timing = env.timing
#         # self._data = {'x': x, 'y': y}

#         return x, y


# class ReservoirPyTask(Task):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.horizon = None

#     @property
#     def name(self):
#         return self._name

#     @name.setter
#     def name(self, name):
#         if name not in RESERVOIRPY_TASKS:
#             raise ValueError("Task not included in ReservoirPy tasks")

#         self._name = name

#     def fetch_data(self, n_trials=None, horizon=1, win=30, **kwargs):
#         """
#         _summary_

#         Parameters
#         ----------
#         n_trials : _type_, optional
#             _description_, by default None
#         horizon : int, optional
#             _description_, by default 1
#         win : int, optional
#             _description_, by default 30

#         Returns
#         -------
#         _type_
#             _description_

#         Raises
#         ------
#         ValueError
#             _description_
#         ValueError
#             _description_
#         """
#         if n_trials is not None:
#             self.n_trials = n_trials

#         # make sure horizon is a list. Exclude 0.
#         if isinstance(horizon, (int, np.integer)):
#             horizon = [horizon]

#         # check that horizon has elements with same sign
#         if np.unique(np.sign(horizon)).size != 1:
#             raise ValueError("Horizon sohuld have elements with same sign")

#         # calculate absolute maximum horizon
#         abs_horizon_max = np.max(np.abs(horizon))
#         if win < abs_horizon_max:
#             raise ValueError("Absolute maximum horizon should be within window")

#         # generate input data
#         env = getattr(datasets, self._name)
#         x = env(n_timesteps=self.n_trials + win + abs_horizon_max + 1, **kwargs)

#         # output data
#         y = np.hstack([x[win + h : -abs_horizon_max + h - 1] for h in horizon])

#         # update input data
#         x = x[win : -abs_horizon_max - 1]

#         if x.ndim == 1:
#             x = x[:, np.newaxis]

#         if y.ndim == 1:
#             y = y[:, np.newaxis]

#         # set attributes
#         if x.squeeze().ndim == 1:
#             self.n_features = 1
#         elif x.squeeze().ndim == 2:
#             self.n_features = x.shape[1]

#         if y.squeeze().ndim == 1:
#             self.n_targets = 1
#         elif y.squeeze().ndim == 2:
#             self.n_targets = y.shape[1]

#         self.horizon = horizon
#         # self._data = {'x': x, 'y': y}

#         return x, y


# class Conn2ResTask(Task):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.horizon_max = None

#     @property
#     def name(self):
#         return self._name

#     @name.setter
#     def name(self, name):
#         if name not in CONN2RES_TASKS:
#             raise ValueError("Task not included in conn2res tasks")

#         self._name = name

#     def fetch_data(self, n_trials=None, horizon_max=-20, win=30, **kwargs):
#         """
#         _summary_

#         Parameters
#         ----------
#         n_trials : _type_, optional
#             _description_, by default None
#         horizon_max : _type_, optional
#             _description_, by default None
#         win : int, optional
#             _description_, by default 30

#         Returns
#         -------
#         _type_
#             _description_

#         Raises
#         ------
#         ValueError
#             _description_
#         ValueError
#             _description_
#         """
#         if n_trials is not None:
#             self.n_trials = n_trials

#         # generate horizon as a list inclusive of horizon_max
#         sign_ = np.sign(horizon_max)
#         horizon = np.arange(
#             sign_,
#             sign_ + horizon_max,
#             sign_,
#         )

#         # calculate absolute maximum horizon
#         abs_horizon_max = np.abs(horizon_max)
#         if win < abs_horizon_max:
#             raise ValueError("Absolute maximum horizon should be within window")

#         # generate input data
#         x = np.random.uniform(-1, 1, (self.n_trials + win + abs_horizon_max + 1))[
#             :, np.newaxis
#         ]

#         # output data
#         y = np.hstack([x[win + h : -abs_horizon_max + h - 1] for h in horizon])

#         # update input data
#         x = x[win : -abs_horizon_max - 1]

#         if x.ndim == 1:
#             x = x[:, np.newaxis]

#         if y.ndim == 1:
#             y = y[:, np.newaxis]

#         # set attributes
#         if x.squeeze().ndim == 1:
#             self.n_features = 1
#         elif x.squeeze().ndim == 2:
#             self.n_features = x.shape[1]

#         if y.squeeze().ndim == 1:
#             self.n_targets = 1
#         elif y.squeeze().ndim == 2:
#             self.n_targets = y.shape[1]

#         self.horizon_max = horizon_max
#         # self._data = {'x': x, 'y': y}

#         return x, y
