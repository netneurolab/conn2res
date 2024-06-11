import os

import numpy as np
from scipy import signal

def memory_capacity(n_trials=None, horizon_max=-20, win=30, 
                   low=-1,high=1,input_gain=None,add_bias=False,
                   seed=None, **kwargs):
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

    # use random number generator for reproducibility
    rng = np.random.default_rng(seed=seed)

    # generate input data
    x = rng.uniform(low=low, high=high, size=(n_trials + win + abs_horizon_max + 1))[
        :, np.newaxis
    ]

    # output data
    y = np.hstack([x[win + h : -abs_horizon_max + h - 1] for h in horizon])

    #This extracts the portion of x that will be sliced off the front
    z=x[:win]
    
    # update input data
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

    horizon_max = horizon_max
    # _data = {'x': x, 'y': y}

    return x, y, z


def non_linear_transformation(n_trials=500,n_cycles=10,waveform='square',input_gain=None,add_bias=False, **kwargs):
    print(n_cycles)
    cycle_duration = n_trials/n_cycles
    t = np.arange(n_trials)
    # length = np.pi * 2.0 * cycle_duration
    # sins = np.arange(0,length+length/n_trials, length/n_trials)
    # print(sins)
    # x = np.sin(sins)[:,np.newaxis]

    x = np.sin(2 * np.pi * t / cycle_duration)[:,np.newaxis]

    if input_gain is not None:
        x *= input_gain

    if add_bias:
        x = np.hstack((np.ones((n_trials, 1)), x))

    if waveform == 'sawtooth':
        y = signal.sawtooth(x)
        return x,y,None
    
    y = signal.square(x)
    return x,y,None

    
