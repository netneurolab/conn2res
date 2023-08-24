# -*- coding: utf-8 -*-
"""
Functions for basic matrix manipulations
"""
import numpy as np

# consensus
from sklearn.utils.validation import (check_consistent_length, check_array)

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
    try: 
        return np.allclose(a, a.T, atol=tol)
    except ValueError:
        print("Matrix is not square.")
        return False


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
