# -*- coding: utf-8 -*-

"""
Created on May 27, 2020
@author: Zhou Xiang
"""

import numpy as np


def simple_unwrap(phase):
    """A simple method to unwrap the input 2D phase.

    A simple method to unwrap the input 2D phase using numpy.unwrap().
    This method is just suitable for the phase without any noise,
    like which is calculated by numpy.angel(), etc.

    Parameters
    ----------
        phase : array-like
            Wrapped phase.

    Returns
    -------
        unwrap_phase : array-like
            Unwrapped phase.
    """
    unwrapped_phase = np.unwrap(np.unwrap(phase), axis=0)
    return unwrapped_phase


def simple_unwrap_1d(phase):
    unwrapped_phase = np.unwrap(phase)
    return unwrapped_phase
