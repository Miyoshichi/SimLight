# -*- coding: utf-8 -*-

"""
Created on May 27, 2020
@author: Zhou Xiang
"""

import numpy as np
import unwrap


def simple_unwrap(phase):
    """A simple method to unwrap the input 2D phase.

    A simple method to unwrap the input 2D phase using numpy.unwrap().
    This method is just suitable for the phase without any noise,
    like which is calculated by numpy.angle(), etc.

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


def __DFS(M, phase, m, n, s):

    def v(x):
        return np.arctan2(np.sin(x), np.cos(x))

    stack = []
    stack.append([m, n])
    M[m, n] = 2
    unwrapped_phase = np.zeros([s, s])

    while (len(stack) != 0):
        [m, n] = stack[-1]
        if (m + 1 < s and n < s and M[m+1, n] == 1 and M[m+1, n] != 0 and
                M[m+1, n] != 2):
            m = m + 1
            M[m, n] = 2
            stack.append([m, n])
            unwrapped_phase[m, n] = (unwrapped_phase[m-1, n] +
                                     v(phase[m, n] - phase[m-1, n]))

        elif (m - 1 > 0 and n < s and M[m-1, n] == 1 and M[m-1, n] != 0 and
              M[m-1, n] != 2):
            m = m - 1
            M[m, n] = 2
            stack.append([m, n])
            unwrapped_phase[m, n] = (unwrapped_phase[m+1, n] +
                                     v(phase[m, n] - phase[m+1, n]))

        elif (m < s and n + 1 < s and M[m, n+1] == 1 and M[m, n+1] != 0 and
              M[m, n+1] != 2):
            n = n + 1
            M[m, n] = 2
            stack.append([m, n])
            unwrapped_phase[m, n] = (unwrapped_phase[m, n-1] +
                                     v(phase[m, n] - phase[m, n-1]))

        elif (m < s and n - 1 > 0 and M[m, n-1] == 1 and M[m, n-1] != 0 and
              M[m, n-1] != 2):
            n = n - 1
            M[m, n] = 2
            stack.append([m, n])
            unwrapped_phase[m, n] = (unwrapped_phase[m, n+1] +
                                     v(phase[m, n] - phase[m, n+1]))

        else:
            stack.pop()

    return unwrapped_phase


def dfs_unwrap(phase):
    M = np.ones(phase.shape)
    s = phase.shape[0]

    start_pixel = np.where(M == 1)
    m = start_pixel[0][0]
    n = start_pixel[1][0]

    unwrapped_phase = __DFS(M, phase, m, n, s)

    return unwrapped_phase


def pypi_unwrap(phase):
    unwrapped_phase = unwrap.unwrap(phase,
                                    wrap_around_axis_0=False,
                                    wrap_around_axis_1=False,
                                    wrap_around_axis_2=False)

    return unwrapped_phase
