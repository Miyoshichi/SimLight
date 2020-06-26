# -*- coding: utf-8 -*-

"""
Created on May 22, 2020
@author: Zhou Xiang
"""

import math
import numpy as np


def pv(phase, mask=False):
    """
    Calculate the PV(peek to valley) value of a wavefront.

    Args:
        phase:
            Wavefront to be calculated.
    Returns:
        pv:
            The PV value of input wavefront.
    """
    if mask is True:
        x = np.linspace(-1, 1, phase.shape[0])
        X, Y = np.meshgrid(x, x)
        R = np.sqrt(X**2 + Y**2)
        phase[R > 1] = np.nan

    pv = np.nanmax(phase) - np.nanmin(phase)
    return pv


def rms(phase, mask=False):
    """
    Calculate the RMS(root mean square) value of a wavefront.

    Args:
        phase:
            Wavefront to be calculated.
    Returns:
        pv:
            The RMS value of input wavefront.
    """
    size = phase.size
    if mask is True:
        x = np.linspace(-1, 1, phase.shape[0])
        X, Y = np.meshgrid(x, x)
        R = np.sqrt(X**2 + Y**2)
        phase[R > 1] = np.nan
        size = np.pi * phase.size / 4

    deviation = np.nansum((phase - np.nanmean(phase))**2)
    rms = math.sqrt(deviation / size)
    return rms


def circle_aperature(field, mask_r):
    """
    Filter the circle aperature of a light field.

    Args:
        field:
            Input square field.
        mask_r: float
            Radius of a circle mask (between 0 and 1).
    Returns:
        X:
            Filtered meshgird X.
        Y:
            Filtered meshgrid Y.
    """
    length = field.shape[0]
    norm_length = np.linspace(-1, 1, length)
    X, Y = np.meshgrid(norm_length, norm_length)
    norm_radius = np.sqrt(X**2 + Y**2)
    X[norm_radius > mask_r] = np.nan
    Y[norm_radius > mask_r] = np.nan

    return X, Y, norm_radius
