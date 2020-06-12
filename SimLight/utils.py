# -*- coding: utf-8 -*-

"""
Created on May 22, 2020
@author: Zhou Xiang
"""

import math
import numpy as np


def pv(surface):
    """
    Calculate the PV(peek to valley) value of a wavefront.

    Args:
        wavefront:
            Wavefront to be calculated.
    Returns:
        pv:
            The PV value of input wavefront.
    """
    pv = np.nanmax(surface) - np.nanmin(surface)
    return pv


def rms(surface):
    """
    Calculate the RMS(root mean square) value of a wavefront.

    Args:
        wavefront:
            Wavefront to be calculated.
    Returns:
        pv:
            The RMS value of input wavefront.
    """
    deviation = np.nansum((surface - np.nanmean(surface))**2)
    rms = math.sqrt(deviation / surface.size)
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
