# -*- coding: utf-8 -*-

"""
Created on May 22, 2020
@author: Zhou Xiang
"""

import math
import numpy as np

import SimLight as sl


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
        field: tuple
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


def zernike_to_sidel(zernike_coefficients):
    """
    Covert Zernike polynomials coefficients to Sidel polynomials
    coefficients.

    Args:
        zernike_coefficients: list
    Returns:
        sidel_coefficients: list
    """
    z = zernike_coefficients
    s = np.zeros((6, 2))

    rad = 180 / np.pi
    # piston
    s[0][0] = z[0] + np.sqrt(3) * z[4] + np.sqrt(5) * z[12]
    # tilt
    s[1][0] = np.sqrt((z[1] - np.sqrt(8) * z[7])**2 +
                      (z[2] - np.sqrt(8) * z[8])**2) * 2
    s[1][1] = np.arctan2(z[1] - np.sqrt(8) * z[7],
                         z[2] - np.sqrt(8) * z[8]) * rad
    # astigmatism
    s[3][0] = 2 * np.sqrt(6 * (z[3]**2 + z[5]**2))
    s[3][1] = 0.5 * np.arctan2(z[3], z[5]) * rad
    # defocus
    s[2][0] = 2 * (np.sqrt(3) * z[4] - 3 * np.sqrt(5) * z[12] - 0.25 * s[3][0])
    # coma
    s[4][0] = 6 * np.sqrt(2 * (z[7]**2 + z[8]**2))
    s[4][1] = np.arctan2(z[7], z[8]) * rad
    # spherical
    s[5][0] = 6 * np.sqrt(5) * z[12]

    sidel_coefficients = s

    return sidel_coefficients


def longitude_to_wavefront(delta_W, h, wavelength=0.550):
    """
    Covert the longitudinal spherical aberration function to sidel
    coefficients.

    Args:
    Returns:
    """
    delta_W /= wavelength * 1e-3 / (2 * np.pi)

    N = len(delta_W)
    x = np.linspace(-1, 1, 2 * N)
    X, Y = np.meshgrid(x, x)
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    sidel = sl.zernike.SidelCoefficients()
    sidel.coefficients[5][0] = delta_W / 4 / rho**3

    return sidel
