# -*- coding: utf-8 -*-

"""
Created on May 22, 2020
@author: Zhou Xiang
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import SimLight as sl
from .calc import phase


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


def longitude_to_wavefront(lens, longitude, wavelength=0.550):
    """
    Covert the longitudinal spherical aberration function to sidel
    coefficients.

    Args:
        lens: tuple
            The lens with spherical aberration.
        longitude: list
            The longitudinal spherical aberration funciton of the lens.
        wavelength: float
            The wavelenth of used light.
    Returns:
        field: tuple
            The light field after passing through the lens.
    Yeild:
        Two figures of longitudinal aberration and wavefront.
    """
    D = lens.D
    f = lens.f
    mask_r = 1
    N = len(longitude)
    center = int(N / 2)
    k = 2 * np.pi / wavelength
    if (np.abs(np.max(longitude[center, center:])) >
            np.abs(np.min(longitude[center, center:]))):
        l_max_value = np.abs(np.max(longitude[center, center:]) * 2.5)
    else:
        l_max_value = np.abs(np.min(longitude[center, center:]) * 2.5)

    x = np.linspace(-D / 2, D / 2, N)
    X, Y = np.meshgrid(x, x)
    rho = np.sqrt(X**2 + Y**2) / (D / 2)

    delta_w = longitude * D / -f**2
    delta_W = delta_w / (wavelength * 1e-3 / (2 * np.pi))
    W = delta_W * rho / 4
    varphi = W * k

    field = sl.PlaneWave(wavelength, D, N)
    field.complex_amp *= np.exp(-1j * varphi)
    phase_ = phase(field, unwrap=True)
    phase_ = wavelength * phase_ / (2 * np.pi)
    _, _, norm_radius = circle_aperature(phase_, mask_r)
    w_max_value = np.max(phase_[norm_radius <= mask_r])
    w_min_value = np.min(phase_[norm_radius <= mask_r])
    PV = 'P-V: ' + str(round(pv(phase_, mask=True), 3)) + ' λ'
    RMS = 'RMS: ' + str(round(rms(phase_, mask=True), 3)) + ' λ'

    fig = plt.figure(figsize=(9, 6))
    grid = plt.GridSpec(8, 8, wspace=0.5, hspace=0.5)
    ax1 = fig.add_subplot(grid[0:6, 0:2])
    ax2 = fig.add_subplot(grid[0:6, 3:8])

    im1 = ax1.plot(longitude[center, center:], x[center:])
    ax1.spines['top'].set_color('none')
    ax1.spines['bottom'].set_position(('data', 0))
    ax1.spines['left'].set_position(('data', 0))
    ax1.spines['right'].set_color('none')
    ax1.set_xlim((-l_max_value, l_max_value))
    ax1.set_xlabel('Longitudinal aberration [mm]')
    ax1.set_ylabel('Size [mm]', rotation=0, position=(0, 1.01), labelpad=-20)

    extent = [-D / 2, D / 2, -D / 2, D / 2]
    im2 = ax2.imshow(phase_, cmap='rainbow', extent=extent,
                     vmin=w_min_value, vmax=w_max_value)
    mask = patches.Circle([0, 0], D * mask_r / 2, fc='none', ec='none',)
    ax2.add_patch(mask)
    im2.set_clip_path(mask)
    ax2.text(0.05, 0.95, PV, fontsize=12, horizontalalignment='left',
             transform=ax2.transAxes)
    ax2.text(0.05, 0.90, RMS, fontsize=12, horizontalalignment='left',
             transform=ax2.transAxes)
    plt.colorbar(im2, ax=ax2)

    plt.show()

    return field
