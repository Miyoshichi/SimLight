# -*- coding: utf-8 -*-

"""
Created on May 22, 2020
@author: Zhou Xiang
"""

import os
import math
import numpy as np
import matlab
import matlab.engine
import scipy.interpolate
import scipy.io

import SimLight as sl
from .unwrap import simple_unwrap
from .units import *


def phase(field, unwrap=False):
    """
    Calculate the phase of a light field.

    Parameters
    ----------
        field : SimLight.Field
            The light field to be calculated.
        unwrap : bool, optional, {True, False}, default False
            Whether to unwrap the phase.

    Returns
    ---------
        phase : array-like, float
            The phase of the light field.
    """
    if isinstance(field, sl.Field) is True:
        N = field.N
        wavelength = field.wavelength
        phase = np.angle(field.complex_amp)
    elif isinstance(field, np.ndarray) is True:
        N = field.shape[0]
        wavelength = 1 * µm
        phase = np.angle(field)
    else:
        raise ValueError('Invalid light field.')

    if unwrap is True:
        phase = simple_unwrap(phase)

    return phase


def intensity(field, norm_type=1):
    """
    Calculate the intensity of a light field.

    Parameters
    ----------
        field : SimLight.Field
            The light field to be calculated.
        norm_type : int, optional, {0, 1, 2}, default 0
            Type of normalization, where
                0 for no normalization,
                1 for normalize up to 1,
                2 for normalize up to 255.

    Returns
    ----------
        intensity : array-like, float
            The intensity of the light field.
    """
    if isinstance(field, sl.Field) is True:
        intensity = np.abs(field.complex_amp)**2
    elif isinstance(field, np.ndarray) is True:
        intensity = np.abs(field)**2
    else:
        raise ValueError('Invalid light field')

    if norm_type < 0 or norm_type > 2 or type(norm_type) is not int:
        raise ValueError('Unknown normalization type.')
    elif norm_type >= 1:
        intensity /= np.max(intensity)
        if norm_type == 2:
            intensity *= 255

    return intensity


def psf(field, aperture_type='circle'):
    """
    Calculate the point spread function of a light field.

    Parameters
    ----------
        field : SimLight.Field
            The light fiedl.
        aperture_type : str, optional, {'circle', 'square'},
        default 'circle'
            The shape of the aperture.
                circle: circle aperture
                square: square aperture

    Returns
    ----------
        psf : array-like, float
            Point spread function of the input light field.
    """
    field = sl.Field.copy(field)

    N = field.N
    size = field.size
    complex_amp = field.complex_amp
    upper = 0
    lower = N - 1

    size_mag = size / 25.4
    N_mag = N / 100

    if aperture_type is 'circle':
        x = np.linspace(-size / 2, size / 2, N)
        X, Y = np.meshgrid(x, x)
        R = np.sqrt(X**2 + Y**2)
        r = size / 2
        complex_amp[R >= r] = 0

    psf_N = int(N * (N_mag / size_mag) / 2) * 2
    if psf_N > N:
        complex_amp_bigger = np.zeros([psf_N, psf_N], dtype=complex)
        upper = int((psf_N - N) / 2)
        lower = upper + N
        complex_amp_bigger[upper:lower, upper:lower] = complex_amp
        complex_amp = complex_amp_bigger
        N = psf_N
        size *= N_mag / size_mag

    psf = np.abs(np.fft.fftshift(np.fft.fft2(complex_amp)))**2
    psf /= np.max(psf)
    psf = psf[upper:lower, upper:lower]

    return psf


def aberration(field, zernike):
    """
    Return a aberrated light field due to the input Zernike cofficients.

    Parameters
    ----------
        field : SimLight.Field
            The light field to be calculated.
        zernike : SimLight.Zernike
            The Zernike Polynomials.

    Returns
    ----------
        aberrated_field : SimLight.Field
            The aberrated light field.
    """
    field = sl.Field.copy(field)

    N = field.N
    # size = field.size
    k = 2 * np.pi / (field.wavelength * 1e6)
    n = zernike.n
    m = zernike.m
    norm = zernike.norm
    m_abs = abs(m)
    j = zernike.j
    coeff = zernike.coefficients

    # x = np.linspace(-size, size, N)
    # x = np.linspace(-size / 25.4, size / 25.4, N)
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    # R_(n, m)(rho) = sum(k = 0 -> (n - m)/2): r_(n, m)(k) * rho(n, k)
    n_minus_m_half = (n - m_abs) / 2
    n_plus_m_half = (n + m_abs) / 2
    r = np.zeros((j, int(max(n_minus_m_half)) + 1))
    rho_exp = np.zeros((j, int(max(n_minus_m_half)) + 1), dtype=int)
    R = np.zeros((j, N, N))
    for i in range(j):
        for ii in range(int(n_minus_m_half[i]) + 1):
            r[i][ii] = ((-1)**ii * math.factorial(n[i] - ii)) /\
                (math.factorial(ii) * math.factorial(n_plus_m_half[i] - ii) *
                 math.factorial(n_minus_m_half[i] - ii))
            rho_exp[i][ii] = n[i] - 2 * ii
            R[:][:][i] = R[:][:][i] + r[i][ii] * (rho**(rho_exp[i][ii]))
    # Z_(n, m)(j) = R_(n, m)(rho) * cos(m * theta) or sin(m * theta)
    Z = np.zeros((j, N, N))
    for i in range(j):
        if m[i] < 0:
            Z[:][:][i] = R[:][:][i] * np.sin(m_abs[i] * theta)
        else:
            Z[:][:][i] = R[:][:][i] * np.cos(m_abs[i] * theta)
    # W(y, x) = zernike_coeff * Z
    phi = np.zeros((N, N))
    for i in range(j):
        phi = phi + coeff[i] * Z[:][:][i] * norm[i]

    varphi = -k * phi
    field.complex_amp *= np.exp(1j * varphi)

    return field


def sidel_aberration(field, sidel):
    """
    Return a aberrated light field due to the input Sidel cofficients.

    Parameters
    ----------
        field : SimLight.Field
            The light field to be calculated.
        sidel : SimLight.Sidel
            The Sernike Polynomials.

    Returns
    ----------
        aberrated_field : SimLight.Field
            The aberrated light field.
    """
    field = sl.Field.copy(field)

    N = field.N
    k = 2 * np.pi / field.wavelength
    W = sidel.coefficients

    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    h = 1
    rad = np.pi / 180

    piston = W[0][0] * h**2
    tilt = W[1][0] * h * rho * np.cos(theta - W[1][1] * rad)
    defocus = W[2][0] * rho**2
    astigmatism = W[3][0] * h**2 * rho**2 * np.cos(theta - W[3][1] * rad)**2
    coma = W[4][0] * h * rho**3 * np.cos(theta - W[4][1] * rad)
    spherical = W[5][0] * rho**4
    surface = piston + tilt + defocus + astigmatism + coma + spherical

    varphi = k * surface
    field.complex_amp *= np.exp(-1j * varphi)

    return field


def zernike_coeffs(field, j, **kwargs):
    """
    Return the Zernike coefficients of wavefront of a light field.

    Parameters
    ----------
        field : SimLight.Field
            A light field.
        j : int
            Order of Zernike polynomials.

    Returns
    ----------
        coeffs : array-like, float
            Zernike coefficients.
    """
    module_dir = os.path.dirname(sl.__file__)
    os.chdir(module_dir)

    if isinstance(field, sl.Field) is True:
        field = sl.Field.copy(field)
        wavelength = field.wavelength
        wavefront = phase(field, unwrap=True)
    else:
        wavelength = kwargs['wavelength']
        wavefront = field / wavelength * (2 * np.pi) * µm

    # size = field.size
    # N = field.N

    # x = np.linspace(-size / 2, size / 2, N)
    # X, Y = np.meshgrid(x, x)
    # theta, R = cart2pol(X, Y)

    n, m, _ = sl.zernike.ZernikeCoefficients.order(j)

    wavefront = wavefront.tolist()
    wavefront = matlab.double(wavefront)
    n = n.tolist()
    n = matlab.double(n)
    m = m.tolist()
    m = matlab.double(m)

    ml = matlab.engine.start_matlab()
    coeffs = ml.zernike_coeff(wavefront, wavelength, n, m)
    coeffs = np.asarray(coeffs).flatten()
    coeffs = np.round_(coeffs, decimals=4)

    return coeffs


def delta_wavefront(field, sidel):
    """
    Return the longitude aberration of input light field.

    Parameters
    ----------
        field : SimLight.Field
            The light field of the lens with aberration.
        sidel : SimLight.Sidel
            Sidel coefficients of the lens.

    Returns
    ----------
        delta_W : array-like, float
            Derivative of the aberrated wavefront.
    """
    field = sl.Field.copy(field)

    size = field.size
    N = field.N
    k = 2 * np.pi / field.wavelength
    W = sidel.coefficients

    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    h = 1
    rad = np.pi / 180

    piston = W[0][0] * h**2
    tilt = W[1][0] * h * (np.cos(theta - W[1][1] * rad) -
                          rho * np.sin(theta))
    defocus = 2 * W[2][0] * rho
    astigmatism = W[3][0] * h**2 * (2 * rho * np.cos(theta - W[3][1] * rad)**2
                                    - rho**2 * np.sin(theta))
    coma = W[4][0] * h * (3 * rho**2 * np.cos(theta - W[4][1] * rad) -
                          rho**3 * np.sin(2 * theta))
    spherical = 4 * W[5][0] * rho**3

    delta_W = piston + tilt + defocus + astigmatism + coma + spherical
    delta_W *= k

    return delta_W


def deformable_mirror(field, K):
    """
    Calculate moving distance of actuators of a deformable mirror.

    Parameters
    ----------
        field : SimLight.Field
            A light field incident on a deformable mirror.
        K : int
            Actuators of the deformable mirror in one direction.

    Returns
    ----------
        dm_field : SimLight.Field
            Light field generated by deformable mirror for aberration
            compensation.
    """
    field = sl.Field.copy(field)

    phase_ = phase(field, unwrap=True)

    wavelength = field.wavelength
    size = field.size
    N = field.N
    dm_field = sl.PlaneWave(wavelength, size, N)

    x_dm = np.linspace((-K + 1) / 2, (K - 1) / 2, K)
    X_dm, Y_dm = np.meshgrid(x_dm, x_dm)
    x = np.linspace(-size / 2, size / 2, N)
    X, Y = np.meshgrid(x, x)

    dm_points = np.zeros((K, K))
    dm_points_X = np.zeros((K, K))
    dm_points_Y = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            ii = int(N / 8 * i + N / 16)
            jj = int(N / 8 * j + N / 16)
            dm_points[i][j] = phase_[ii][jj] / 2
            dm_points_X[i][j] = X[ii][jj]
            dm_points_Y[i][j] = Y[ii][jj]
    dm_phase = scipy.interpolate.interp2d(dm_points_X, dm_points_Y,
                                          dm_points, kind='cubic')
    dm_phase = dm_phase(x, x) * 2

    res_phase = phase_ - dm_phase
    dm_field.complex_amp *= np.exp(-1j * res_phase)

    return dm_field
