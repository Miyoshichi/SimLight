# -*- coding: utf-8 -*-

"""
Created on May 22, 2020
@author: Zhou Xiang
"""

import math
import numpy as np
import scipy.interpolate

import SimLight as sl
from .unwrap import simple_unwrap


def phase(field, unwrap=False):
    """
    Calculate the phase of a light field.

    Args:
        field: tuple
            The light field to be calculated.
        unwrap: bool
            Whether to unwrap the phase. (optional, default is False)
    Returns：
        phase:
            The phase of the light field.
    """
    if isinstance(field, sl.Field) is True:
        phase = np.angle(field.complex_amp)
    elif isinstance(field, np.ndarray) is True:
        phase = np.angle(field)
    else:
        raise ValueError('Invalid light field.')

    if unwrap is True:
        phase = simple_unwrap(phase)

    return phase


def intensity(field, norm_type=1):
    """
    Calculate the intensity of a light field.

    Args:
        field: tuple
            The light field to be calculated.
        norm_type: int
            Type of normalization. (optional, default is 1).
            0: no normalization
            1: normalize to 0~1
            2: normalize to 0~255
    Returns：
        intensity:
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

    Args:
        field: tuple
            The light fiedl.
        aperture_type: str
            The shape of the aperture. (optional, default is 'circle')
                circle: circle aperture
                square: square aperture
    Returns:
        psf:
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

    Args:
        field: tuple
            The light field to be calculated.
        zernike: tuple
            The Zernike Polynomials.
    Returns:
        aberrated_field: tuple
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

    Args:
        field: tuple
            The light field to be calculated.
        sidel: tuple
            The Sernike Polynomials.
    Returns:
        aberrated_field: tuple
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


def delta_wavefront(field, sidel):
    """
    Return the longitude aberration of input light field.

    Args:
        field: tuple
            The light field of the lens with aberration.
        sidel: tuple
            Sidel coefficients of the lens.
    Returns:
        delta_W: array
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
