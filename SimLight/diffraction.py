# -*- coding: utf-8 -*-

"""
Created on June 22, 2020
@author: Zhou Xiang
"""

import numpy as np
import scipy.special
from pyfftw.interfaces.numpy_fft import fft2, ifft2


def fresnel(field, z):
    """
    Calculate the near-field Fresnel Diffraction of the light field.

    Args:
        field: tuple
            The light field to be calculated.
        z: float
            Propagation distance.
    Returns:
        field: tuple
            The diffracted light field.
    """
    N = field.N
    dx = field.size / field.N
    kz = 2 * np.pi / field.wavelength * z
    cos_kz = np.cos(kz)
    sin_kz = np.sin(kz)
    _fftargs = {'planner_effort': 'FFTW_ESTIMATE',
                'overwrite_input': True,
                'threads': -1}

    half_field = int(N / 2)

    in_outF = np.zeros((2 * N, 2 * N), dtype=complex)
    in_outK = np.zeros((2 * N, 2 * N), dtype=complex)

    ii2N = np.ones((2 * N), dtype=float)
    ii2N[1::2] = -1
    iiij2N = np.outer(ii2N, ii2N)
    iiij2_half_field = iiij2N[:2 * half_field, :2 * half_field]
    iiijN = iiij2N[:N, :N]

    RR = np.sqrt(1/(2 * field.wavelength * z)) * dx * 2
    io = np.arange(0, (2 * half_field) + 1)
    R1 = RR * (io - half_field)
    fs, fc = scipy.special.fresnel(R1)
    fss = np.outer(fs, fs)
    fsc = np.outer(fs, fc)
    fcs = np.outer(fc, fs)
    fcc = np.outer(fc, fc)

    temp_re = (fsc[1:, 1:] + fcs[1:, 1:])
    temp_re -= fsc[:-1, 1:]
    temp_re -= fcs[:-1, 1:]
    temp_re -= fsc[1:, :-1]
    temp_re -= fcs[1:, :-1]
    temp_re += fsc[:-1, :-1]
    temp_re += fcs[:-1, :-1]
    temp_im = (-fcc[1:, 1:] + fss[1:, 1:])
    temp_im += fcc[:-1, 1:]
    temp_im -= fss[:-1, 1:]
    temp_im += fcc[1:, :-1]
    temp_im -= fss[1:, :-1]
    temp_im -= fcc[:-1, :-1]
    temp_im += fss[:-1, :-1]

    temp_K = 1j * temp_im
    temp_K += temp_re
    temp_K *= iiij2_half_field
    temp_K *= 0.5
    in_outK[(N - half_field):(N + half_field),
            (N - half_field):(N + half_field)] = temp_K
    in_outF[(N - half_field):(N + half_field),
            (N - half_field):(N + half_field)] =\
        field.complex_amp[(N - 2 * half_field):N,
                          (N - 2 * half_field):N]
    in_outF[(N - half_field):(N + half_field),
            (N - half_field):(N + half_field)] *= iiij2_half_field

    in_outK = fft2(in_outK, **_fftargs)
    in_outF = fft2(in_outF, **_fftargs)
    in_outF *= in_outK
    in_outF *= iiij2N
    in_outF = ifft2(in_outF, **_fftargs)

    Ftemp = (in_outF[half_field:N+half_field, half_field:N+half_field]
             - in_outF[half_field-1:N+half_field-1, half_field:N+half_field])
    Ftemp += in_outF[half_field-1:N+half_field-1, half_field-1:N+half_field-1]
    Ftemp -= in_outF[half_field:N+half_field, half_field-1:N+half_field-1]
    comp = complex(cos_kz, sin_kz)
    Ftemp *= 0.25 * comp
    Ftemp *= iiijN
    field.complex_amp = Ftemp

    return field


def fresnel2(field, z):
    """
    Calculate the near-field Fresnel Diffraction of the light field.

    Args:
        field: tuple
            The light field to be calculated.
        z: float
            Propagation distance.
    Returns:
        field: tuple
            The diffracted light field.
    """
    field.complex_amp *= np.exp(1j / z)
    field.complex_amp = np.fft.fftshift(np.fft.fft2(field.complex_amp))

    return field


def fraunhofer(field, z):
    """
    Calculate the far-field Fraunhofer Diffraction of the light field.

    Args:
        field: tuple
            The light field to be calculated.
        z: float
            Propagation distance.
    Returns:
        field: tuple
            The diffracted light field.
    """
    field.complex_amp *= np.exp(1j / z)
    field.complex_amp = 1j * np.exp(1j * 2 * np.pi / field.wavelength * z) /\
        field.wavelength / z *\
        np.fft.fftshift(np.fft.fft2(field.complex_amp))

    return field
