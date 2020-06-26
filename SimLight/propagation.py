# -*- coding: utf-8 -*-

"""
Created on June 22, 2020
@author: Zhou Xiang
"""

import math
import numpy as np

import SimLight as sl
from .diffraction import fresnel, fresnel2, fraunhofer


def propagation(field, lens, z):
    """
    Calculate the light field after passing through a lens without considering
    diffraction.

    Args:
        field: tuple
            The light field to be calculated.
        lens: tuple
            The lens which a light will pass through.
        z: float
            Propagation distance after passing through.
    Returns:
        field_out: tuple
            The light field after passing through a lens.
    """
    k = 2 * np.pi / field.wavelength
    x = np.linspace(-field.size / 2, field.size / 2, field.N)
    X, Y = np.meshgrid(x, x)

    # switch - case
    def simple_lens():
        r = np.sqrt(X**2 + Y**2 + (lens.f - z)**2)
        phi = k * np.sqrt(X**2 + Y**2 + (lens.f - z)**2)
        return r, phi

    def cylindrical_lens():
        if lens.direction == 0:
            x = X
        else:
            x = Y
        r = np.sqrt(x**2 + (lens.f - z)**2)
        phi = k * np.sqrt(x**2 + (lens.f - z)**2)
        return r, phi

    options = {
        'lens': simple_lens,
        'cylindrical lens': cylindrical_lens
    }

    r, phi = options[lens.lens_type]()
    if lens.f < 0:
        phi = -phi
    field.complex_amp *= (np.exp(1j * phi) / r)

    return field


def near_field_propagation(field, lens, z):
    """
    Calculate the light field after passing through a lens.

    Args:
        field: tuple
            The light field to be calculated.
        lens: tuple
            The lens which a light will pass through.
        z: float
            Propagation distance after passing through.
    Returns:
        field_out: tuple
            The light field after passing through a lens.
    """
    # check of input parameters
    if z < 0:
        raise ValueError('The propagation distance cannot be negative.')

    field = sl.Field.copy(field)

    if lens.D > field.size:
        size = lens.D if z <= 2 * lens.f else (z - lens.f) / lens.f * lens.D
        N = field.N * math.ceil(size / field.size)
        complex_amp = np.zeros([N, N], dtype=complex)
        L = int((N - field.N) / 2)
        R = L + field.N
        complex_amp[L:R, L:R] = field.complex_amp
        field.complex_amp = complex_amp
        field.size = size
        field.N = N
    elif lens.D <= field.size:
        size = field.size if z <= 2 * lens.f\
                          else (z - lens.f) / lens.f * field.size
        N = field.N * math.ceil(size / field.size)
        complex_amp = np.zeros([N, N], dtype=complex)
        lens_N = int((field.N * lens.D / field.size) / 2) * 2
        L = int((N - lens_N) / 2)
        R = L + lens_N
        L_in = int((field.N - lens_N) / 2)
        R_in = L_in + lens_N
        complex_amp[L:R, L:R] = field.complex_amp[L_in:R_in, L_in:R_in]
        field.complex_amp = complex_amp
        field.size = size
        field.N = N

    x = np.linspace(-size / 2, size / 2, N)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    k = 2 * np.pi / field.wavelength

    # switch - case
    def simple_lens():
        phi = -k * (X**2 + Y**2) / (2 * lens.f)
        return phi

    def cylindrical_lens():
        if lens.direction == 0:
            x = X
        else:
            x = Y
        phi = -k * (X**2) / (2 * lens.f)
        return phi

    options = {
        'lens': simple_lens,
        'cylindrical lens': cylindrical_lens
    }

    phi = options[lens.lens_type]()
    if lens.f < 0:
        phi = -phi

    # complex amplitude after passing through lens
    field.complex_amp *= np.exp(1j * phi)
    field.complex_amp[R >= field.size / 2] = 0
    # complex amplitude passing the distance z
    if z != 0:
        field = fresnel(field, z)

    return field
