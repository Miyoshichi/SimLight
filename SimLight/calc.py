# -*- coding: utf-8 -*-

"""
Created on May 22, 2020
@author: Zhou Xiang
"""

import numpy as np

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


def lens(field, lens, z):
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
    k = 2 * np.pi / field.wavelength
    x = np.linspace(-field.size / 2, field.size / 2, field.N)
    X, Y = np.meshgrid(x, x)

    # switch - case
    def simple_lens():
        r = np.sqrt(X**2 + Y**2 + (lens.f - z)**2)
        # phi = -k * (X**2 + Y**2) / (2 * lens.f)
        phi = k * np.sqrt(X**2 + Y**2 + (lens.f - z)**2)
        return r, phi

    def cylindrical_lens():
        if lens.direction == 0:
            x = X
        else:
            x = Y
        r = np.sqrt(x**2 + (lens.f - z)**2)
        # phi = -k * (X**2) / (2 * lens.f)
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
