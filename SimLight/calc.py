# -*- coding: utf-8 -*-

"""
Created on May 22, 2020
@author: Zhou Xiang
"""

import numpy as np

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
    phase = np.angle(field.complex_amp)

    if unwrap is True:
        phase = simple_unwrap(phase)

    return phase


def intensity(field, norm_type=0):
    """
    Calculate the intensity of a light field.

    Args:
        field: tuple
            The light field to be calculated.
        norm_type: int
            Type of normalization. (optional, default is 0)
            0: no normalization.
            1: normalize to 0~1.
            2: normalize to 0~255.
    Returns：
        intensity:
            The intensity of the light field.
    """
    intensity = np.abs(field.complex_amp)**2

    if norm_type < 0 or norm_type > 2:
        raise ValueError('Unknown normalization type.')
    elif norm_type == 0:
        print('normalization type: 0')
    elif norm_type >= 1:
        intensity /= np.max(intensity)
        if norm_type == 1:
            print('normalization type: 1')
        elif norm_type == 2:
            intensity *= 255
            print('normalizaiton: 2')
        else:
            raise ValueError('Unknown normalization type.')
    else:
        raise ValueError('Unknown normalization type.')

    return intensity
