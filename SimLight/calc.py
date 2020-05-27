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
            Whether to unwrap the phase.
    Returns：
        phase:
            The phase of the light field.
    """
    phase = np.angle(field.complex_amp)

    if unwrap is True:
        phase = simple_unwrap(phase)

    return phase
