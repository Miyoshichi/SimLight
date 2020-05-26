# -*- coding: utf-8 -*-

"""
Created on May 22, 2020
@author: Zhou Xiang
"""

import numpy as np


def phase(field):
    """
    Calculate the phase of a light field.

    Args:
        field:
            The light field to be calculated.
    Returns：
        phase:
            The phase of the light field.
    """
    phase = np.angle(field.complex_amp)

    return phase
