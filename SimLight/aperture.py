# -*- coding: utf-8 -*-

"""
Created on June 26, 2020
@author: Zhou Xiang
"""

import numpy as np

import SimLight as sl


def circle_aperture(field, radius=1):
    """
    Generate a circle aperture.

    Parameters
    ----------
        field : SimLight.Field
            Input light field.
        radius : float, optional, default 1
            Normalized radius of the circle.
    """
    field = sl.Field.copy(field)

    x = np.linspace(-field.size / 2, field.size / 2, field.N)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    real_r = radius * field.size / 2
    field.complex_amp[R > real_r] = 0
    field.complex_amp2[R > real_r] = 0

    return field
