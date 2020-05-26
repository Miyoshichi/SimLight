# -*- coding: utf-8 -*-

"""
Created on May 21, 2020
@author: Zhou Xiang
"""

import math
import numpy as np


class Field:
    """
    A basic light field.

    Args:
        wavelength: float
            physical wavelength of input light, unit: µm
        size: float
            physical size of input light field, unit: mm
                circle: diameter
                square: side length
        N: int
            pixel numbers of input light field in one dimension
    """
    def __init__(self, wavelength=1.0, size=0, N=0):
        """
        Initialize a new input light field.

        Args:
            wavelength: float
                physical wavelength of input light, unit: µm
            size: float
                physical size of input light field, unit: mm
                    circle: diameter
                    square: side length
            N: int
                pixel numbers of input light field in one dimension
        """
        # input error check
        if wavelength <= 0:
            raise ValueError('Wavelength cannot be less than 0.')
        if size < 0:
            raise ValueError('Light field cannot be smaller than 0.')
        if N <= 0:
            raise ValueError('Cannot generate zero light field')

        self._wavelength = wavelength
        self._size = size
        self._N = N
        self._complex_amp = np.ones([N, N], dtype=np.complex)

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def size(self):
        return self._size

    @property
    def N(self):
        return self._N

    @property
    def complex_amp(self):
        return self._complex_amp


class PlaneWave(Field):
    """
    A plane wave light field.

    Args:
    x_tilt: float
        Tilt in x direction, unit: rad
    y_tilt: float
        Tilt in y direciton, unit: rad
    """
    def __init__(self, wavelength, size, N, x_tilt=0, y_tilt=0):
        super().__init__(wavelength, size, N)
        self._x_tilt = x_tilt
        self._y_tilt = y_tilt
        self._field_type = 'plane wave'
        self._complex_amp = self.__tilt(self._complex_amp)

    def __tilt(self, complex_amp):
        h, w = self._N, self._N
        cy, cx = int(h / 2), int(w / 2)
        Y, X = np.mgrid[:h, :w]
        dx = self._size / self._N
        Y = (Y - cy) * dx
        X = (X - cx) * dx
        k = 2 * np.pi / self._wavelength
        phi = -k * (self._x_tilt * X + self._y_tilt * Y)
        complex_amp *= np.exp(1j * phi)
        return complex_amp

    @property
    def x_tilt(self):
        return self._x_tilt

    @property
    def y_tilt(self):
        return self._y_tilt

    @property
    def field_type(self):
        return self._field_type
