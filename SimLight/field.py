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
            Physical wavelength of input light, unit: µm.
        size: float
            Physical size of input light field, unit: mm.
                circle: diameter
                square: side length
        N: int
            Pixel numbers of input light field in one dimension.
    """
    def __init__(self, wavelength=1.0, size=0, N=0):
        """
        A basic light field.

        Args:
            wavelength: float
                Physical wavelength of input light, unit: µm.
            size: float
                Physical size of input light field, unit: mm.
                    circle: diameter
                    square: side length
            N: int
                Pixel numbers of input light field in one dimension.
        """
        # check of inputted parameters
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
        wavelength: float
            Physical wavelength of input light, unit: µm.
        size: float
            Physical size of input light field, unit: mm.
                circle: diameter
                square: side length
        N: int
            Pixel numbers of input light field in one dimension.
        x_tilt: float
            Tilt coefficient in x direction, unit: rad.
        y_tilt: float
            Tilt coefficient in y direciton, unit: rad.
    """
    def __init__(self, wavelength, size, N, x_tilt=0, y_tilt=0):
        """
        A plane wave light field.

        Args:
            x_tilt: float
                Tilt in x direction, unit: rad.
            y_tilt: float
                Tilt in y direciton, unit: rad.
        """
        super().__init__(wavelength, size, N)
        self._x_tilt = x_tilt
        self._y_tilt = y_tilt
        self._field_type = 'plane wave'
        self._complex_amp = self.__tilt(self._complex_amp)

    def __tilt(self, complex_amp):
        """
        Return a tilted light field.
        """
        x = np.linspace(-self._size / 2, self._size / 2, self._N)
        X, Y = np.meshgrid(x, x)
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


class SphericalWave(Field):
    """
    A spherical wave light field.

    Args:
        wavelength: float
            Physical wavelength of input light, unit: µm.
        size: float
            Physical size of input light field, unit: mm.
                circle: diameter
                square: side length
        N: int
            Pixel numbers of input light field in one
            dimension.
        z: float
            The propagation distance of the spherical wave
            from center, unit: mm.
    """
    def __init__(self, wavelength, size, N, z=0):
        """
        A spherical wave light field.

        Args:
            z: float
                The propagation distance of the spherical wave
                from center, unit: mm.
        """
        super().__init__(wavelength, size, N)
        self._z = z
        self._field_type = 'spherical wave'
        self._complex_amp = self.__sphere(self._complex_amp)

    def __sphere(self, complex_amp):
        """
        Return a spherical wave.
        """
        x = np.linspace(-self._size / 2, self._size / 2, self._N)
        X, Y = np.meshgrid(x, x)
        d = np.sqrt(X**2 + Y**2)
        r = np.sqrt(d**2 + self._z**2)
        k = 2 * np.pi / self._wavelength
        phi = -k * r
        complex_amp *= np.exp(1j * phi) / r
        return complex_amp

    @property
    def z(self):
        return self._z

    @property
    def field_type(self):
        return self._field_type
