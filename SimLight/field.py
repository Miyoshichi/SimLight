# -*- coding: utf-8 -*-

"""
Created on May 21, 2020
@author: Zhou Xiang
"""

import math
import copy
import numpy as np

from .plottools import plot_wavefront, plot_intensity
from .utils import cart2pol
from .units import *


class Field:
    """
    A basic light field.

    Parameters
    ----------
        wavelength : float
            Physical wavelength of input light.
        size : float
            Physical size of input light field.
                circle: diameter
                square: side length
        N : int
            Pixel numbers of input light field in one dimension.
    """
    def __init__(self, wavelength=1.0, size=0, N=0):
        """
        A basic light field.

        Parameters
        ----------
            wavelength : float
                Physical wavelength of input light.
            size : float
                Physical size of input light field.
                    circle: diameter
                    square: side length
            N : int
                Pixel numbers of input light field in one dimension.
        """
        # check of inputted parameters
        if wavelength <= 0:
            raise ValueError('Wavelength cannot be less than 0.')
        if size <= 0:
            raise ValueError('Light field cannot be smaller than 0.')
        if N <= 0:
            raise ValueError('Cannot generate zero light field')

        self._wavelength = wavelength
        self._size = size
        self._N = N
        self._curvature = 0
        self._phase_ratio = 1
        self._complex_amp = np.ones([N, N], dtype=np.complex)
        self._complex_amp2 = np.ones([N, N], dtype=np.complex)

    def zernike_aberration(self):
        """TODO
        """
        pass

    def plot_wavefront(self, noise=False, mask_r=None, dimension=2,
                       unit='mm', title=''):
        """Plot the wavefront.

        Plot the wavefront of light field using matplotlib.

        Parameters
        ----------
            field : SimLight.Field
                A light field.
            mask_r : float, optional, from 0 to 1, default None
                Radius of a circle mask.
            dimension : int, optional, {1, 2, 3}, default 2
                Dimension of the showing wavefront, where
                    2 for surface,
                    3 for 3d.
            unit : str, optional, {'m', 'cm', 'mm', 'um', 'µm', 'nm'},
            default 'µm'
                Unit used for FOV.
            title : str, optional
                Title of the figure.
        """
        mask_r = mask_r
        dimension = dimension
        unit = unit
        title = title
        fid = 'SimLight'
        field = [self._wavelength,
                 self._size,
                 self._N,
                 self._complex_amp2,
                 self._phase_ratio,
                 fid]
        plot_wavefront(field, noise, mask_r, dimension, unit, title)

    def plot_intensity(self, mask_r=None, norm_type=0, dimension=2, mag=1,
                       unit='µm', title=''):
        """Plot the intensity.

        Plot the intensity of light field using matplotlib.

        Parameters
        ----------
            field : SimLight.Field
                A light field.
            mask_r : float, optional, from 0 to 1, default None
                Radius of a circle mask.
            norm_type : int, optional, {0, 1, 2}, default 0
                Type of normalization, where
                    0 for no normalization,
                    1 for normalize up to 1,
                    2 for normalize up to 255.
            dimension : int, optional, {1, 2}, default 2
                Dimension of the showing intensity, where
                    1 for showing the intensity in a line,
                    2 for showing the intensity in a surface.
            mag : float, optional, default 1
                Magnification of the figure.
            unit : str, optional, {'m', 'cm', 'mm', 'um', 'µm', 'nm'},
            default 'µm'
                Unit used for FOV.
            title : str, optional, default ''
                Title of the figure.
        """
        mask_r = mask_r
        norm_type = norm_type
        dimension = dimension
        mag = mag
        unit = unit
        title = title
        field = [self._size, self._N, self._complex_amp]
        plot_intensity(field, mask_r, norm_type, dimension, mag, unit,
                       title)

    def zernike_coeffs(self, j):
        """
        docstring
        """
        x = np.linspace(-self._size / 2, self._size / 2, self._N)
        X, Y = np.meshgrid(x, x)
        theta, r = cart2pol(X, Y)

    @classmethod
    def copy(cls, field):
        """
        Create a copy of the input light field so that the original field
        would not be changed.

        Parameters
        ----------
            field : SimLight.Field
                Input light field to copy.

        Returns
        ----------
            copied_field : SimLight.Field
                A new copied light field.
        """
        return copy.deepcopy(field)

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = size

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, N):
        self._N = N

    @property
    def curvature(self):
        return self._curvature

    @curvature.setter
    def curvature(self, curvature):
        self._curvature = curvature

    @property
    def phase_ratio(self):
        return self._phase_ratio

    @phase_ratio.setter
    def phase_ratio(self, phase_ratio):
        self._phase_ratio = phase_ratio

    @property
    def complex_amp(self):
        return self._complex_amp

    @complex_amp.setter
    def complex_amp(self, complex_amp):
        self._complex_amp = complex_amp

    @property
    def complex_amp2(self):
        return self._complex_amp2

    @complex_amp2.setter
    def complex_amp2(self, complex_amp2):
        self._complex_amp2 = complex_amp2


class PlaneWave(Field):
    """
    A plane wave light field.

    Parameters
    ----------
        wavelength : float
            Physical wavelength of input light.
        size : float
            Physical size of input light field.
                circle: diameter
                square: side length
        N : int
            Pixel numbers of input light field in one dimension.
        x_tilt : float
            Tilt coefficient in x direction, unit: rad.
        y_tilt : float
            Tilt coefficient in y direciton, unit: rad.
    """
    def __init__(self, wavelength, size, N, x_tilt=0, y_tilt=0):
        """
        A plane wave light field.

        Parameters
        ----------
            x_tilt : float
                Tilt in x direction, unit: rad.
            y_tilt : float
                Tilt in y direciton, unit: rad.
        """
        super().__init__(wavelength, size, N)
        self._x_tilt = x_tilt
        self._y_tilt = y_tilt
        self._field_type = 'plane wave'
        self._complex_amp *= self.__tilt(self._wavelength,
                                         self._size,
                                         self._N,
                                         [self._x_tilt, self._y_tilt])
        self._complex_amp2 *= self.__tilt(self._wavelength,
                                          self._size,
                                          self._N,
                                          [self._x_tilt, self._y_tilt])

    @staticmethod
    def __tilt(wavelength, size, N, tilt):
        """
        Return a tilted light field.
        U = A * exp(ikr - φ0)

        Parameters
        ----------
            wavelength : float
                Physical wavelength of input light.
            size : float
                Physical size of input light field.
                    circle: diameter
                    square: side length
            N : int
                Pixel numbers of input light field in one dimension.
            tilt : list, [x_tilt, y_tilt]
                x_tilt: float
                    Tilt coefficient in x direction, unit: rad.
                y_tilt: float
                    Tilt coefficient in y direciton, unit: rad.
        """
        x = np.linspace(-size / 2, size / 2, N)
        X, Y = np.meshgrid(x, x)
        k = 2 * np.pi / wavelength
        phi = -k * (tilt[0] * X + tilt[1] * Y)
        return np.exp(1j * phi)

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

    Parameters
    ----------
        wavelength : float
            Physical wavelength of input light.
        size : float
            Physical size of input light field.
                circle: diameter
                square: side length
        N : int
            Pixel numbers of input light field in one dimension.
        z : float
            The propagation distance of the spherical wave from center.
    """
    def __init__(self, wavelength, size, N, z=0):
        """
        A spherical wave light field.

        Parameters
        ----------s
            z : float
                The propagation distance of the spherical wave from center.
        """
        super().__init__(wavelength, size, N)
        self._z = z
        self._field_type = 'spherical wave'
        self._complex_amp *= self.__sphere(self._wavelength, self._size,
                                           self._N, self._z)
        self._complex_amp2 *= self.__sphere(self._wavelength, self._size,
                                            self._N, self._z)

    @staticmethod
    def __sphere(wavelength, size, N, z):
        """
        Return a spherical wave.
        U = (A / r) * exp(ikr - φ0)
            where r = √(x^2 + y^2 + z^2)

        Parameters
        ----------
            wavelength : float
                Physical wavelength of input light.
            size : float
                Physical size of input light field.
                    circle: diameter
                    square: side length
            N : int
                Pixel numbers of input light field in one dimension.
            z : float
                The propagation distance of the spherical wave from center.
        """
        x = np.linspace(-size / 2, size / 2, N)
        X, Y = np.meshgrid(x, x)
        r = np.sqrt(X**2 + Y**2 + z**2)
        k = 2 * np.pi / wavelength
        phi = -k * r
        return np.exp(1j * phi) / r

    @property
    def z(self):
        return self._z

    @property
    def field_type(self):
        return self._field_type


class Gaussian(Field):
    """
    A gaussian beam light field.

    Parameters
    ----------
        wavelength : float
            Physical wavelength of input light.
        size : float
            Physical size of input light field.
                circle: diameter
                square: side length
        N : int
            Pixel numbers of input light field in one dimension.
        w0 : float
            Size of the waist.
        z : float
            The propagation distance of the gaussian beam from the waist.
    """
    def __init__(self, wavelength, size, N, w0=0, z=0):
        """
        A spherical wave light field.

        Parameters
        ----------
            w0 : float
                Size of the waist.
            z : float
                The propagation distance of the gaussian beam
                from the waist.
        """
        super().__init__(wavelength, size, N)
        if w0 == 0:
            w0 = self._size / 2
        else:
            w0 /= 2

        self._w0 = w0
        self._z = z
        self._field_type = 'gaussian beam'
        self._complex_amp *= self.__gaussian(self._wavelength, self._size,
                                             self._N, self._w0, self._z)
        self._complex_amp2 *= self.__gaussian(self._wavelength, self._size,
                                              self._N, self._w0, self._z)

    @staticmethod
    def __gaussian(wavelength, size, N, w0, z):
        """
        Return a TEM00 mode gaussian beam.
        U = (A / ω(z)) * exp(-(x^2 + y^2) / ω^2(z)) *
            exp(-ik(z + (x^2 + y^2) / 2r(z)) + iφ(z))
            where ω(z) = ω0 * √(1 + (z / zR)^2)
                  r(z) = z * (1 + (zR / z)^2)
                  φ(z) = arctan(z / zR)
                  zR = πω0^2 / λ

        Parameters
        ----------
            wavelength : float
                Physical wavelength of input light.
            size : float
                Physical size of input light field.
                    circle: diameter
                    square: side length
            N : int
                Pixel numbers of input light field in one dimension.
            w0 : float
                Size of the waist.
            z : float
                The propagation distance of the gaussian beam
                from the waist.
        """
        x = np.linspace(-size / 2, size / 2, N)
        X, Y = np.meshgrid(x, x)
        z_R = np.pi * w0**2 / wavelength
        w_z = w0 * np.sqrt(1 + (z / z_R)**2)
        r_z = z * (1 + (z_R / z)**2) if z is not 0 else float('inf')
        phi_z = np.arctan2(z, z_R)
        k = 2 * np.pi / wavelength
        return np.exp(-(X**2 + Y**2) / w_z**2) *\
            np.exp(-1j * k * (z + (X**2 + Y**2) / (2 * r_z)) + 1j * phi_z) /\
            w_z

    @property
    def w0(self):
        return self._w0

    @property
    def z(self):
        return self._z

    @property
    def field_type(self):
        return self._field_type
