# -*- coding: utf-8 -*-

"""
Created on Nov 25, 2020
@author: Zhou Xiang
"""

import math

import numpy as np
import biobeam

from .units import *


class ScatteringLayer:
    """
    docstring
    """
    def __init__(self,
                 wavelength=1.0, size=(0.,) * 3, n0=1.0, n=None, dn=None):
        """
        """
        self._wavelength = wavelength
        self._size = size
        self._n0 = n0
        self._n = n
        self._dn = dn
        # self._N = self.__set_N(N)
        self._curvature = 0
        self._phase_ratio = 1
        self._complex_amp_3d = self.__set_complex_amp_3d(self._wavelength,
                                                         self._size,
                                                         self._dn)
        self._complex_amp_3d2 = self.__set_complex_amp_3d(self._wavelength,
                                                          self._size,
                                                          self._dn)

    @classmethod
    def new_scattering_layer(cls,
                             wavelength=1.0, size=(0.,) * 3,
                             n0=1.0, n=None, dn=None):
        # check of input parameters
        if n is None and dn is None:
            raise ValueError(('Cannot generate a new scattering layer '
                              'without the refractive index or '
                              'relative refractive index distribution. '
                              '(\'n\' and \'dn\')'))
        if n is not None and dn is None:
            n = np.asarray(n)
            dn = np.asarray(n - n0)
        if dn is not None and n is None:
            dn = np.asarray(dn)
            n = np.asarray(dn + n0)
        if n is not None and dn is not None:
            n = np.asarray(n)
            n_ = np.asarray(dn + n0)
            if (n_ - n).any():
                raise ValueError('')

        scattering_layer = cls(wavelength, size, n0, n, dn)
        return scattering_layer

    # @staticmethod
    # def __set_N(size, N):
    #     N = np.asarray(N)
    #     Nx = N[0]

    #     if len(N.shape) > 1:
    #         Ny = N[1]
    #         if len(N.shape) > 2:
    #             Nz = N[2]
    #     else:
    #         Ny = int(size[0] / Nx * size[1])
    #         Nz = math.ceil(size[0] / Nx * size[2])

    #     return (Nx, Ny, Nz)

    @staticmethod
    def __set_complex_amp_3d(wavelength, size, dn):
        wavelength /= µm
        for s in size:
            s /= µm
        m = biobeam.Bpm3d(size=size,
                          lam=wavelength,
                          dn=dn)
        u = m.propagate()
        return u

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
    def n0(self):
        return self._n0

    @n0.setter
    def n0(self, n0):
        self._n0 = n0

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        self._n = n

    @property
    def dn(self):
        return self._dn

    @dn.setter
    def dn(self, dn):
        self._dn = dn
