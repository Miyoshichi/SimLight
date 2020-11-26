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
                 wavelength=1.0, size=(0.,) * 3, N=0,
                 n0=1.0, n=None, dn=None):
        """
        """
        self._wavelength = wavelength
        self._size = size
        self._n0 = n0
        self._n = n
        self._dn = dn
        self._N = N
        self._curvature = 0
        self._phase_ratio = 1
        self._complex_amp = self.__set_complex_amp(self._wavelength,
                                                   self._size,
                                                   self._dn)
        self._complex_amp2 = self.__set_complex_amp(self._wavelength,
                                                    self._size,
                                                    self._dn)

    @classmethod
    def new_scattering_layer(cls,
                             wavelength=1.0, size=(0.,) * 3, N=0,
                             n0=1.0, n=None, dn=None):
        if isinstance(N, int) is True or isinstance(N, float) is True:
            Nx = Ny = N
            Nz = math.ceil(size[2] / size[0] * N)
        elif len(N) == 2:
            Nx = Ny = N[0]
            Nz = N[1]
        elif len(N) == 3:
            Nx = N[0]
            Ny = N[1]
            Nz = N[2]
        else:
            raise ValueError('Invalid N.')
        N = (Nx, Ny, Nz)

        if n is None and dn is None:
            dn = np.zeros(N, dtype=np.float32)
            n = np.array(dn + n0)
        if n is not None and dn is None:
            n = np.asarray(n)
            dn = np.array(n - n0)
        if dn is not None:
            dn = np.asarray(dn)
            n = np.array(dn + n0)

        scattering_layer = cls(wavelength, size, N, n0, n, dn)
        return scattering_layer

    @staticmethod
    def __set_complex_amp(wavelength, size, dn):
        # units conversion
        wavelength /= µm
        size = list(size)
        for i in range(len(size)):
            size[i] /= µm
        size = tuple(size)

        m = biobeam.Bpm3d(size=size,
                          lam=wavelength,
                          dn=dn,
                          n_volumes=1)
        u = m.propagate()

        return u[:, :, -1]

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
