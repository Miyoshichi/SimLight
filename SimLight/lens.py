# -*- coding: utf-8 -*-

"""
Created on June 15, 2020
@author: Zhou Xiang
"""

import numpy as np

from .utils import zernike_to_sidel


class Lens:
    """
    A fundmental lens which is central symmetry.

    Args:
        D: float
            Physical size (diameter) of the lens, unit: mm.
        f: float
            Focal length of a lens.
    """
    counts = 0

    def __init__(self, D, f=float('inf'), coeff=[]):
        """
        A fundmental lens which is central symmetry.

        Args:
            D: float
                Physical size (diameter) of lens, unit: mm.
            f: float
                Focal length of lens.
            abbr: float
                Zernike coefficients of the lens's aberration.
        """
        # check of inputted parameters
        if D <= 0:
            raise ValueError('Light field cannot be smaller than 0.')
        if f == 0:
            raise ValueError('Focal length cannot be 0.')

        Lens.counts += 1
        self._D = D
        self._f = f
        self._coeff = coeff
        self._zernike, self._sidel = self.__coefficients(self._coeff)
        self._lens_number = Lens.counts
        self._lens_type = 'lens'
        self._F = self.__F_number(self._D, self._f)

    @classmethod
    def new_lens(cls, D, f=float('inf'), coeff=[]):
        # cls.counts += 1
        lens = cls(D, f, coeff)
        return lens

    @staticmethod
    def __coefficients(coeff):
        zernike = np.zeros(15)
        if coeff is not []:
            j = len(coeff)
            if j > 15:
                raise ValueError('Unspported geometrical aberration.')
            zernike[:j] = coeff
        sidel = zernike_to_sidel(zernike)
        return zernike, sidel

    @staticmethod
    def __F_number(D, f):
        """
        Calculate the F# of a lens.

        Args:
            D: float
                Physical size (diameter) of lens, unit: mm.
            f: float
                Focal length of lens.
        Returns:
            F: float
                F# of a lens.
        """
        F = f / D
        return F

    @property
    def D(self):
        return self._D

    @property
    def f(self):
        return self._f

    @property
    def coeff(self):
        return self._coeff

    @property
    def zernike(self):
        return self._zernike

    @property
    def sidel(self):
        return self._sidel

    @coeff.setter
    @zernike.setter
    @sidel.setter
    def coeff(self, coeff):
        self._coeff = coeff
        self._zernike, self._sidel = self.__coefficients(self._coeff)

    @property
    def lens_number(self):
        return self._lens_number

    @property
    def lens_type(self):
        return self._lens_type

    @property
    def F(self):
        return self._F


class CylindricalLens(Lens):
    """
    A cylindrical lens.

    Args:
        D: float
            Physical size of the lens, unit: mm.
        f: float
            Focal length of a lens.
        direction: int
            Cylindrical direction. (optional, default is 0.)
            0: x direction
            1: y direction
    """
    def __init__(self, D, f, aber, direction=0):
        """
        A cylindrical lens.

        Args:
            D: float
                Physical size of the lens, unit: mm.
            f: float
                Focal length of a lens.
            abbr: float
                Zernike coefficients of the lens's aberration.
            direction: int
                Direction of curve. (optional, default is 0.)
                0: x direction (horizontal)
                1: y direction (vertical)
        """
        super().__init__(D, f, aber)
        self._lens_type = 'cylindrical lens'
        self._direction = direction

    @classmethod
    def new_lens(cls, D, f, aber, direction=0):
        # Lens.counts += 1
        cylindrical_lens = cls(D, f, aber, direction)
        return cylindrical_lens

    @property
    def lens_type(self):
        return self._lens_type

    @property
    def direction(self):
        return self._direction
