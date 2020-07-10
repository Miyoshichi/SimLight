# -*- coding: utf-8 -*-

"""
Created on June 23, 2020
@author: Zhou Xiang
"""

import math
import numpy as np
import matplotlib.pyplot as plt

import SimLight as sl
import SimLight.plottools as slpl


class ZernikeCoefficients:
    """
    Return a list of Zernike polynomials cofficients.

    Args:
        j: int
            The terms of Zernike polynomials
        coefficients: list
            Predefined cofficients. (optional, default is void.)
    """
    def __init__(self, j, coefficients=[]):
        """
        Return a list of Zernike polynomials cofficients.
        """
        # check of input parameters:
        if j <= 0:
            raise ValueError('Should have at least 1 order.')
        elif isinstance(j, int) is not True:
            raise ValueError('The order should be int type')

        self._j = j
        self._n, self._m, self._norm = self.__order(self._j)
        self._coefficients = self.__zernike_coefficients(j, coefficients)

    @staticmethod
    def __order(j):
        n = np.zeros(j, dtype=int)
        m = np.zeros(j, dtype=int)
        norm = np.zeros(j)
        for i in range(1, j):
            n[i] = math.ceil((np.sqrt(8 * (i + 1) + 1) - 1) / 2 - 1)
            norm[i] = np.sqrt(2 * (n[i] + 1))
            if n[i] > n[i-1]:
                m[i] = -n[i]
            elif i == ((n[i] + 1) * (n[i] + 2)) / 2:
                m[i] = n[i]
            else:
                m[i] = m[i-1] + 2
            if m[i] == 0:
                norm[i] /= np.sqrt(2)
        return n, m, norm

    @staticmethod
    def __zernike_coefficients(j, input_coefficients):
        coefficients = np.zeros(j)
        if (input_coefficients is not []
                or (input_coefficients - coefficients).any() is True):
            order = len(input_coefficients)
            coefficients[:order] = input_coefficients
        return coefficients

    def show_coefficients(self):
        """
        Show the Zernike coefficients in figure.
        """
        terms = range(1, self._j + 1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['bottom'].set_position(('data', 0))
        ax.bar(terms, self._coefficients, tick_label=terms)

    def shwo_psf(self):
        """
        Show the figure of point spread function of a Zernike polynimials
        defined surface which the default parameters are 0.633 µm of
        wavelength, 25.4 mm of size and 500 pixels of N (grid number).
        """
        # default parameters
        wavelength = 0.6328
        size = 25.4
        N = 500

        field = sl.PlaneWave(wavelength, size, N)
        zernike = ZernikeCoefficients(self._j, self._coefficients)
        aber = sl.calc.aberration(field, zernike)
        slpl.plot_psf(aber)

    @property
    def j(self):
        return self._j

    @property
    def n(self):
        return self._n

    @property
    def m(self):
        return self._m

    @property
    def norm(self):
        return self._norm

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coefficients):
        self._coefficients = coefficients


class SidelCoefficients:
    """
    Return a list of Sidel Polynomials cofficients.

    Args:
        coefficients: list, [magnitude, angle]
            Predefined cofficients. (optional, default is void.)
    """
    def __init__(self, coefficients=[]):
        """
        Return a list of Sidel polynomials cofficients.
        """
        self._coefficients = self.__sidel_coefficients(coefficients)

    @staticmethod
    def __sidel_coefficients(input_coefficients):
        if len(input_coefficients) > 6:
            raise ValueError('Sidel polynomials coefficients cannot be '
                             'larger than 6.')
        coefficients = np.zeros((6, 2))
        if (input_coefficients is not []
                or (input_coefficients - coefficients).any() is True):
            order = len(input_coefficients)
            coefficients[:order] = input_coefficients
        return coefficients

    def show_surface(self):
        pass

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coefficients):
        self._coefficients = coefficients
