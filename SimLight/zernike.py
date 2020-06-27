# -*- coding: utf-8 -*-

"""
Created on June 23, 2020
@author: Zhou Xiang
"""

import math
import numpy as np
import matplotlib.pyplot as plt


class ZernikeCofficients:
    """
    Return a list of Zernike Polynomials cofficients.

    Args:
        j: int
            The terms of Zernike Polynomials
        cofficients: list
            Predefined cofficients. (optional, default is void.)
    """
    def __init__(self, j, cofficients=[]):
        """
        Return a list of Zernike Polynomials cofficients.
        """
        # check of input parameters:
        if j <= 0:
            raise ValueError('Should have at least 1 order.')
        elif isinstance(j, int) is not True:
            raise ValueError('The order should be int type')

        self._j = j
        self._input_cofficients = cofficients
        self._n, self._m, self._norm = self.__order(self._j)
        self._cofficients = self.__cofficients(self._j,
                                               self._input_cofficients)

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
        return n, m, norm

    @staticmethod
    def __cofficients(j, input_cofficients):
        cofficients = np.zeros(j)
        if input_cofficients:
            order = len(input_cofficients)
            cofficients[:order] = input_cofficients
        return cofficients

    def show_coefficients(self):
        """
        Show the Zernike coefficients in figure.
        """
        terms = range(1, self._j + 1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['bottom'].set_position(('data', 0))
        ax.bar(terms, self._cofficients, tick_label=terms)

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
    def input_cofficients(self):
        return self._input_cofficients

    @property
    def cofficients(self):
        return self._cofficients

    @cofficients.setter
    def cofficients(self, cofficients):
        self._cofficients = cofficients
