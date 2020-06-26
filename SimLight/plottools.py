# -*- coding: utf-8 -*-

"""
Created on May 22, 2020
@author: Zhou Xiang
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

import SimLight as sl
from .utils import pv, rms, circle_aperature
from .calc import phase, intensity, psf


def plot_wavefront(field, mask_r=None, dimension=2, title=''):
    """
    Plot the wavefront of light field using matplotlib.

    Args:
        field:
            A light field.
        mask_r: float
            Radius of a circle mask. (optional, between 0 and 1,
            default is None).
        dimension: int
            Dimension of figure. (optional, default is 2, i.e. surface)
            2: surface
            3: 3d
        title: str
            Title of the figure. (optional).
    """
    unwrap = True

    # check of input parameters
    if mask_r:
        if mask_r > 1 or mask_r < 0:
            raise ValueError('Invalid radius of circle mask.')
    if dimension:
        if dimension < 1 or dimension > 3 or type(dimension) is not int:
            raise ValueError('Invalid dimension.')
    if isinstance(field, sl.Field) is True:
        wavelength = field.wavelength
        size = field.size
        N = field.N
        phase_ = phase(field, unwrap=unwrap)
    elif isinstance(field, list) is True:
        wavelength = field[0]
        size = field[1]
        N = field[2]
        phase_ = phase(field[3], unwrap=unwrap)
    else:
        raise ValueError('Invalid light field.')

    phase_ = wavelength * phase_ / (2 * np.pi)

    fig = plt.figure()

    if mask_r:
        _, _, norm_radius = circle_aperature(phase_, mask_r)
        max_value = np.max(phase_[norm_radius <= mask_r])
        min_value = np.min(phase_[norm_radius <= mask_r])
        PV = 'P-V: ' + str(round(pv(phase_, mask=True), 3)) + ' λ'
        RMS = 'RMS: ' + str(round(rms(phase_, mask=True), 3)) + ' λ'
    else:
        max_value = np.max(phase_)
        min_value = np.min(phase_)
        PV = 'P-V: ' + str(round(pv(phase_), 3)) + ' λ'
        RMS = 'RMS: ' + str(round(rms(phase_), 3)) + ' λ'

    if dimension == 2:
        extent = [-size / 2, size / 2, -size / 2, size / 2]
        ax = fig.add_subplot(111)
        im = ax.imshow(phase_, cmap='rainbow', extent=extent,
                       vmin=min_value, vmax=max_value)
        if mask_r:
            mask = patches.Circle([0, 0], size * mask_r / 2,
                                  fc='none', ec='none',)
            ax.add_patch(mask)
            im.set_clip_path(mask)
        ax.text(0.05, 0.95, PV, fontsize=12, horizontalalignment='left',
                transform=ax.transAxes)
        ax.text(0.05, 0.90, RMS, fontsize=12, horizontalalignment='left',
                transform=ax.transAxes)
        fig.colorbar(im)
    elif dimension == 3:
        ax = fig.add_subplot(111, projection='3d')
        length = np.linspace(-size / 2, size / 2, phase_.shape[0])
        X, Y = np.meshgrid(length, length)
        if mask_r:
            radius = np.sqrt(X**2 + Y**2)
            X[radius > size * mask_r / 2] = np.nan
            Y[radius > size * mask_r / 2] = np.nan
        stride = math.ceil(N / 25)
        im = ax.plot_surface(X, Y, phase_, rstride=stride, cstride=stride,
                             cmap='rainbow', vmin=min_value, vmax=max_value)
        ax.set_zlabel('Wavefront [λ]')
        ax.text2D(0.00, 0.95, PV, fontsize=12, horizontalalignment='left',
                  transform=ax.transAxes)
        ax.text2D(0.00, 0.90, RMS, fontsize=12, horizontalalignment='left',
                  transform=ax.transAxes)
        fig.colorbar(im)
    else:
        ax = fig.add_subplot(111)
        center = int(phase_.shape[0] / 2)
        if mask_r:
            length = int((phase_.shape[0] * mask_r) / 2) * 2
            X = np.linspace(-size * mask_r / 2, size * mask_r / 2, length)
            [left, right] = [center - length / 2, center + length / 2]
            im = ax.plot(X, phase_[center][int(left):int(right)])
        else:
            X = np.linspace(-size / 2, size / 2, phase_.shape[0])
            im = ax.plot(X, phase_[center])
        ax.set_xlabel('Size [mm]')
        ax.set_ylabel('Phase [λ]')

    if title:
        ax.set_title(title)

    plt.show()


def plot_intensity(field, mask_r=None, norm_type=0, dimension=2, title=''):
    """
    Plot the intensity of light field using matplotlib.

    Args:
        field:
            A light field.
        mask_r: float
            Radius of a circle mask. (optional, between 0 and 1,
            default is None).
        norm_type: int
            Type of normalization. (optional, default is 0)
            0: no normalization.
            1: normalize to 0~1.
            2: normalize to 0~255.
        dimension: int
            Dimension of figure. (optional, default is 2, i.e. surface)
            1: line
            2: surface
        title: str
            Title of the figure. (optional).
    """
    # check of input parameters
    if mask_r:
        if mask_r > 1 or mask_r < 0:
            raise ValueError('Invalid radius of circle mask.')
    if dimension:
        if dimension < 1 or dimension > 2 or type(dimension) is not int:
            raise ValueError('Invalid dimension.')
    if isinstance(field, sl.Field) is True:
        size = field.size
        intensity_ = intensity(field, norm_type=norm_type)
    elif isinstance(field, list) is True:
        size = field[0]
        intensity_ = intensity(field[1], norm_type=norm_type)
    else:
        raise ValueError('Invalid light field.')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if dimension == 2:
        extent = [-size / 2, size / 2, -size / 2, size / 2]
        im = ax.imshow(intensity_, cmap='gist_gray', extent=extent, vmin=0)
        if mask_r:
            mask = patches.Circle([0, 0], mask_r, fc='none', ec='none')
            ax.add_patch(mask)
            im.set_clip_path(mask)
            ax.set_xlabel('Size [mm]')
        fig.colorbar(im)
    else:
        center = int(intensity_.shape[0] / 2)
        if mask_r:
            length = int((intensity_.shape[0] * mask_r) / 2) * 2
            X = np.linspace(-size * mask_r / 2, size * mask_r / 2, length)
            [left, right] = [center - length / 2, center + length / 2]
            im = ax.plot(X, intensity_[center][int(left):int(right)])
        else:
            X = np.linspace(-size / 2, size / 2, intensity_.shape[0])
            im = ax.plot(X, intensity_[center])
        ax.set_xlabel('Size [mm]')
        ax.set_ylabel('Intensity [a.u.]')

    if title:
        ax.set_title(title)

    plt.show()


def plot_psf(field, aperture_type='circle', dimension=2, title=''):
    """
    Show the figure of point spread function of a light field.

    Args:
        field: tuple
            The light fiedl.
        aperture_type: str
            The shape of the aperture. (optional, default is 'circle')
                circle: circle aperture
                square: square aperture
        dimension: int
            Dimension of figure. (optional, default is 2, i.e. surface)
            1: line
            2: surface
        title: str
            Title of the figure. (optional).
    """
    aperture = ['circle', 'square']
    # check of input parameters
    if isinstance(field, sl.Field) is True:
        size = field.size
        if aperture_type not in aperture:
            raise ValueError('Unsupport aperture type')
        psf_ = psf(field, aperture_type)
    else:
        raise ValueError('Invalid light field.')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if dimension == 2:
        extent = [-1, 1, -1, 1]
        im = ax.imshow(psf_, cmap='rainbow', extent=extent, vmin=0)
    else:
        center = int(psf_.shape[0] / 2)
        X = np.linspace(-size / 2, size / 2, psf_.shape[0])
        im = ax.plot(X, psf_[center])
        ax.set_ylabel('Intensity [a.u.]')

    if title:
        ax.set_title(title)

    plt.show()
