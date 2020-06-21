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
from .calc import phase, intensity


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
    # input error check
    if mask_r:
        if mask_r > 1 or mask_r < 0:
            raise ValueError('Invalid radius of circle mask.')
    if dimension:
        if dimension < 1 or dimension > 3 or type(dimension) is not int:
            raise ValueError('Invalid dimension.')
    if isinstance(field, sl.Field) is True:
        size = field.size
        phase_ = phase(field, unwrap=True)
        N = field.N
    elif isinstance(field, list) is True:
        size = field[0]
        phase_ = phase(field[1], unwrap=True)
        N = field[2]
    else:
        raise ValueError('Invalid light field')

    fig = plt.figure()

    if dimension == 2:
        extent = [-size / 2, size / 2, -size / 2, size / 2]
        ax = fig.add_subplot(111)
        im = ax.imshow(phase_, cmap='rainbow', extent=extent)
        if mask_r:
            mask = patches.Circle([0, 0], size * mask_r / 2,
                                  fc='none', ec='none')
            ax.add_patch(mask)
            im.set_clip_path(mask)
        fig.colorbar(im)
    elif dimension == 3:
        ax = fig.add_subplot(111, projection='3d')
        length = np.linspace(-size / 2, size / 2, phase_.shape[0])
        X, Y = np.meshgrid(length, length)
        if mask_r:
            _, _, norm_radius = circle_aperature(phase_, mask_r)
            radius = np.sqrt(X**2 + Y**2)
            X[radius > size * mask_r / 2] = np.nan
            Y[radius > size * mask_r / 2] = np.nan
            max_value = np.max(phase_[norm_radius <= mask_r])
            min_value = np.min(phase_[norm_radius <= mask_r])
        else:
            max_value = np.max(phase_)
            min_value = np.min(phase_)
        stride = math.ceil(N / 25)
        im = ax.plot_surface(X, Y, phase_, rstride=stride, cstride=stride,
                             cmap='rainbow', vmin=min_value, vmax=max_value)
        ax.set_zlabel('Wavefront [rad]')
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
        ax.set_ylabel('Phase [rad]')

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
    # input error check
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
        raise ValueError('Invalid light field')

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
