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

from .utils import pv, rms, circle_aperature
from .calc import phase


def plot_wavefront(field, mask_r=None, plot3d=False, title=''):
    """
    Plot the wavefront of light field using matplotlib.

    Args:
        field:
            A light field.
        mask_r: float
            Radius of a circle mask. (optional, between 0 and 1,
            default is None)
        plot3d: bool
            Whether plot the figure in 3d. (optional, default is
            false)
        title: str
            Title of the figure. (optional)
    """
    # input error check
    if mask_r:
        if mask_r > 1 or mask_r < 0:
            raise ValueError('Invalid radius of circle mask.')

    phase_ = phase(field, unwrap=True)

    fig = plt.figure()
    if not plot3d:
        ax = fig.add_subplot(111)
        im = ax.imshow(phase_, cmap='rainbow', extent=[-1, 1, -1, 1])
        if mask_r:
            mask = patches.Circle([0, 0], mask_r, fc='none', ec='none')
            ax.add_patch(mask)
            im.set_clip_path(mask)
    else:
        ax = fig.add_subplot(111, projection='3d')
        if mask_r:
            X, Y, norm_radius = circle_aperature(phase_, mask_r)
            max_value = np.max(phase_[norm_radius <= mask_r])
            min_value = np.min(phase_[norm_radius <= mask_r])
        else:
            max_value = np.max(phase_)
            min_value = np.min(phase_)
        stride = math.ceil(field.N / 25)
        im = ax.plot_surface(X, Y, phase_, rstride=stride, cstride=stride,
                             cmap='rainbow', vmin=min_value, vmax=max_value)
    if title:
        ax.set_title(title)
    fig.colorbar(im)

    plt.show()
