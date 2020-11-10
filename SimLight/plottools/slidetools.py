# -*- coding: utf-8 -*-

"""
Created on Nov 10, 2020
@author: Zhou Xiang
"""

import math
from matplotlib.pyplot import bar
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib_scalebar.scalebar import ScaleBar
from numpy.lib.function_base import average
import scipy.interpolate

import SimLight as sl
from ..utils import pv, rms, return_circle_aperature
from ..calc import phase, intensity, psf, zernike_coeffs
from ..units import *

np.random.seed(235)


def slide_plot_wavefront(field, noise=False, mask_r=None, dimension=2,
                         unit='mm', title='', return_data=False, **kwargs):
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
        unit : str, optional, {'m', 'cm', 'mm', 'um', 'µm', 'nm'}, default 'µm'
            Unit used for FOV.
        title : str, optional
            Title of the figure.
        return_data : bool, optional, default False
            Return the wavefront data or not.

    Returns
    ----------
        phase_ : numpy.ndarray
            Wavefront data.
    """
    unwrap = True

    field = sl.Field.copy(field)

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
        lambdaflag = False
    elif isinstance(field, list) is True:
        if len(field) == 6:
            wavelength = field[0]
            size = field[1]
            N = field[2]
            phase_ratio = field[4]
            phase_ = phase(field[3],
                           unwrap=unwrap,
                           phase_ratio=phase_ratio)
            lambdaflag = False
        elif len(field) == 5:
            wavelength = field[0]
            size = field[1]
            N = field[2]
            phase_ = field[3]
            lambdaflag = True
        else:
            raise ValueError('Invalid light field.')
    else:
        raise ValueError('Invalid light field.')

    # unit
    units = {
        'm': m,
        'cm': cm,
        'mm': mm,
        'um': µm,
        'µm': µm,
        'nm': nm
    }
    unit_ = units[unit]

    if lambdaflag is False:
        phase_ = wavelength * phase_ / (2 * np.pi) / µm
    if noise is True:
        noise_data = np.random.rand(N, N) * 1e-1
        phase_ += noise_data

    if mask_r:
        _, _, norm_radius = return_circle_aperature(phase_, mask_r)
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
        fig = plt.figure()
        length = np.linspace(-size / 2, size / 2, phase_.shape[0])
        X, Y = np.meshgrid(length, length)
        extent = [-size / 2, size / 2, -size / 2, size / 2]
        ax = fig.add_subplot(111)
        im = ax.imshow(phase_, cmap='rainbow', extent=extent,
                       vmin=min_value, vmax=max_value)
        if mask_r:
            mask = patches.Circle([0, 0], size * mask_r / 2,
                                  fc='none', ec='none',)
            ax.add_patch(mask)
            im.set_clip_path(mask)
            radius = np.sqrt(X**2 + Y**2)
            phase_[radius > size * mask_r / 2] = 0
        xticks = np.linspace(-size / 2, size / 2, 5)
        yticks = np.linspace(-size / 2, size / 2, 5)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        xticklabels = ax.get_xticks() / unit_
        yticklabels = ax.get_yticks() / unit_
        ax.set_xticklabels(xticklabels.astype(np.float16))
        ax.set_yticklabels(yticklabels.astype(np.float16))
        ax.set_xlabel('Size [%s]' % unit)
        ax.text(0.05, 0.95, PV, fontsize=12, horizontalalignment='left',
                transform=ax.transAxes)
        ax.text(0.05, 0.90, RMS, fontsize=12, horizontalalignment='left',
                transform=ax.transAxes)
        if kwargs['colorbar']:
            fig.colorbar(im)
        if title:
            fig.suptitle(title)
    elif dimension == 3:
        plt.rcParams.update({
            'grid.linewidth': 0.5,
            'grid.color': [0, 0, 0, 0.1],
        })
        length = np.linspace(-size / 2, size / 2, phase_.shape[0])
        X, Y = np.meshgrid(length, length)
        upper_value = max_value + (max_value - min_value) / 2
        lower_value = min_value - (max_value - min_value) / 5
        rccount = 100
        if mask_r:
            radius = np.sqrt(X**2 + Y**2)
            phase_[radius > size * mask_r / 2] = np.nan
        fig = plt.figure(figsize=(8, 5))
        if kwargs['colorbar']:
            ax = fig.add_subplot(111)
            caxins = inset_axes(ax,
                                width='2.5%',
                                height='85%',
                                loc='right',
                                bbox_to_anchor=(-0.075, -0.025, 1, 1),
                                bbox_transform=ax.transAxes,
                                borderpad=0)
        ax = fig.add_subplot(111, projection='3d')
        if PV != 'P-V: 0.0 λ' and RMS != 'RMS: 0.0 λ' and kwargs['cont']:
            cset = ax.contourf(X, Y, phase_,
                               zdir='z',
                               offset=lower_value,
                               cmap='rainbow', alpha=0.5)
        im = ax.plot_surface(X, Y, phase_,
                             rcount=rccount, ccount=rccount,
                             cmap='rainbow', alpha=0.9,
                             vmin=min_value, vmax=max_value)
        ax.view_init(elev=50, azim=45)
        if kwargs['labels']:
            ax.set_zlim(lower_value, upper_value)
            xticks = np.linspace(-size / 2, size / 2, 5)
            yticks = np.linspace(-size / 2, size / 2, 5)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            xticklabels = ax.get_xticks() / mm
            yticklabels = ax.get_yticks() / mm
            ax.set_xticklabels(xticklabels.astype(np.float16))
            ax.set_yticklabels(yticklabels.astype(np.float16))
            ax.set_xlabel('Size [%s]' % unit)
            ax.set_zlabel('Wavefront [λ]')
        else:
            ax._axis3don = False
        ax.grid(True) if kwargs['grid'] else ax.grid(False)
        if kwargs['pv_rms']:
            ax.text2D(0.925, 0.75, PV,
                      fontsize=12,
                      horizontalalignment='right',
                      transform=ax.transAxes)
            ax.text2D(0.925, 0.70, RMS,
                      fontsize=12,
                      horizontalalignment='right',
                      transform=ax.transAxes)
        if kwargs['colorbar']:
            fig.colorbar(im, cax=caxins)
        if mask_r:
            radius = np.sqrt(X**2 + Y**2)
            phase_[radius > size * mask_r / 2] = 0
        if title:
            if kwargs['colorbar']:
                fig.suptitle(title, x=0.575, y=0.9)
            else:
                plt.title(title)
    else:
        fig = plt.figure()
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
        xticklabels = ax.get_xticks() / mm
        ax.set_xticklabels(xticklabels.astype(int))
        ax.set_xlabel('Size [%s]' % unit)
        ax.set_ylabel('Wavefront [λ]')
        if title:
            fig.suptitle(title)

    plt.show()

    if noise is True or return_data is True:
        return phase_
