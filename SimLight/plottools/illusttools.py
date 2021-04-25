# -*- coding: utf-8 -*-

"""
Created on May 22, 2020
@author: Zhou Xiang
"""

import math
from matplotlib.pyplot import axis, bar
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib_scalebar.scalebar import ScaleBar
from numpy.lib.function_base import average
import scipy.interpolate

import SimLight as sl
from ..utils import pv, rms, return_circle_aperature, approximate_size
from ..calc import phase, intensity, psf, zernike_coeffs
from ..units import *

np.random.seed(235)


class Wavefront:
    def __init__(self, phase, size):
        self._wavefront_distribution = phase
        self._pv = self.__pv(phase)
        self._rms = self.__rms(phase)
        self._size = size
        self._scale = size / phase.shape[0]

    @staticmethod
    def __pv(wavefront):
        return pv(wavefront, mask=True)

    @staticmethod
    def __rms(wavefront):
        return rms(wavefront, mask=True)

    @property
    def wavefront(self):
        return self._wavefront

    @property
    def pv(self):
        return self._pv

    @property
    def rms(self):
        return self._rms

    @property
    def size(self):
        return self._size

    @property
    def scale(self):
        return self._scale


class Intensity:
    def __init__(self, raw_intensity, normlized_intensity, size):
        self._intensity_distribution = raw_intensity
        self._camera_intensity = raw_intensity
        self._normlized_intensity = normlized_intensity
        self._max_intensity = self.__max_intensity(raw_intensity)
        self._size = size
        self._scale = size / raw_intensity.shape[0]

    @staticmethod
    def __max_intensity(intensity):
        return np.nanmax(intensity)

    @property
    def intensity_distribution(self):
        return self._intensity_distribution

    @property
    def camera_intensity(self):
        return self._camera_intensity

    @camera_intensity.setter
    def camera_intensity(self, camera_intensity):
        self._camera_intensity = camera_intensity

    @property
    def normlized_intensity(self):
        return self._normlized_intensity

    @property
    def max_intensity(self):
        return self._max_intensity

    @property
    def size(self):
        return self._size

    @property
    def scale(self):
        return self._scale


def _gradient_fill(ax, x, y, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes
        will be used.
    x, y : array-like
        The data values of the line.
    Additional arguments are passed on to matplotlib's ``plot``
    function.

    Returns
    ----------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath
        the curve.
    """
    line, = ax.plot(x, y, **kwargs)

    zorder = line.get_zorder()
    fill_color = line.get_color()
    line_alpha = line.get_alpha()
    alpha = 0.3 if line_alpha is None else 0.3 * line_alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:, :, :3] = rgb
    z[:, :, -1] = np.linspace(0, alpha, 100)[:, None]

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    ydiff = ymax - ymin
    extent = [xmin, xmax, ymin - 0.05 * ydiff, ymax + 0.05 * ydiff]
    im = ax.imshow(z, extent=extent,
                   aspect='auto', origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = patches.Polygon(xy,
                                facecolor='none',
                                edgecolor='none',
                                closed=True)

    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)

    return line, im


def plot_wavefront(field, noise=None, mask_r=None, unwrap=True,
                   dimension=2, unit='mm',
                   title='', return_data=False, showing=True,
                   **kwargs):
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
    elif isinstance(field, sl.ScatteringLayer) is True:
        wavelength = field.wavelength
        size = field.size[0]
        N = field.N[0]
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
    elif isinstance(field, np.ndarray):
        wavelength = kwargs['wavelength']
        size = kwargs['size']
        N = kwargs['N']
        phase_ = field
        lambdaflag = True
    else:
        raise ValueError('Invalid light field.')
    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
    else:
        cmap = 'rainbow'

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
    if noise:
        noise_data = np.random.rand(N, N) * noise
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

    appr_size = approximate_size(size, unit_)

    if showing is True:
        if dimension == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            length = np.linspace(-size / 2, size / 2, phase_.shape[0])
            X, Y = np.meshgrid(length, length)
            if mask_r:
                extent = [-appr_size / 2, appr_size / 2,
                          -appr_size / 2, appr_size / 2]
                im = ax.imshow(phase_, cmap=cmap, extent=extent,
                               vmin=min_value, vmax=max_value)
                mask = patches.Circle([0, 0], size * mask_r / 2,
                                      fc='none', ec='k',)
                ax.add_patch(mask)
                im.set_clip_path(mask)
                radius = np.sqrt(X**2 + Y**2)
                phase_[radius > size * mask_r / 2] = 0
                ax.spines['bottom'].set_color('none')
                ax.spines['top'].set_color('none')
                ax.spines['left'].set_color('none')
                ax.spines['right'].set_color('none')
                ax.tick_params(axis='both', width=0)
                # xticks = np.linspace(-appr_size / 2, appr_size / 2, 5)
                # yticks = np.linspace(-appr_size / 2, appr_size / 2, 5)
                # ax.set_xticks(xticks)
                # ax.set_yticks(yticks)
                ax.text(0., 0.975, PV,
                        fontsize=12,
                        horizontalalignment='left',
                        transform=ax.transAxes)
                ax.text(0., 0.925, RMS,
                        fontsize=12,
                        horizontalalignment='left',
                        transform=ax.transAxes)
            else:
                extent = [-size / 2, size / 2, -size / 2, size / 2]
                im = ax.imshow(phase_, cmap='rainbow', extent=extent,
                               vmin=min_value, vmax=max_value)
                # xticks = np.linspace(-size / 2, size / 2, 5)
                # yticks = np.linspace(-size / 2, size / 2, 5)
                ax.text(0.05, 0.925, PV,
                        fontsize=12,
                        horizontalalignment='left',
                        transform=ax.transAxes)
                ax.text(0.05, 0.875, RMS,
                        fontsize=12,
                        horizontalalignment='left',
                        transform=ax.transAxes)
            ax.xaxis.set_major_locator(ticker.LinearLocator(5))
            ax.yaxis.set_major_locator(ticker.LinearLocator(5))
            xticklabels = ax.get_xticks() / unit_
            yticklabels = ax.get_yticks() / unit_
            ax.set_xticklabels(xticklabels.astype(np.float16))
            ax.set_yticklabels(yticklabels.astype(np.float16))
            ax.set_xlabel('x [%s]' % unit)
            ax.set_ylabel('y [%s]' % unit)
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
            # stride = math.ceil(N / 25)
            rccount = 100
            if mask_r:
                radius = np.sqrt(X**2 + Y**2)
                # X[radius > size * mask_r / 2] = np.nan
                # Y[radius > size * mask_r / 2] = np.nan
                phase_[radius > size * mask_r / 2] = np.nan
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
            # aspect = 40
            # pad_fraction = 0.5
            # divider = make_axes_locatable(ax)
            # width = axes_size.AxesY(ax, aspect=1. / aspect)
            # pad = axes_size.Fraction(pad_fraction, width)
            # cax = divider.append_axes('right', size=width, pad=pad)
            caxins = inset_axes(ax,
                                width='2.5%',
                                height='85%',
                                loc='right',
                                bbox_to_anchor=(-0.075, -0.025, 1, 1),
                                bbox_transform=ax.transAxes,
                                borderpad=0)
            ax = fig.add_subplot(111, projection='3d')
            # ax.view_init(elev=20, azim=-60)
            if PV != 'P-V: 0.0 λ' or RMS != 'RMS: 0.0 λ':
                cset = ax.contourf(X, Y, phase_,
                                   zdir='z',
                                   offset=lower_value,
                                   cmap='rainbow', alpha=0.5)
            im = ax.plot_surface(X, Y, phase_,
                                 rcount=rccount, ccount=rccount,
                                 cmap=cmap, alpha=0.9,
                                 vmin=min_value, vmax=max_value)
            ax.set_zlim(lower_value, upper_value)
            # xticks = np.linspace(-size / 2, size / 2, 5)
            # yticks = np.linspace(-size / 2, size / 2, 5)
            # ax.set_xticks(xticks)
            # ax.set_yticks(yticks)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
            ax.zaxis.set_major_locator(ticker.MaxNLocator(6))
            xticklabels = ax.get_xticks() / unit_
            yticklabels = ax.get_yticks() / unit_
            ax.set_xticklabels(xticklabels.astype(np.float16))
            ax.set_yticklabels(yticklabels.astype(np.float16))
            ax.set_xlabel('x [%s]' % unit)
            ax.set_ylabel('y [%s]' % unit)
            ax.set_zlabel('Waves [λ]')
            ax.xaxis.line.set_color('none')
            ax.yaxis.line.set_color('none')
            ax.zaxis.line.set_color('none')
            ax.tick_params(which='both', axis='both', colors='gray', width=2)
            for tickline in ax.zaxis.get_ticklines():
                tickline.set_solid_capstyle('round')
                tickline.set_linewidth(2)
            for tickline in ax.xaxis.get_majorticklines():
                tickline.set_solid_capstyle('round')
                tickline.set_linewidth(2)
            for tickline in ax.yaxis.get_majorticklines():
                tickline.set_solid_capstyle('round')
                tickline.set_linewidth(2)
            ax.text2D(0.925, 0.75, PV,
                      fontsize=12,
                      horizontalalignment='right',
                      transform=ax.transAxes)
            ax.text2D(0.925, 0.70, RMS,
                      fontsize=12,
                      horizontalalignment='right',
                      transform=ax.transAxes)
            fig.colorbar(im, cax=caxins)
            if mask_r:
                radius = np.sqrt(X**2 + Y**2)
                phase_[radius > size * mask_r / 2] = 0
            if title:
                # ax.set_title(title)
                fig.suptitle(title, x=0.575, y=0.9)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            center = int(phase_.shape[0] / 2)
            color = 'cornflowerblue'
            if mask_r:
                length = int((phase_.shape[0] * mask_r) / 2) * 2
                X = np.linspace(-size * mask_r / 2, size * mask_r / 2, length)
                left, right = center - length / 2, center + length / 2
                _gradient_fill(ax, X, phase_[center][int(left):int(right)],
                               linewidth=3,
                               color=color)
            else:
                X = np.linspace(-size / 2, size / 2, phase_.shape[0])
                _gradient_fill(ax, X, phase_[center],
                               linewidth=3,
                               color=color)
            xticks = np.linspace(-appr_size / 2, appr_size / 2, 5)
            ax.set_xticks(xticks)
            xticklabels = ax.get_xticks() / unit_
            ax.set_xticklabels(xticklabels.astype(np.float16))
            ax.grid(True, axis='y', linewidth=0.5, color='lightgray')
            ax.set_xlabel('Size [%s]' % unit)
            ax.set_ylabel('Waves [λ]')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            x_minor_ticklabels = (xticklabels[1] - xticklabels[0]) / 5 * unit_
            ax.xaxis.set_minor_locator(
                ticker.MultipleLocator(x_minor_ticklabels))
            ax.tick_params(which='major', axis='x', colors='gray',
                           width=2, size=2.5)
            ax.tick_params(which='minor', axis='x', colors='gray',
                           width=2, size=0.5)
            ax.tick_params(which='both', axis='y', colors='gray', width=0)
            for tickline in ax.xaxis.get_majorticklines():
                tickline._marker._capstyle = 'round'
            for tickline in ax.xaxis.get_minorticklines():
                tickline._marker._capstyle = 'round'

            if title:
                fig.suptitle(title)

        plt.show()

    if noise is True or return_data is True:
        wavefront = Wavefront(phase_, size)
        return wavefront
        # return phase_


def plot_intensity(field, mask_r=None, norm_type=0, dimension=2, mag=1,
                   unit='µm', title='', return_data=False, showing=True,
                   **kwargs):
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
        unit : str, optional, {'m', 'cm', 'mm', 'um', 'µm', 'nm'}, default 'µm'
            Unit used for FOV.
        title : str, optional, default ''
            Title of the figure.
        return_data : bool, optional, default False
            Return the intensity data or not.

    Returns
    ----------
        intensity_ : numpy.ndarray
            Intensity data.
    """

    # check of input parameters
    if mask_r:
        if mask_r > 1 or mask_r < 0:
            raise ValueError('Invalid radius of circle mask.')
    if norm_type:
        if norm_type < 0 or norm_type > 2 or type(norm_type) is not int:
            raise ValueError('Invalid type of normalization.')
    if dimension:
        if dimension < 1 or dimension > 2 or type(dimension) is not int:
            raise ValueError('Invalid dimension.')
    if isinstance(field, sl.Field) is True:
        field = sl.Field.copy(field)
        size = field.size
        raw_intensity, intensity_ = intensity(field, norm_type=norm_type)
    elif isinstance(field, sl.ScatteringLayer) is True:
        size = field.size[0]
        raw_intensity, intensity_ = intensity(field, norm_type=norm_type)
    elif isinstance(field, Intensity) is True:
        size = field.size
        raw_intensity = field.camera_intensity
        intensity_ = field.normlized_intensity
    elif isinstance(field, list) is True:
        size = field[0]
        raw_intensity, intensity_ = intensity(field[2],
                                              norm_type=norm_type)
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

    # light field size Magnification
    if mag < 1:
        lower = int(intensity_.shape[0] * (1 - mag) / 2)
        upper = int(intensity_.shape[0] * (1 - (1 - mag) / 2))
        intensity_ = intensity_[lower:upper, lower:upper]
        size *= mag
    elif mag > 1:
        lower = int(intensity_.shape[0] * (mag - 1) / 2)
        upper = int(intensity_.shape[0] * (mag + 1) / 2)
        new_intensity = np.zeros((lower + upper, lower + upper))
        new_intensity[lower:upper, lower:upper] = intensity_
        intensity_ = new_intensity
        size *= mag

    appr_size = approximate_size(size, unit_)

    if showing is True:

        fig = plt.figure()
        ax = fig.add_subplot(111)

        if dimension == 2:
            extent = [-size / 2, size / 2, -size / 2, size / 2]
            cmap = kwargs['cmap'] if 'cmap' in kwargs else 'gist_gray'
            if norm_type == 2:
                vmax = 255
            elif norm_type == 1:
                vmax = 1
            else:
                vmax = intensity_.max()
            im = ax.imshow(intensity_,
                           cmap=cmap,
                           extent=extent,
                           vmin=0,
                           vmax=vmax)
            # scalebar = AnchoredSizeBar(ax.transData,
            #                            N / 5 * µm,
            #                            'test',
            #                            'lower left')
            scalebar = ScaleBar(1.0, 'm', 'si-length',
                                height_fraction=0.02,
                                length_fraction=0.25,
                                location='lower left',
                                scale_loc='top',
                                frameon=False,
                                color='white')
            ax.add_artist(scalebar)
            if mask_r:
                mask = patches.Circle([0, 0], mask_r, fc='none', ec='none')
                ax.add_patch(mask)
                im.set_clip_path(mask)
            # xticks = np.linspace(-size / 2, size / 2, 5)
            # yticks = np.linspace(-size / 2, size / 2, 5)
            # ax.set_xticks(xticks)
            # ax.set_yticks(yticks)
            # xticklabels = ax.get_xticks() / unit_
            # yticklabels = ax.get_yticks() / unit_
            # ax.set_xticklabels(xticklabels.astype(np.float16))
            # ax.set_yticklabels(yticklabels.astype(np.float16))
            # ax.set_xlabel('Size [%s]' % unit)
            ax.set_yticks([])
            ax.set_xticks([])
            fig.colorbar(im)
        else:
            # fig.set_size_inches(6, 2)
            center = int(intensity_.shape[0] / 2)
            # cmap = plt.get_cmap('Accent')
            # color = cmap(np.asarray([2]))[0]
            color = ('cornflowerblue'
                     if kwargs['color'] is None
                     else kwargs['color'])

            if mask_r:
                length = int((intensity_.shape[0] * mask_r) / 2) * 2
                X = np.linspace(-size * mask_r / 2, size * mask_r / 2, length)
                [left, right] = [center - length / 2, center + length / 2]
                # line = ax.plot(X, intensity_[center][int(left):int(right)],
                #                linewidth=3,
                #                color=color)
                # shadow = ax.fill(X, intensity_[center][int(left):int(right)])
                _gradient_fill(ax, X,
                               intensity_[center][int(left):int(right)],
                               linewidth=3,
                               color=color)
            else:
                X = np.linspace(-size / 2, size / 2, intensity_.shape[0])
                # line = ax.plot(X, intensity_[center])
                _gradient_fill(ax, X, intensity_[center],
                               linewidth=3,
                               color=color)
            xticks = np.linspace(-appr_size / 2, appr_size / 2, 5)
            ax.set_xticks(xticks)
            xticklabels = ax.get_xticks() / unit_
            # yticklabels = ax.get_yticks()
            # topax = ax.secondary_xaxis('top')
            # rightax = ax.secondary_yaxis('right')
            ax.set_xticklabels(xticklabels.astype(np.float16))
            ax.grid(True, axis='y', linewidth=0.5, color='lightgray')
            ax.set_xlabel('Size [%s]' % unit)
            ax.set_ylabel('Intensity [a.u.]')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            # topax.spines['top'].set_color('none')
            # rightax.spines['right'].set_color('none')
            x_minor_ticklabels = (xticklabels[1] - xticklabels[0]) / 5 * unit_
            # y_minor_ticklabels = (yticklabels[1] - yticklabels[0]) / 5
            ax.xaxis.set_minor_locator(
                ticker.MultipleLocator(x_minor_ticklabels))
            # ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_minor_ticklabels))
            # topax.xaxis.set_minor_locator(ticker.MultipleLocator(x_minor_ticklabels))
            # topax.xaxis.set_major_formatter(ticker.NullFormatter())
            # rightax.yaxis.set_minor_locator(ticker.MultipleLocator(y_minor_ticklabels))
            # rightax.yaxis.set_major_formatter(ticker.NullFormatter())
            ax.tick_params(which='major', axis='x', colors='gray',
                           width=2, size=2.5)
            ax.tick_params(which='minor', axis='x', colors='gray',
                           width=2, size=0.5)
            ax.tick_params(which='both', axis='y', colors='gray', width=0)
            # topax.tick_params(which='both', colors='gray', width=2)
            # rightax.tick_params(which='both', colors='gray', width=2)
            for tickline in ax.xaxis.get_majorticklines():
                tickline._marker._capstyle = 'round'
            for tickline in ax.xaxis.get_minorticklines():
                tickline._marker._capstyle = 'round'

        if title:
            fig.suptitle(title)

        plt.show()

    if return_data:
        if return_data == 'ins':
            intensity_ins = Intensity(raw_intensity, intensity_, size)
        elif return_data == 'ndarray':
            intensity_ins = intensity_
        else:
            raise ValueError('Unknown return data type.')

        return intensity_ins


def plot_vertical_intensity(field_3d, norm_type=0, mag=1, title=''):
    """Plot the intensity in vertical direction.

    Plot the intensity of light field in vertical direction using
    matplotlib.

    Parameters
    ----------
        field_3d : list of SimLight.Field
            A list of SimLight.Fields of a whole light field.
        norm_type : int, optional, {0, 1, 2}, default 0
            Type of normalization, where
                0 for no normalization,
                1 for normalize up to 1,
                2 for normalize up to 255.
        mag : float, optional, default 1
            Original parameter used when calling
            SimLight.near_field_propagation()
        title : str, optional, default ''
            Title of the figure.
    """
    # check of input parameters
    if norm_type:
        if norm_type < 0 or norm_type > 2 or type(norm_type) is not int:
            raise ValueError('Invalid type of normalization.')
    # if isinstance(field, sl.Field) is True:
    #     size = field.size
    #     intensity_ = intensity(field, norm_type=norm_type)
    # else:
    #     raise ValueError('Invalid light field.')

    complex_amp_3d = []
    for field in field_3d:
        complex_amp_3d.append(field.complex_amp)
    complex_amp_3d = np.asarray(complex_amp_3d)

    z = complex_amp_3d.shape[0]  # optical axis pixels (z)
    h_v = complex_amp_3d.shape[1]  # vertical axis pixels (y)
    h_w = complex_amp_3d.shape[2]  # horizontal axis pixels (x)
    center_z = int(z / 2 - 1)
    center_y = int(h_v / 2 - 1)
    center_x = int(h_w / 2 - 1)
    vertical_field = complex_amp_3d[:, :, center_x]
    vertical_intensity = intensity(vertical_field, norm_type=norm_type)

    dx = 0.1 * µm  # dx in optical axis (z)
    z_range = z * dx
    x_range = field_3d[0].size / mag
    aspect_ratio = z_range / x_range
    # print(z_range / µm, x_range / µm, aspect_ratio)

    lower = int(vertical_intensity.shape[1] * ((mag - 1) / (mag * 2)))
    upper = int(vertical_intensity.shape[1] * ((mag + 1) / (mag * 2)))
    vertical_intensity_mag = vertical_intensity[:, lower:upper]

    # fig = plt.figure()
    fig = plt.figure(figsize=(2.4, 2.4 * aspect_ratio))
    ax = fig.add_subplot(111)
    extent = [-x_range / 2, x_range / 2, 1 * µm, z_range + 1 * µm]
    im = ax.imshow(vertical_intensity_mag, cmap='hot', extent=extent)
    scalebar = ScaleBar(1.0, 'm', 'si-length',
                        height_fraction=0.02,
                        length_fraction=0.25,
                        location='lower left',
                        scale_loc='top',
                        frameon=False,
                        color='white')
    ax.add_artist(scalebar)
    # xticklabels = ax.get_xticks() / µm
    # yticklabels = ax.get_yticks() / µm
    # ax.set_xticklabels(xticklabels.astype(np.float16))
    # ax.set_yticklabels(yticklabels.astype(int))
    # ax.set_xlabel('Size [µm]')
    # ax.set_ylabel('Distance [µm]')
    ax.set_yticks([])
    ax.set_xticks([])

    if title:
        fig.suptitle(title, x=0, y=0.5,
                     rotation='vertical',
                     verticalalignment='center',
                     horizontalalignment='left')

    plt.show()


def plot_multi_intensities_diff(*fields, mask_r=None, shift=None,
                                labels=None, norm_type=0,
                                figsize=(6.4, 4.8), mag=1,
                                unit='µm', title='', ind=False):
    """Plot the intensity difference of the light fields in one line.

    Plot the intensity difference of the light fields in one line.

    Parameters
    ----------
        fields : array-like, SimLight.Field
            Light fields to compare.
        shift : array-like or list, optional, default None
            Shift pixels of the line in intensity.
        labels : list, optional, default None
            Labels of all light fields.
        norm_type : int, optional, {0, 1, 2}, default 0
            Type of normalization, where
                0 for no normalization,
                1 for normalize up to 1,
                2 for normalize up to 255.
        figsize : list, optional, default (6.4, 4.8)
            Physical size of the figure in inch.
        mag : float, optional, default 1
            Magnification of the figure.
        unit : str, optional, {'m', 'cm', 'mm', 'um', 'µm', 'nm'}, default 'µm'
            Unit used for FOV.
        title : list, optional, default ''
            Title of the figure.

    Examples
    ----------
    >>> slpl.plot_multi_intensities_diff()
    """
    # check of input parameters
    if norm_type:
        if norm_type < 0 or norm_type > 2 or type(norm_type) is not int:
            raise ValueError('Invalid type of normalization.')

    max_size = fields[0].size
    intensities_ = []
    # intensities_.append(intensity(field_ref, norm_type=0))
    for field in fields:
        if isinstance(field, sl.Field) is True:
            # if field.size != max_size:
            if field.size > max_size:
                max_size = field.size
            # else:
            #     max_size = field.size
            intensities_.append(intensity(field, norm_type=0))
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

    intensities = []
    for index, field in enumerate(fields):
        frac = max_size / field.size
        lower = int(intensities_[index].shape[0] * (frac - 1) / 2)
        upper = int(intensities_[index].shape[0] * (frac + 1) / 2)
        new_intensity = np.zeros((lower + upper, lower + upper))
        new_intensity[lower:upper, lower:upper] = intensities_[index]
        intensities.append(new_intensity)
    intensities = np.asarray(intensities)

    if norm_type > 0:
        max_value = np.max(intensities)
        intensities /= max_value
        if norm_type > 1:
            intensities *= 255

    if mag < 1:
        lower = int(intensities[0].shape[0] * (1 - mag) / 2)
        upper = int(intensities[0].shape[0] * (1 - (1 - mag) / 2))
        new_intensities = np.zeros((len(intensities),
                                    upper - lower,
                                    upper - lower))
        new_intensities[:] = intensities[:, lower:upper, lower:upper]
        intensities = new_intensities
    elif mag > 1:
        lower = int(intensities[0].shape[0] * (mag - 1) / 2)
        upper = int(intensities[0].shape[0] * (mag + 1) / 2)
        new_intensities = np.zeros((len(intensities),
                                    lower + upper,
                                    lower + upper))
        new_intensities[:, lower:upper, lower:upper] = intensities[:]
        intensities = new_intensities
    max_size *= mag

    appr_size = approximate_size(max_size, unit_)

    fig = plt.figure(figsize=figsize)
    if ind is False:
        ax = fig.add_subplot(111)
    else:
        N = len(fields)
        ax = [fig.add_subplot(1, N, i + 1) for i in range(N)]

    # center = int(intensities[0].shape[0] / 2)
    # X = np.linspace(-size / 2, size / 2, intensities[0].shape[0])
    cmap = plt.get_cmap('Dark2')
    colors = cmap(np.arange(len(fields)))
    shift_ = np.zeros(len(fields) + 1, dtype=int)
    if shift:
        shift_[:len(shift)] = shift
    if ind is False:
        for index, intensity_ in enumerate(intensities):
            center = int(intensity_.shape[0] / 2)
            X = np.linspace(-max_size / 2, max_size / 2,
                            intensity_.shape[0])
            # ax.plot(X, intensity_[center + shift_[index]],
            #         color=colors[index],
            #         linewidth=3)
            _gradient_fill(ax, X, intensity_[center + shift_[index]],
                           color=colors[index],
                           linewidth=3,
                           label=labels[index] if labels else None)
        ax.grid(True, axis='y', linewidth=0.5, color='lightgray')
        xticks = np.linspace(-appr_size / 2, appr_size / 2, 5)
        ax.set_xticks(xticks)
        xticklabels = ax.get_xticks() / unit_
        ax.set_xticklabels(xticklabels.astype(np.float16))
        ax.set_xlabel('Size [%s]' % unit)
        ax.set_ylabel('Intensity [a.u.]')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        x_minor_ticklabels = (xticklabels[1] - xticklabels[0]) / 5 * unit_
        ax.xaxis.set_minor_locator(
            ticker.MultipleLocator(x_minor_ticklabels))
        ax.tick_params(which='major', axis='x', colors='gray',
                       width=2, size=2.5)
        ax.tick_params(which='minor', axis='x', colors='gray',
                       width=2, size=0.5)
        ax.tick_params(which='both', axis='y', colors='gray', width=0)
        for tickline in ax.xaxis.get_majorticklines():
            tickline._marker._capstyle = 'round'
        for tickline in ax.xaxis.get_minorticklines():
            tickline._marker._capstyle = 'round'
        if labels:
            l, _, _, _ = ax.get_position().bounds
            bbox_to_anchor = (l, -0.1, 1 - 2 * l, 0.1)
            fig.legend(bbox_to_anchor=bbox_to_anchor,
                       ncol=len(labels),
                       mode='expand')
        if title:
            fig.suptitle(title)
    else:
        for i, axis in enumerate(ax):
            center = int(intensities[i].shape[0] / 2)
            X = np.linspace(-max_size / 2, max_size / 2,
                            intensities[i].shape[0])
            for j, intensity_ in enumerate(intensities):
                # center = int(intensity_.shape[0] / 2)
                # X = np.linspace(-max_size / 2, max_size / 2,
                #                 intensity_.shape[0])
                if i != j:
                    axis.plot(X, intensity_[center + shift_[j]],
                              color='lightgray',
                              linewidth=2)
            _gradient_fill(axis, X, intensities[i][center + shift_[i]],
                           color=colors[i],
                           linewidth=3,
                           label=labels[i] if labels else None)
            xticks = np.linspace(-appr_size / 2, appr_size / 2, 5)
            axis.set_xticks(xticks)
            xticklabels = axis.get_xticks() / unit_
            axis.set_xticklabels(xticklabels.astype(np.float))
            axis.set_xlabel('Size [%s]' % unit)
            if i == 0:
                axis.set_ylabel('Intensity [a.u.]')
            else:
                axis.yaxis.set_major_formatter(ticker.NullFormatter())
                axis.tick_params(axis='y', length=0)
            axis.grid(True, axis='y', linewidth=0.5, color='lightgray')
            axis.spines['bottom'].set_color('none')
            axis.spines['left'].set_color('none')
            axis.spines['top'].set_color('none')
            axis.spines['right'].set_color('none')
            x_minor_ticklabels = (xticklabels[1] -
                                  xticklabels[0]) / 5 * unit_
            axis.xaxis.set_minor_locator(
                ticker.MultipleLocator(x_minor_ticklabels))
            axis.tick_params(which='major', axis='x',
                             colors='gray', width=2, size=2.5)
            axis.tick_params(which='minor', axis='x',
                             colors='gray', width=2, size=0.5)
            axis.tick_params(which='both', axis='y',
                             colors='gray', width=0)
            for tickline in axis.xaxis.get_majorticklines():
                tickline._marker._capstyle = 'round'
            for tickline in axis.xaxis.get_minorticklines():
                tickline._marker._capstyle = 'round'
            axis.set_ylim(-0.05 * intensities.max(),
                          1.05 * intensities.max())
        if labels:
            l, _, _, _ = ax[0].get_position().bounds
            bbox_to_anchor = (l / 2, -0.1, 1 - l + 0.03, 0.1)
            fig.legend(bbox_to_anchor=bbox_to_anchor,
                       ncol=len(labels),
                       mode='expand')
        if title:
            fig.suptitle(title, va='top')
        fig.tight_layout()

    plt.show()


def plot_wavefront_diff(*fields, mask_r=1, dimension=2, unit='mm',
                        title='', **kwargs):
    """
    docstring
    """
    # input check
    if len(fields) != 2:
        raise ValueError('Input light field number error.')

    wavelength = (fields[-1].wavelength
                  if isinstance(fields[-1], sl.Field)
                  else kwargs['wavelength'])
    size = (fields[-1].size
            if isinstance(fields[-1], sl.Field) else kwargs['size'])
    N = fields[-1].N if isinstance(fields[-1], sl.Field) else kwargs['N']

    # if isinstance(fields[0], sl.Field) is True:
    #     for index, field in enumerate(fields):
    #         phase_ = phase(field, unwrap=True)
    #         name = 'wavefront' + str(index + 1)
    #         globals()[name] = phase_
    #     diff_wavefront = wavefront1 - wavefront2
    # else:
    #     phase_ = phase(fields[-1], unwrap=True)
    #     diff_wavefront = fields[0] / wavelength * (2 * np.pi) * µm - phase_

    # diff_field = sl.PlaneWave(wavelength, size, N)
    # diff_field.complex_amp *= np.exp(-1j * diff_wavefront)
    # diff_field.complex_amp2 = diff_field.complex_amp

    for index, field in enumerate(fields):
        name = 'surface' + str(index + 1)
        if isinstance(field, sl.Field) is True:
            phase_ = phase(field, unwrap=True)
            globals()[name] = phase_ * wavelength / (2 * np.pi) / µm
        elif isinstance(field, np.ndarray) is True:
            globals()[name] = field
        elif isinstance(field, list) is True:
            globals()[name] = field[3]
        else:
            raise ValueError('Invalid light field.')

    # diff_surface = surface1 - surface2
    diff_surface = globals()['surface1'] - globals()['surface2']
    fid = 'SimLight.plottools'
    diff_field = [wavelength, size, N, diff_surface, fid]

    plot_wavefront(diff_field,
                   mask_r=mask_r,
                   dimension=dimension,
                   unit=unit,
                   title=title)


def plot_psf(field, aperture_type='circle', dimension=2, title=''):
    """Show the figure of point spread function (PSF).

    Show the figure of point spread function of a light field.

    Parameters
    ----------
        field : SimLight.Field
            The light fiedl.
        aperture_type : str, optional, {'circle', 'square'}, default 'circle'
            The shape of the aperture.
                'circle' for circle aperture,
                'square' for square aperture.
        dimension : int, optional, {1, 2}, default 2
            Dimension of figure, where
                1 for line,
                2 for surface.
        title : str, optional, default ''
            Title of the figure.
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
        im = ax.imshow(psf_, cmap='gist_gray', extent=extent, vmin=0)
        fig.colorbar(im)
    else:
        center = int(psf_.shape[0] / 2)
        X = np.linspace(-size / 2, size / 2, psf_.shape[0])
        im = ax.plot(X, psf_[center])
        ax.set_ylabel('Intensity [a.u.]')

    if title:
        fig.suptitle(title)

    plt.show()


def plot_longitudinal_aberration(lens, wavelength=0.550, title=''):
    """Plot longitudinal aberration.

    Plot the graph of the longitudinal aberration acrroding to the
    Sidel coefficients.

    Parameters
    ----------
        lens : SimLight.Lens
            The lens has the aberration.
        wavelength : float, optional, default 0.550
            The wavelength of the light for calculating the longitudinal
            aberration.
        title : str, optional, default ''
            Title of the figure.
    """
    # default parameters
    size = lens.D
    N = 1000
    f = lens.f
    center = int(N / 2)

    field = sl.PlaneWave(wavelength, size, N)
    sidel = sl.zernike.SidelCoefficients(lens.sidel)
    # spherical aberration
    sidel.coefficients[2][0] = 0
    delta_W = sl.delta_wavefront(field, sidel)
    delta_W *= wavelength * 1e-3 / (2 * np.pi)
    # delta_W *= wavelength * 1e-3

    x = np.linspace(-size / 2, size / 2, N)
    height = x[center:]
    delta_W_line = delta_W[center, center:]
    longitude = delta_W_line * -(f / size)**2 * 16
    # longitude = delta_W_line * -f**2 / size

    fig = plt.figure(figsize=(2, 6))
    ax = fig.add_subplot(111)
    im = ax.plot(longitude, height)

    if np.abs(np.max(longitude)) > np.abs(np.min(longitude)):
        max_value = np.abs(np.max(longitude) * 2.5)
    else:
        max_value = np.abs(np.min(longitude) * 2.5)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    ax.spines['right'].set_color('none')
    ax.set_xlim((-max_value, max_value))
    ax.set_xlabel('Longitudinal aberration [mm]')
    ax.set_ylabel('Size [mm]', rotation=0, position=(0, 1.01), labelpad=-20)

    if title:
        fig.suptitle(title)

    plt.show()


def plot_dm_wavefront(field, K, j=15, limits=[], mask_r=None, unit='mm',
                      title='', **kwargs):
    """Plot deformable mirror concerned graphs.

    Plot the stroke of each actuators and the generated compensation
    wavefront.

    Parameters
    ----------
        field : SimLight.Field
            The input light field to be compensated.
        K : int
            The maximum number of actuators in one direction.
        mask_r : float, optional, from 0 to 1, default None.
            Radius of a circle mask.
        title : str, optional, default ''
            The title of the two figures.
    """
    unwrap = True

    # check of input parameters
    if mask_r:
        if mask_r > 1 or mask_r < 0:
            raise ValueError('Invalid radius of circle mask.')
    if isinstance(field, sl.Field) is True:
        wavelength = field.wavelength
        size = field.size
        N = field.N
        phase_ = phase(field, unwrap=unwrap)
        surface = wavelength * phase_ / (2 * np.pi) / µm
    elif isinstance(field, list) is True:
        wavelength = field[0]
        size = field[1]
        N = field[2]
        phase_ = phase(field[3], unwrap=unwrap)
        surface = wavelength * phase_ / (2 * np.pi) / µm
    elif isinstance(field, np.ndarray) is True:
        wavelength = kwargs['wavelength']
        size = kwargs['size']
        N = field.shape[0]
        phase_ = field / wavelength * (2 * np.pi) * µm
        surface = field
    else:
        raise ValueError('Invalid light field.')

    if limits:
        coeffs = zernike_coeffs(surface, j,
                                nflag='pv',
                                return_raw=True,
                                wavelength=wavelength)
        for index, limit in enumerate(limits):
            if 2 * limit / wavelength < abs(coeffs[index]):
                coeffs[index] = limit / wavelength
        ltd_F = sl.PlaneWave(wavelength, size, N)
        ltd_Z = sl.zernike.ZernikeCoefficients(j, coeffs)
        ltd_F = sl.aberration(ltd_F, ltd_Z, nflag='pv')
        ltd_phase = phase(ltd_F, unwrap=True)
    else:
        ltd_phase = phase_

    # # unit conversion
    # wavelength /= µm

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

    ltd_surface = wavelength * ltd_phase / (2 * np.pi) / µm
    x_dm = np.linspace((-K + 1) / 2, (K - 1) / 2, K)
    X_dm, Y_dm = np.meshgrid(x_dm, x_dm)
    r_dm = np.sqrt(X_dm**2 + Y_dm**2)
    x = np.linspace(-size / 2, size / 2, N)
    X, Y = np.meshgrid(x, x)

    # K = dm.K
    dm_points = np.zeros((K, K))
    dm_points_X = np.zeros((K, K))
    dm_points_Y = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            ii = int(N / K * i + N / (2 * K))
            jj = int(N / K * j + N / (2 * K))
            dm_points[i][j] = ltd_surface[ii][jj] / 2
            dm_points_X[i][j] = X[ii][jj]
            dm_points_Y[i][j] = Y[ii][jj]
            if i == 0 or i == K - 1 or j == 0 or j == K - 1:
                dm_points[i][j] == 0
    average = np.average(dm_points)
    dm_points -= average

    dm_wavefront = scipy.interpolate.interp2d(dm_points_X, dm_points_Y,
                                              dm_points, kind='cubic')
    dm_wavefront = dm_wavefront(x, x)

    dm_points *= wavelength / µm
    dm_wavefront *= wavelength / µm

    # fig = plt.figure(figsize=(10, 4))
    # grid = plt.GridSpec(6, 11, wspace=0.5, hspace=0.5)
    # ax1 = fig.add_subplot(grid[0:6, 0:5])
    # ax2 = fig.add_subplot(grid[0:6, 5:11])
    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)

    if mask_r:
        dm_points[r_dm > mask_r * np.sqrt(2) * (3 / 4) * (K / 2)] = np.nan
        max_value1 = np.nanmax(dm_points)
        min_value1 = np.nanmin(dm_points)
        _, _, norm_radius = return_circle_aperature(dm_wavefront, mask_r)
        max_value2 = np.max(dm_wavefront[norm_radius <= mask_r])
        min_value2 = np.min(dm_wavefront[norm_radius <= mask_r])
        PV = 'P-V: ' + str(round(pv(dm_wavefront, mask=True), 3)) + ' µm'
        RMS = 'RMS: ' + str(round(rms(dm_wavefront, mask=True), 3)) + ' µm'
    else:
        max_value1 = np.nanmax(dm_points)
        min_value1 = np.nanmin(dm_points)
        max_value2 = np.max(dm_wavefront)
        min_value2 = np.min(dm_wavefront)
        PV = 'P-V: ' + str(round(pv(dm_wavefront), 3)) + ' µm'
        RMS = 'RMS: ' + str(round(rms(dm_wavefront), 3)) + ' µm'

    extent = [-size / 2, size / 2, -size / 2, size / 2]
    im1 = ax1.imshow(dm_points, cmap='rainbow',
                     vmin=min_value1, vmax=max_value1)
    fig1.colorbar(im1)
    ax1.set_xticks(np.arange(dm_points.shape[1]))
    ax1.set_yticks(np.arange(dm_points.shape[0]))
    ax1.set_xticklabels(np.arange(1, K + 1))
    ax1.set_yticklabels(np.arange(1, K + 1))
    # for edge, spine in ax1.spines.items():
    #     spine.set_visible(False)
    ax1.set_xticks(np.arange(dm_points.shape[1] + 1) - 0.5, minor=True)
    ax1.set_yticks(np.arange(dm_points.shape[0] + 1) - 0.5, minor=True)
    ax1.grid(which='minor', color='w', linestyle='-', linewidth=3)
    ax1.tick_params(which='both', bottom=False, left=False)
    textcolors = ('white', 'black')
    valfmt = matplotlib.ticker.StrMethodFormatter('{x:.2f}')
    threshold = im1.norm(np.nanmax(dm_points)) / 2
    kw = dict(horizontalalignment='center', verticalalignment='center')
    kw.update()
    texts = []
    for i in range(dm_points.shape[0]):
        for j in range(dm_points.shape[1]):
            kw.update(color=textcolors[int(im1.norm(dm_points[i, j]) >
                                           threshold)])
            text = im1.axes.text(j, i, valfmt(dm_points[i, j], None), **kw)
            texts.append(text)
    im2 = ax2.imshow(dm_wavefront, extent=extent, cmap='rainbow',
                     vmin=min_value2, vmax=max_value2)
    fig2.colorbar(im2)
    xticks = np.linspace(-size / 2, size / 2, 5)
    yticks = np.linspace(-size / 2, size / 2, 5)
    ax2.set_xticks(xticks)
    ax2.set_yticks(yticks)
    xticklabels = ax2.get_xticks() / mm
    yticklabels = ax2.get_yticks() / mm
    ax2.set_xticklabels(xticklabels.astype(np.float16))
    ax2.set_yticklabels(yticklabels.astype(np.float16))
    ax2.set_xlabel('Size [%s]' % unit)
    ax2.text(0.05, 0.95, PV,
             fontsize=12,
             horizontalalignment='left',
             transform=ax2.transAxes)
    ax2.text(0.05, 0.90, RMS,
             fontsize=12,
             horizontalalignment='left',
             transform=ax2.transAxes)
    if mask_r:
        mask = patches.Circle([0, 0], size * mask_r / 2,
                              fc='none', ec='none',)
        ax2.add_patch(mask)
        im2.set_clip_path(mask)

    if title:
        ax1.set_title(title[0])
        ax2.set_title(title[1])

    plt.show()


def plot_zernike_coeffs(*coeffs, labels=None, title=''):
    """
    Plot the coefficients of a Zernike polynomials.

    Parameters
    ----------
        coeffs : array-like or list
            Zernike coefficients.
        labels ： list, optional, default None
            Labels of all light fields.
        title : str, optional, default ''
            Title of the figure.
    """
    figheight = 4.8
    figwidth = 9.6 * len(coeffs) / 2 if len(coeffs) > 1 else 9.6
    figsize = (figwidth, figheight)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    width = 0.7 / len(coeffs) if len(coeffs) > 1 else 0.35
    xticks = np.arange(len(coeffs[0])) + 1
    cmap_name = 'Accent'
    cmap = plt.get_cmap(cmap_name)
    colors = (cmap(np.arange(len(coeffs)))
              if len(coeffs) > 1 else ['cornflowerblue'])

    for index, coeff in enumerate(coeffs):
        orders = np.arange(len(coeff))
        shift = (math.ceil(index - len(coeffs) / 2)
                 if len(coeffs) % 2 != 0
                 else index - (len(coeffs) - 1) / 2)
        locals()['im' + str(index + 1)] = ax.bar(orders + width * shift,
                                                 coeff,
                                                 color=colors[index],
                                                 width=width,
                                                 edgecolor='white',
                                                 linewidth=1,
                                                 label=(labels[index]
                                                        if labels
                                                        else None),
                                                 zorder=2)

    prop = abs(np.min(coeffs)) / (abs(np.min(coeffs)) + np.max(coeffs))
    labelpad = 250 * prop

    yticks = ax.get_yticks()
    ylim = ax.get_ylim()

    for i in range(1, len(yticks), 2):
        if yticks[i + 1] < np.max(ylim):
            ax.axhspan(yticks[i], yticks[i + 1], facecolor='whitesmoke',
                       zorder=0)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.set_xlabel('Zernike polynomials orders', labelpad=labelpad)
    ax.set_ylabel('Zernike coefficients')
    ax.tick_params(axis='y', colors='gray', width=0)
    ax.tick_params(axis='x', colors='gray', width=2, direction='inout')
    for tickline in ax.xaxis.get_majorticklines():
        tickline._marker._capstyle = 'round'
    ax.xaxis.set_zorder(3)
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_xticklabels(xticks)
    ax.set_ylim(ylim)
    ax.grid(True, axis='y', linewidth=0.5, color='lightgray', zorder=0)

    def autolabel(ims):
        """
        Attach a text label above each bar, displaying its height.
        """
        for im in ims:
            height = im.get_height()
            if abs(height - 0) > 1e-3:
                xpos = im.get_x() + im.get_width() / 2
                xy = (xpos, height) if height > 0 else (xpos, 0)
                ax.annotate('{:.3f}'.format(height),
                            xy=xy,
                            xytext=(0, 3),
                            textcoords='offset points',
                            fontsize='small',
                            rotation=45,
                            rotation_mode='anchor',
                            ha='left',
                            va='bottom')

    for i in range(len(coeffs)):
        autolabel(locals()['im' + str(i + 1)])

    if labels:
        ax.legend(loc='upper left')
        # l, _, _, _ = ax.get_position().bounds
        # bbox_to_anchor = (l / 2, -0.1, 1 - l + 0.03, 0.1)
        # fig.legend(bbox_to_anchor=bbox_to_anchor,
        #            ncol=len(labels),
        #            mode='expand')
    if title:
        # ax.set_title(title, pad=20)
        fig.suptitle(title)

    plt.show()
