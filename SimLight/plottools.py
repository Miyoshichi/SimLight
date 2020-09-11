# -*- coding: utf-8 -*-

"""
Created on May 22, 2020
@author: Zhou Xiang
"""

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from numpy.lib.function_base import average
import scipy.interpolate

import SimLight as sl
from .utils import pv, rms, circle_aperature
from .calc import phase, intensity, psf, delta_wavefront
from .unwrap import simple_unwrap_1d
from .units import *


def plot_wavefront(field, mask_r=None, dimension=2, unit='mm', title=''):
    """Plot the wavefront.

    Plot the wavefront of light field using matplotlib.

    Parameters
    ----------
        field : Field
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

    phase_ = wavelength * phase_ / (2 * np.pi) / µm

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
        xticklabels = ax.get_xticks() / unit_
        yticklabels = ax.get_yticks() / unit_
        ax.set_xticklabels(xticklabels.astype(int))
        ax.set_yticklabels(yticklabels.astype(int))
        ax.set_xlabel('Size [%s]' % unit)
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
            # X[radius > size * mask_r / 2] = np.nan
            # Y[radius > size * mask_r / 2] = np.nan
            phase_[radius > size * mask_r / 2] = np.nan
        stride = math.ceil(N / 25)
        im = ax.plot_surface(X, Y, phase_, rstride=stride, cstride=stride,
                             cmap='rainbow', vmin=min_value, vmax=max_value)
        xticklabels = ax.get_xticks() / mm
        yticklabels = ax.get_yticks() / mm
        ax.set_xticklabels(xticklabels.astype(int))
        ax.set_yticklabels(yticklabels.astype(int))
        ax.set_xlabel('Size [%s]' % unit)
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
        xticklabels = ax.get_xticks() / mm
        ax.set_xticklabels(xticklabels.astype(int))
        ax.set_xlabel('Size [%s]' % unit)
        ax.set_ylabel('Phase [λ]')

    if title:
        ax.set_title(title)

    plt.show()


def plot_intensity(field, mask_r=None, norm_type=0, dimension=2, mag=1,
                   unit='µm', title=''):
    """Plot the intensity.

    Plot the intensity of light field using matplotlib.

    Parameters
    ----------
        field : Field
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
        size = field.size
        intensity_ = intensity(field, norm_type=norm_type)
    elif isinstance(field, list) is True:
        size = field[0]
        intensity_ = intensity(field[1], norm_type=norm_type)
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

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if dimension == 2:
        extent = [-size / 2, size / 2, -size / 2, size / 2]
        im = ax.imshow(intensity_, cmap='gist_gray', extent=extent, vmin=0)
        if mask_r:
            mask = patches.Circle([0, 0], mask_r, fc='none', ec='none')
            ax.add_patch(mask)
            im.set_clip_path(mask)
        xticklabels = ax.get_xticks() / unit_
        yticklabels = ax.get_yticks() / unit_
        ax.set_xticklabels(xticklabels.astype(int))
        ax.set_yticklabels(yticklabels.astype(int))
        ax.set_xlabel('Size [%s]' % unit)
        fig.colorbar(im)
    else:
        # fig.set_size_inches(6, 2)
        center = int(intensity_.shape[0] / 2)
        if mask_r:
            length = int((intensity_.shape[0] * mask_r) / 2) * 2
            X = np.linspace(-size * mask_r / 2, size * mask_r / 2, length)
            [left, right] = [center - length / 2, center + length / 2]
            im = ax.plot(X, intensity_[center][int(left):int(right)])
        else:
            X = np.linspace(-size / 2, size / 2, intensity_.shape[0])
            im = ax.plot(X, intensity_[center])
        xticklabels = ax.get_xticks() / unit_
        ax.set_xticklabels(xticklabels.astype(int))
        ax.grid(True)
        ax.set_xlabel('Size [%s]' % unit)
        ax.set_ylabel('Intensity [a.u.]')

    if title:
        ax.set_title(title)

    plt.show()


def plot_two_intensities_diff(field1, field2,
                              label1='Reference', label2='Reality',
                              norm_type=0, mag=1, title=''):
    """Plot the intensity difference of the two light fields.

    Plot the intensity difference of the two light fields.
    .. deprecated:: 0.0.3
        `plot_two_intensities_diff` will be removed by
        `plot_multi_intensities_diff` because it can compare multiple
        light fields at a time.

    Parameters
    ----------
        field1 : tuple
            Reference light field to compare.
        field2 : tuple
            Another light field to compare.
        label1 : str
            Label of field1.
        label2 : str
            Label of field2.
        norm_type : int, optional, {0, 1, 2}, default 0
            Type of normalization, where
                0 for no normalization,
                1 for normalize up to 1,
                2 for normalize up to 255.
        mag : float, optional, default 1
            Magnification of the figure.
        title : str, optional, default ''
            Title of the figure.
    """
    # check of input parameters
    if norm_type:
        if norm_type < 0 or norm_type > 2 or type(norm_type) is not int:
            raise ValueError('Invalid type of normalization.')
    if (isinstance(field1, sl.Field) is True and
            isinstance(field2, sl.Field) is True):
        if field1.size != field2.size:
            raise ValueError('Cannot campare the two light fields'
                             'with different sizes.')
        else:
            size = field1.size
            intensity1 = intensity(field1, norm_type=0)
            intensity2 = intensity(field2, norm_type=0)
    else:
        raise ValueError('Invalid light field.')

    if norm_type > 0:
        max_value = max(np.max(intensity1), np.max(intensity2))
        intensity1 /= max_value
        intensity2 /= max_value
        if norm_type > 1:
            intensity1 *= 255
            intensity2 *= 255

    if mag < 1:
        lower = int(intensity1.shape[0] * (1 - mag) / 2)
        upper = int(intensity1.shape[0] * (1 - (1 - mag) / 2))
        intensity1 = intensity1[lower:upper, lower:upper]
        intensity2 = intensity1[lower:upper, lower:upper]
        size *= mag
    elif mag > 1:
        lower = int(intensity1.shape[0] * (mag - 1) / 2)
        upper = int(intensity1.shape[0] * (mag + 1) / 2)
        new_intensity1 = np.zeros((lower + upper, lower + upper))
        new_intensity2 = new_intensity1
        new_intensity1[lower:upper, lower:upper] = intensity1
        new_intensity2[lower:upper, lower:upper] = intensity2
        intensity1 = new_intensity1
        intensity2 = new_intensity2
        size *= mag

    fig = plt.figure()
    ax = fig.add_subplot(111)

    center = int(intensity1.shape[0] / 2)
    X = np.linspace(-size / 2, size / 2, intensity1.shape[0])
    im1 = ax.plot(X, intensity1[center], label=label1)
    im2 = ax.plot(X, intensity2[center], label=label2)
    ax.legend()
    ax.grid(True)
    ax.set_xlabel('Size [mm]')
    ax.set_ylabel('Intensity [a.u.]')

    if title:
        ax.set_title(title)

    plt.show()


def plot_multi_intensities_diff(*fields, shift=None, labels=None,
                                norm_type=0, figsize=(6.4, 4.8), mag=1,
                                unit='µm', title=''):
    """Plot the intensity difference of the light fields in one line.

    Plot the intensity difference of the light fields in one line.

    Parameters
    ----------
        fields : array-like, Field
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

    intensities = []
    for index, field in enumerate(fields):
        frac = max_size / field.size
        lower = int(intensities_[index].shape[0] * (frac - 1) / 2)
        upper = int(intensities_[index].shape[0] * (frac + 1) / 2)
        new_intensity = np.zeros((lower + upper, lower + upper))
        new_intensity[lower:upper, lower:upper] = intensities_[index]
        intensities.append(new_intensity)
    intensities = np.array(intensities)

    if norm_type > 0:
        max_value = np.max(intensities[0])
        intensities /= max_value
        if norm_type > 1:
            intensities *= 255
    # else:
    #     intensities /= 1

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

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

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

    # center = int(intensities[0].shape[0] / 2)
    # X = np.linspace(-size / 2, size / 2, intensities[0].shape[0])
    shift_ = np.zeros(len(fields) + 1, dtype=int)
    if shift:
        shift_[:len(shift)] = shift
    for index, intensity_ in enumerate(intensities):
        center = int(intensity_.shape[0] / 2)
        X = np.linspace(-max_size / 2, max_size / 2, intensity_.shape[0])
        ax.plot(X, intensity_[center + shift_[index]])
    ax.grid(True)
    xticklabels = ax.get_xticks() / unit_
    ax.set_xticklabels(xticklabels.astype(int))
    ax.set_xlabel('Size [%s]' % unit)
    ax.set_ylabel('Intensity [a.u.]')

    if labels:
        ax.legend(labels)
    if title:
        ax.set_title(title)

    plt.show()


def plot_psf(field, aperture_type='circle', dimension=2, title=''):
    """Show the figure of point spread function (PSF).

    Show the figure of point spread function of a light field.

    Parameters
    ----------
        field : Field
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
        ax.set_title(title)

    plt.show()


def plot_longitudinal_aberration(lens, wavelength=0.550, title=''):
    """Plot longitudinal aberration.

    Plot the graph of the longitudinal aberration acrroding to the
    Sidel coefficients.

    Parameters
    ----------
        lens : Lens
            The lens has the aberration.
        wavelength : float, optional, default 0.550
            The wavelength of the light for calculating the longitudinal
            aberration.
        title : str, optional, default ''
            The titles of the figure.
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
        ax.set_title(title)

    plt.show()


def plot_dm_wavefront(field, K, mask_r=None, unit='mm', title=''):
    """Plot deformable mirror concerned graphs.

    Plot the stroke of each actuators and the generated compensation
    wavefront.

    Parameters
    ----------
        field : Field
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
    elif isinstance(field, list) is True:
        wavelength = field[0]
        size = field[1]
        N = field[2]
        phase_ = phase(field[3], unwrap=unwrap)
    else:
        raise ValueError('Invalid light field.')

    # unit conversion
    wavelength /= µm

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

    phase_ = wavelength * phase_ / (2 * np.pi)
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
            ii = int(N / 8 * i + N / 16)
            jj = int(N / 8 * j + N / 16)
            dm_points[i][j] = phase_[ii][jj] / 2
            dm_points_X[i][j] = X[ii][jj]
            dm_points_Y[i][j] = Y[ii][jj]
            if i == 0 or i == K - 1 or j == 0 or j == K - 1:
                dm_points[i][j] == 0
    average = np.average(dm_points)
    dm_points -= average

    dm_wavefront = scipy.interpolate.interp2d(dm_points_X, dm_points_Y,
                                              dm_points, kind='cubic')
    dm_wavefront = dm_wavefront(x, x)

    dm_points *= wavelength
    dm_wavefront *= wavelength

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
        _, _, norm_radius = circle_aperature(dm_wavefront, mask_r)
        max_value2 = np.max(dm_wavefront[norm_radius <= mask_r])
        min_value2 = np.min(dm_wavefront[norm_radius <= mask_r])
        PV = 'P-V: ' + str(round(pv(dm_wavefront, mask=True), 3)) + ' µm'
        RMS = 'RMS: ' + str(round(rms(dm_wavefront, mask=True), 3)) + ' µm'
    else:
        max_value1 = np.nanmax(dm_points)
        min_value1 = np.nanmin(dm_points)
        max_value2 = np.max(dm_wavefront)
        min_value2 = np.min(dm_wavefront)
        PV = 'P-V: ' + str(round(pv(dm_wavefront), 3)) + ' λ'
        RMS = 'RMS: ' + str(round(rms(dm_wavefront), 3)) + ' λ'

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
    xticklabels = ax2.get_xticks() / mm
    yticklabels = ax2.get_yticks() / mm
    ax2.set_xticklabels(xticklabels.astype(int))
    ax2.set_yticklabels(yticklabels.astype(int))
    ax2.set_xlabel('Size [%s]' % unit)
    ax2.text(0.05, 0.95, PV, fontsize=12, horizontalalignment='left',
             transform=ax2.transAxes)
    ax2.text(0.05, 0.90, RMS, fontsize=12, horizontalalignment='left',
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
