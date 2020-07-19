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
from .calc import phase, intensity, psf, delta_wavefront
from .unwrap import simple_unwrap_1d


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


def plot_intensity(field, mask_r=None, norm_type=0, dimension=2, mag=1,
                   title=''):
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
        mag: float
            Magnification of the figure. (optional)
        title: str
            Title of the figure. (optional)
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
    if mag < 1:
        lower = int(intensity_.shape[0] * (1 - mag) / 2)
        upper = int(intensity_.shape[0] * (1 - (1 - mag) / 2))
        intensity_ = intensity_[lower:upper, lower:upper]
        size *= mag
    elif mag > 1:
        lower = int(intensity_.shape[0] * (mag - 1) / 2)
        upper = int(intensity_.shape[0] * (mag + 1) / 2)
        new_intensity = np.zeros(lower + upper, lower + upper)
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
            ax.set_xlabel('Size [mm]')
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
        ax.grid(True)
        ax.set_xlabel('Size [mm]')
        ax.set_ylabel('Intensity [a.u.]')

    if title:
        ax.set_title(title)

    plt.show()


def plot_two_intensities_diff(field1, field2,
                              label1='Reference', label2='Reality',
                              norm_type=0, mag=1, title=''):
    """Deprecated
    Plot the intensity difference of the two light field.

    Args:
        field1: tuple
            Reference light field to compare.
        field2: tuple
            Another light field to compare.
        label1: str
            Label of field1.
        label2: str
            Label of field2.
        norm_type: int
            Type of normalization. (optional, default is 0)
            0: no normalization.
            1: normalize to 0~1.
            2: normalize to 0~255.
        mag: float
            Magnification of the figure. (optional)
        title: str
            Title of the figure. (optional).
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


def plot_multi_intensities_diff(field_ref, *fields, shift=None, labels=None,
                                norm_type=0, title=''):
    """
    """
    # check of input parameters
    if norm_type:
        if norm_type < 0 or norm_type > 2 or type(norm_type) is not int:
            raise ValueError('Invalid type of normalization.')
    if isinstance(field_ref, sl.Field) is True:
        size = field_ref.size
        intensities = []
        intensities.append(intensity(field_ref, norm_type=0))
        for field in fields:
            if isinstance(field, sl.Field) is True:
                if field.size != size:
                    raise ValueError('Cannot campare the two light fields'
                                     'with different sizes.')
                else:
                    # size = field_ref.size
                    intensities.append(intensity(field, norm_type=0))
            else:
                raise ValueError('Invalid light field.')
    else:
        raise ValueError('Invalid light field.')

    if norm_type > 0:
        max_value = np.max(intensities[0])
        intensities /= max_value
        if norm_type > 1:
            intensities *= 255

    fig = plt.figure()
    ax = fig.add_subplot(111)

    center = int(intensities[0].shape[0] / 2)
    X = np.linspace(-size / 2, size / 2, intensities[0].shape[0])
    shift_ = np.zeros(len(fields) + 1, dtype=int)
    if shift:
        shift[:len(shift)] = shift
    for index, intensity_ in enumerate(intensities):
        ax.plot(X, intensity_[:, center + shift_[index]])
    ax.grid(True)
    ax.set_xlabel('Size [mm]')
    ax.set_ylabel('Intensity [a.u.]')

    if labels:
        ax.legend(labels)
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


def plot_longitude(lens, wavelength=0.550, title=''):
    """
    Show the graph of the longitudinal aberration acrroding to the
    Sidel coefficients.
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

    x = np.linspace(-size / 2, size / 2, N)
    height = x[center:]
    delta_W_line = delta_W[center, center:]
    # longitude = delta_W_line * -(f / (size / 2))**2
    longitude = delta_W_line * -f**2 / size

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
