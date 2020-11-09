# -*- coding: utf-8 -*-

"""
Created on June 22, 2020
@author: Zhou Xiang
"""

import os
import math
import numpy as np
import julia.Main as jl
import scipy.interpolate
import PIL.Image
import skimage.transform

import SimLight as sl
from .diffraction import fresnel, fresnel2, fraunhofer
from .utils import run_time_calc
from .units import *


def propagation(field, lens, z):
    """
    Calculate the light field after passing through a lens without considering
    diffraction.

    Parameters
    ----------
        field : SimLight.Filed
            The light field to be calculated.
        lens : SimLight.Lens
            The lens which a light will pass through.
        z : float
            Propagation distance after passing through.
    Returns
    ----------
        field_out : SimLight.Field
            The light field after passing through a lens.
    """
    k = 2 * np.pi / field.wavelength
    x = np.linspace(-field.size / 2, field.size / 2, field.N)
    X, Y = np.meshgrid(x, x)

    # switch - case
    def simple_lens():
        r = np.sqrt(X**2 + Y**2 + (lens.f - z)**2)
        phi = k * np.sqrt(X**2 + Y**2 + (lens.f - z)**2)
        return r, phi

    def cylindrical_lens():
        if lens.direction == 0:
            x = X
        else:
            x = Y
        r = np.sqrt(x**2 + (lens.f - z)**2)
        phi = k * np.sqrt(x**2 + (lens.f - z)**2)
        return r, phi

    options = {
        'lens': simple_lens,
        'cylindrical lens': cylindrical_lens
    }

    r, phi = options[lens.lens_type]()
    if lens.f < 0:
        phi = -phi
    field.complex_amp *= (np.exp(1j * phi) / r)
    field.complex_amp2 *= (np.exp(1j * phi) / r)

    return field


# @run_time_calc
def near_field_propagation(field, lens, z, return_3d_field=False, mag=1,
                           coord='cartesian', flag='py'):
    """
    Calculate the light field after passing through a lens.

    Parameters
    ----------
        field : SimLight.Fields
            The light field to be calculated.
        lens : SimLight.Lens
            The lens which a light will pass through.
        z : float
            Propagation distance after passing through.
        return_3d_field : boolen, optional, default False
            Whether return a 3D light field.
        mag : float, optional, default 1
            Magnification of the light field.
        coord : str, optional, {'cartesian', 'spherical'},
        default 'cartesian'
            Which coordinate system to use.
        flag : str, optional, {'py', 'jl'}, default 'py'
            Valid only when calculating 3D light field, 'py' for python and
            'jl' for Julia.
    Returns
    ----------
        field_out : SimLight.Field
            The light field after passing through a lens.
        field_3d : list
            The list of the 3D field.
    """
    # check of input parameters
    if z < 0:
        raise ValueError('The propagation distance cannot be negative.')
    if flag is 'py':
        pass
    elif flag is 'jl':
        module_dir = os.path.dirname(sl.__file__)
        os.chdir(module_dir)
        jl.include('3d_field_helper.jl')
    else:
        raise ValueError('Unspported language.')

    field = sl.Field.copy(field)
    field_3d = []

    if lens.D > field.size:
        size = lens.D if z <= 2 * lens.f else (z - lens.f) / lens.f * lens.D
        size *= mag
        # N = field.N * math.ceil(size / field.size)
        N = math.ceil(field.N * size / field.size)
        complex_amp = np.zeros([N, N], dtype=complex)
        L = int((N - field.N) / 2)
        R = L + field.N
        complex_amp[L:R, L:R] = field.complex_amp
    # elif lens.D <= field.size:
    else:
        size = field.size if z <= 2 * lens.f\
                          else (z - lens.f) / lens.f * field.size
        size *= mag
        # N = field.N * math.ceil(size / field.size)
        N = math.ceil(field.N * size / field.size)
        complex_amp = np.zeros([N, N], dtype=complex)
        lens_N = int((field.N * lens.D / field.size) / 2) * 2
        L = int((N - lens_N) / 2)
        R = L + lens_N
        L_in = int((field.N - lens_N) / 2)
        R_in = L_in + lens_N
        complex_amp[L:R, L:R] = field.complex_amp[L_in:R_in, L_in:R_in]

    field.complex_amp = complex_amp
    field.complex_amp2 = complex_amp
    field.size = size
    field.N = N

    x = np.linspace(-size / 2, size / 2, N)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    k = 2 * np.pi / field.wavelength

    # effect of lens type
    def simple_lens():
        f = lens.f if coord == 'cartesian' else 10 * m
        phi = -k * (X**2 + Y**2) / (2 * f)
        return phi

    def cylindrical_lens():
        f = lens.f if coord == 'cartesian' else 10 * m
        if lens.direction == 1:
            x = Y
        else:
            x = X
        phi = -k * (x**2) / (2 * f)
        return phi

    lens_types = {
        'lens': simple_lens,
        'cylindrical lens': cylindrical_lens
    }
    phi = lens_types[lens.lens_type]()
    if lens.f < 0:
        phi = -phi

    # complex amplitude after passing through lens
    field.complex_amp *= np.exp(1j * phi)
    field.complex_amp[R >= field.size / 2] = 0
    # leads to intensity calculation error
    # field.complex_amp2 *= np.exp(1j * phi)
    # field.complex_amp2[R >= field.size / 2] = 0
    # TODO temp method
    field.complex_amp2 = field.complex_amp

    # complex amplitude passing the distance z
    # cartesian coordinate method
    def cartesian_coordinate(z_):
        if z_ != 0:
            return fresnel(field, z_)
        else:
            raise ValueError('Propagation distance error.')

    # spherical coordinate method
    def spherical_coordinate(z_):
        large_number = 1e7 * m
        tiny_number = 100 * nm
        f = lens.f
        # size = field.size
        wavelength = field.wavelength
        curvature = field.curvature

        if f == z_:
            # f += tiny_number
            f_ = 10 * m
            f = f_ * lens.f / (f_ - lens.f)
        if abs(z_ - f) < tiny_number:
            z_ = f - tiny_number if z_ < f else f + tiny_number
        if curvature != 0:
            f1 = 1 / curvature
        else:
            f1 = large_number * size**2 / wavelength
        if f + f1 != 0:
            f = (f * f1) / (f + f1)
        else:
            f = large_number * size**2 / wavelength

        z1 = -z_ * f / (z_ - f)
        # cannot propagate to reversely
        # if z1 < 0:
        #     raise ValueError(('Spherical coordinate error: '
        #                       'negative distance of %f') % z_)

        new_field = fresnel(field, z1)
        amp_scale = (f - z_) / f
        curvature = -1 / (z_ - f)
        new_field.size *= amp_scale
        new_field.complex_amp /= amp_scale
        new_field.complex_amp2 /= amp_scale
        new_field.curvature = curvature

        if curvature != 0:
            f_ = -1 / curvature
            h, w = new_field.N, new_field.N
            cy, cx = int(h / 2), int(w / 2)
            Y, X = np.mgrid[:h, :w]
            dx = new_field.size / new_field.N
            Y = (Y - cy) * dx
            X = (X - cx) * dx
            R = X**2 + Y**2
            phi = R * k / (2 * f_)
            new_field.complex_amp *= np.exp(1j * phi)
            new_field.complex_amp2 = new_field.complex_amp
            new_field.curvature = 0

        return new_field

    coords = {
        'cartesian': cartesian_coordinate,
        'spherical': spherical_coordinate
    }
    out_field = coords[coord](z)

    if return_3d_field:
        # defines 3d field range
        # TODO auto foucs calculation
        # dx = field.size / field.N
        dx = 0.1 * µm
        delta_z = 25 * µm
        fix = 1 * µm
        delta_N = int(delta_z / dx)
        # z_range = np.linspace(z - delta_z, z, delta_N)
        z_range = np.linspace(z - delta_z - fix, z - fix, delta_N)
        # z_range = np.linspace(z - fix, z - delta_z - fix, delta_N)

        # calculates light fields
        print('====== Calculating light fields ======')
        for index, z_ in enumerate(z_range):
            print('\r%.2f%% (%d / %d)' % ((index + 1) / len(z_range) * 100,
                                          index + 1,
                                          len(z_range)), end='')
            field_3d.append(coords[coord](z_))
        field_3d_size = []
        for field_ in field_3d:
            field_3d_size.append(field_.size)
            # print(field_.size / µm)
        max_size = np.max(field_3d_size)
        # min_size = np.min(field_3d_size)
        # print(max_size / µm, min_size / µm)
        # pads light fields to same size
        print('\n====== Padding to same size ======')
        for index, field_ in enumerate(field_3d):
            frac = abs(max_size / field_.size)
            lower = int(math.ceil(field_.complex_amp.shape[0] *
                                  (frac - 1) / 2))
            upper = int(math.ceil(field_.complex_amp.shape[0] *
                                  (frac + 1) / 2))
            # prints padding progress
            print(('\r%.2f%% (%d / %d) [%d => %d]' %
                   ((index + 1) / len(field_3d) * 100,
                    index + 1,
                    len(field_3d),
                    field_.N,
                    lower + upper)),
                  end='')
            if flag is 'py':
                new_complex_amp = np.zeros([lower + upper, lower + upper],
                                           dtype=np.complex)
                # if lower + upper >= field_.N:
                #     new_complex_amp[lower:upper,
                #                     lower:upper] = field_.complex_amp
                # else:
                #     new_complex_amp = field.new_complex_amp[lower:upper,
                #                                             lower:upper]
                new_complex_amp[lower:upper,
                                lower:upper] = field_.complex_amp
            else:
                new_complex_amp = jl.pad_to_same_size(field_.complex_amp,
                                                      lower,
                                                      upper)
            field_.complex_amp = np.asarray(new_complex_amp)
            field_.complex_amp2 = field_.complex_amp
            field_.size = max_size
            field_.N = lower + upper

        # resizes light fields to same pixels
        print('\n====== Resizing to same pixels ======')
        field_3d_N = []
        for field_ in field_3d:
            field_3d_N.append(field_.N)
        median_N = int(np.median(field_3d_N))
        for index, field_ in enumerate(field_3d):
            # prints resizing progress
            print(('\r%.2f%% (%d / %d) [%d => %d]' %
                   ((index + 1) / len(field_3d) * 100,
                    index + 1,
                    len(field_3d),
                    field_.N,
                    median_N)),
                  end='')
            # gets real and imagne part of complex amplitudes
            complex_amp_real = np.real(field_.complex_amp)
            complex_amp_imag = np.imag(field_.complex_amp)
            # TODO more effective resize alogrithm
            if flag is 'py':
                # # interpolating method
                # # before interpolating
                # x = np.linspace(-field_.size / 2,
                #                 field_.size / 2, field_.N)
                # y = np.linspace(-field_.size / 2, field_.size / 2,
                #                 field_.N)
                # # after interpolating
                # x_ = np.linspace(-field_.size / 2,
                #                  field_.size / 2,
                #                  median_N)
                # y_ = np.linspace(-field_.size / 2,
                #                  field_.size / 2,
                #                  median_N)
                # # interpolates real part and imagine part respectively
                # resized_real = scipy.interpolate.interp2d(x,
                #                                           y,
                #                                          complex_amp_real,
                #                                           kind='cubic')
                # resized_imag = scipy.interpolate.interp2d(x,
                #                                           y,
                #                                          complex_amp_imag,
                #                                           kind='cubic')
                # field_.complex_amp = (resized_real(x_, y_) +
                #                       resized_imag(x_, y_) * 1j)
                # image resize method
                # # uses scikit-image (high quality, slow speed)
                # resized_real = skimage.transform.resize(complex_amp_real,
                #                                         (median_N,
                #                                          median_N),
                #                                         order=1)
                # resized_imag = skimage.transform.resize(complex_amp_imag,
                #                                         (median_N,
                #                                          median_N),
                #                                         order=1)
                # uses pillow (low quality, fast speed)
                resized_real = np.array(
                    PIL.Image.fromarray(complex_amp_real).resize(
                        size=(median_N, median_N)))
                resized_imag = np.array(
                    PIL.Image.fromarray(complex_amp_imag).resize(
                        size=(median_N, median_N)))
            # uses julia
            else:
                resized_real = np.asarray(
                    jl.resize_to_same_size(complex_amp_real, median_N))
                resized_imag = np.asarray(
                    jl.resize_to_same_size(complex_amp_imag, median_N))
            field_.complex_amp = (resized_real + resized_imag * 1j)
            field_.complex_amp2 = field_.complex_amp
            field_.N = median_N

        return out_field, field_3d
    else:
        return out_field
