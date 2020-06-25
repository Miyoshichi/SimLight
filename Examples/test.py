#!/Users/ruri/anaconda2/envs/python3/bin/python
# -*- coding: utf-8 -*-

import SimLight as sl
import SimLight.plottools as slpt
import SimLight.calc as slc

import numpy as np


def main():
    wavelength = 0.633
    size = 25.4
    res = 200
    f = 50
    A = 1

    F = sl.PlaneWave(wavelength, size, res)
    Z = sl.zernike.ZernikeCofficients(15)
    # Z.cofficients[3] = 0.1 * A
    Z.cofficients[8] = 0.02 * A
    # Z.cofficients[8] = 0.01 * A
    L = sl.Lens.new_lens(size, f)

    F, phi = sl.aberration(F, Z)
    F.plot_wavefront(dimension=3, mask_r=1)

    F = sl.near_field_propagation(F, L, f)


if __name__ == '__main__':
    main()
