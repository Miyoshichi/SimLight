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

    F = sl.SphericalWave(wavelength, size, res, 10)
    L = sl.Lens.new_lens(size, f)

    F = sl.near_field_propagation(F, L, 10)
    F.plot_wavefront(dimension=3, title=F.field_type)


if __name__ == '__main__':
    main()
