#!/Users/ruri/anaconda2/envs/python3/bin/python
# -*- coding: utf-8 -*-

import SimLight as sl
import SimLight.plottools as slpt
import SimLight.calc as slc

import numpy as np


def main():
    input_field = sl.SphericalWave(0.6328, 1, 1000, 5)
    slpt.plot_wavefront(input_field, plot3d=True, title=input_field.field_type)


if __name__ == '__main__':
    main()
