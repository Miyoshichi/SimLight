#!/Users/ruri/anaconda2/envs/python3/bin/python
# -*- coding: utf-8 -*-

import SimLight as sl
import SimLight.plottools as slpt

import numpy as np


def main():
    input_field = sl.SphericalWave(0.6328, 1, 1000, 5)
    slpt.plot_wavefront(input_field, mask_r=0.8, plot3d=True)


if __name__ == '__main__':
    main()
