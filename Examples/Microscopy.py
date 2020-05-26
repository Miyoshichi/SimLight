#!/Users/ruri/anaconda2/envs/python3/bin/python
# -*- coding: utf-8 -*-

import SimLight as sl
import SimLight.plottools as slpt

import numpy as np


def main():
    input_field = sl.PlaneWave(0.6328, 1, 100, 1, 1)
    slpt.plot_wavefront(input_field, None, False, 'Example')
    print(input_field.complex_amp)


if __name__ == '__main__':
    main()
