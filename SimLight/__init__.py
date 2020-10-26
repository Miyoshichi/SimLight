"""
*
*  o-o          o           o     o
* |     o       |    o      |     |
*  o-o    o-O-o |      o--o O--o -o-
*     | | | | | |    | |  | |  |  |
* o--o  | o o o O---o| o--O o  o  o
*                         |
*                      o--o
*
* Copyright @ Zhou Xiang
"""

from ._version import __version__
from .field import Field, PlaneWave, SphericalWave, Gaussian
from .lens import Lens, CylindricalLens
from .propagation import propagation, near_field_propagation
from .calc import (phase, intensity, aberration, sidel_aberration,
                   delta_wavefront, deformable_mirror, zernike_coeffs)
from .aperture import circle_aperture
from .utils import longitudinal_to_wavefront, run_time_calc
from .units import *
from . import plottools, zernike

sl_version = __version__
print('SimLight ' + sl_version + '\n')
