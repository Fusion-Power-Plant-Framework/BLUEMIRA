# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
Constants for use in the structural module.
"""
import numpy as np
from matplotlib.pyplot import get_cmap

from bluemira.base.constants import EPS

# Poisson's ratio
NU = 0.33

# Shear deformation limit (r_g/L)
#   Above which shear deformation properties must be used
SD_LIMIT = 0.1

# The proximity tolerance
#   Used for checking node existence
D_TOLERANCE = 1e-5

# Small number tolerance
#   Used for checking if cos / sin of angles is actually zero
#   Chosen to be slightly larger than:
#      * np.cos(3 * np.pi / 2) =  -1.8369701987210297e-16
#      * np.sin(np.pi * 2) = -2.4492935982947064e-16
NEAR_ZERO = 2 * EPS

# The large displacement ratio (denominator)
R_LARGE_DISP = 100

# Global coordinate system

GLOBAL_COORDS = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

# Mapping of indices to load types
LOAD_MAPPING = {"Fx": 0, "Fy": 1, "Fz": 2, "Mx": 3, "My": 4, "Mz": 5}

# A list of all potential load types
LOAD_TYPES = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

# Mapping of load string to direction vectors (for plotting purposes)
LOAD_STR_VECTORS = {
    "Fx": GLOBAL_COORDS[0],
    "Fy": GLOBAL_COORDS[1],
    "Fz": GLOBAL_COORDS[2],
    "Mx": GLOBAL_COORDS[0],
    "My": GLOBAL_COORDS[1],
    "Mz": GLOBAL_COORDS[2],
}

# Mapping of load integer to direction vectors (for plotting purposes)
LOAD_INT_VECTORS = dict(enumerate(LOAD_STR_VECTORS.values()))

# Mapping of indices to displacement/support types
DISP_MAPPING = {"Dx": 0, "Dy": 1, "Dz": 2, "Rx": 3, "Ry": 4, "Rz": 5}

# Number of interpolation points per Element
N_INTERP = 7

# Color map for Element stresses
STRESS_COLOR = get_cmap("seismic", 1000)

# Color map for Element deflections
DEFLECT_COLOR = get_cmap("viridis", 1000)
