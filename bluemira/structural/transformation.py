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
FE transformation matrices and methods
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from bluemira.structural.geometry import DeformedGeometry, Geometry

from copy import deepcopy

import numpy as np
from scipy.linalg import block_diag

from bluemira.geometry.coordinates import rotation_matrix


def _direction_cosine_matrix(dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Calculates the direction cosine (transformation) matrix for an arbitrary
    vector in space. A number of painful edge cases are handled...

    Works such that the output dcm satisfies the following properties:

    local = dcm @ global
    global = dcm.T @ local

    Parameters
    ----------
    dx:
        The absolute length of the vector in the x global coordinate
    dy:
        The absolute length of the vector in the y global coordinate
    dz:
        The absolute length of the vector in the z global coordinate

    Returns
    -------
    The direction cosine matrix with reference to the global coordinate
    system (3, 3)
    """
    # Please be careful when modifying this function, and instead work on the
    # debugging version which is tested...
    length = np.sqrt(dx**2 + dy**2 + dz**2)
    a = dx / length
    b = dy / length
    c = dz / length
    d = np.hypot(a, b)
    # TODO: Why does the less intuitive, simpler algebra form work better??
    # https://ocw.mit.edu/courses/civil-and-environmental-engineering/1-571-structural-analysis-and-control-spring-2004/readings/connor_ch5.pdf
    if np.isclose(a, 0) and np.isclose(b, 0):
        dcm = np.array(
            [[0.0, 0.0, -np.sign(c)], [0.0, 1.0, 0.0], [np.sign(c), 0.0, 0.0]]
        )
    else:
        dcm = np.array([[a, -b / d, -a * c / d], [b, a / d, -b * c / d], [c, 0, d]]).T

    return dcm


def _direction_cosine_matrix_debugging(dx, dy, dz, debug=False):
    """
    Slow, ugly, safe
    """
    dcm = 0
    u = np.array([dx, dy, dz])
    x_local = u / np.linalg.norm(u)

    x_global = np.array([1.0, 0, 0])
    y_global = np.array([0, 1.0, 0])
    z_global = np.array([0, 0, 1.0])
    globa = np.array([x_global, y_global, z_global])

    if x_local[0] == 1:
        # corresponds to the global coordinate system
        dcm = np.eye(3)
        if debug:
            return dcm, globa
        return dcm
    if x_local[0] == -1:
        # corresponds to the mirrored coordinate system
        dcm = np.eye(3)
        dcm[0, 0] = -1
        local = globa.copy()
        local[0][0] = -1
        if debug:
            return dcm, local
        return dcm
    if abs(x_local[1]) == 1:
        # corresponds to a local y-vector sitting on the global x-vector
        # (rotation about z-axis)
        cos_theta = 0
        sin_theta = np.sign(x_local[1])
        dcm = np.array(
            [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
        )
        if debug:
            local = np.array([[0, sin_theta, 0], [-sin_theta, 0, 0], [0, 0, 1]])
            return dcm, local
        return dcm
    if x_local[2] == 1:
        # corresponds to a local z-vector sitting on the global x-vector
        # rotation about the y-axis
        cos_theta = 0
        sin_theta = -1
        dcm = np.array(
            [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
        )
        if debug:
            local = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            return dcm, local
        return dcm

    y_local = np.array([0, 1.0, 0])
    y_local -= np.dot(x_local, y_local) * x_local
    y_local /= np.linalg.norm(y_local)
    z_local = np.cross(x_local, y_local)

    local = np.array([x_local, y_local, z_local])

    if not isinstance(dcm, int):
        if debug:
            return dcm, local
        return dcm
    c11 = np.dot(x_global, x_local)
    c12 = np.dot(x_global, y_local)
    c13 = np.dot(x_global, z_local)
    c21 = np.dot(y_global, x_local)
    c22 = np.dot(y_global, y_local)
    c23 = np.dot(y_global, z_local)
    c31 = np.dot(z_global, x_local)
    c32 = np.dot(z_global, y_local)
    c33 = np.dot(z_global, z_local)
    dcm = np.array([[c11, c12, c13], [c21, c22, c23], [c31, c32, c33]])
    if not debug:
        return dcm
    return dcm, local


def lambda_matrix(dx: float, dy: float, dz: float) -> np.ndarray:
    """
    3-D transformation matrix, generalised to a 2-node beam.
    Effectively only handles the tiling of the direction cosine matrix
    transform.

    Parameters
    ----------
    dx:
        The absolute length of the vector in the x global coordinate
    dy:
        The absolute length of the vector in the y global coordinate
    dz:
        The absolute length of the vector in the z global coordinate

    Returns
    -------
    The transformation matrix (12, 12)
    """
    dcm = _direction_cosine_matrix(dx, dy, dz)
    return block_diag(*[dcm] * 4)


def cyclic_pattern(
    geometry: Geometry,
    axis: np.ndarray,
    angle: float,
    n: int,
    include_first: bool = True,
) -> List[Union[Geometry, DeformedGeometry]]:
    """
    Build a cyclic pattern of a Geometry.

    Parameters
    ----------
    geometry:
        The geometry to pattern
    axis:
        The axis vector about which to pattern
    angle:
        The pattern angle [degrees]
    n:
        The number of sectors to pattern
    include_first:
        Whether or not to include the first sector in the result

    Returns
    -------
    The patterned and merged geometry
    """
    # Dodge cyclic import
    from bluemira.structural.geometry import DeformedGeometry, Geometry

    if include_first:
        patterned = deepcopy(geometry)
    elif isinstance(geometry, DeformedGeometry):
        patterned = DeformedGeometry(Geometry(), geometry._scale)
    else:
        patterned = Geometry()

    for i in range(1, n):
        sector = deepcopy(geometry)
        theta = np.deg2rad(i * angle)
        t_matrix = rotation_matrix(theta, axis)
        sector.rotate(t_matrix)
        patterned.merge(sector)
        del sector  # Save some RAM

    return patterned
