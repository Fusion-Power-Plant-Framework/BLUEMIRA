# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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
Utility for sets of coordinates
"""

from typing import Iterable

import numba as nb
import numpy as np
from pyquaternion import Quaternion

from bluemira.base.constants import EPS
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.display.plotter import PointsPlotter, register_plotter
from bluemira.geometry.error import CoordinatesError

# =============================================================================
# Pre-processing utilities
# =============================================================================


def xyz_process(func):
    """
    Decorator for parsing x, y, z coordinates to numpy float arrays and dimension
    checking.
    """

    def wrapper(x, y, z=None):
        _validate_coordinates(x, y, z)
        x = np.ascontiguousarray(x, dtype=np.float_)
        y = np.ascontiguousarray(y, dtype=np.float_)
        if z is not None:
            z = np.ascontiguousarray(z, dtype=np.float_)

        return func(x, y, z)

    return wrapper


def _validate_coordinates(x, y, z=None):
    if z is None:
        if not len(x) == len(y):
            raise CoordinatesError(
                "All coordinates must have the same length but "
                f"got len(x) = {len(x)}, len(y) = {len(y)}"
            )
    else:
        if not len(x) == len(y) == len(z):
            raise CoordinatesError(
                "All coordinates must have the same length but "
                f"got len(x) = {len(x)}, len(y) = {len(y)}, len(z) = {len(z)}"
            )


# =============================================================================
# Tools and calculations for sets of coordinates
# =============================================================================


@nb.jit(cache=True, nopython=True)
def get_normal_vector(x, y, z):
    """
    Calculate the normal vector from a series of planar points.

    Parameters
    ----------
    x: np.array
        The x coordinates
    y: np.array
        The y coordinates
    z: np.array
        The z coordinates

    Returns
    -------
    n_hat: np.array(3)
        The normalised normal vector
    """
    if len(x) < 3:
        raise CoordinatesError(
            "Cannot get a normal vector for a set of points with length less than 3."
        )

    if not (len(x) == len(y) == len(z)):
        raise CoordinatesError("Point coordinate vectors must be of equal length.")

    n_hat = np.array([0.0, 0.0, 0.0])  # Force numba to type to floats
    p1 = np.array([x[0], y[0], z[0]])
    p2 = np.array([x[1], y[1], z[1]])
    v1 = p2 - p1

    # Force length 3 vectors to access index 2 without raising IndexErrors elsewhere
    i_max = max(3, len(x) - 1)
    for i in range(2, i_max):
        p3 = np.array([x[i], y[i], z[i]])
        v2 = p3 - p2

        if np.all(np.abs(v2) < EPS):  # np.allclose not available in numba
            v2 = p3 - p1
            if np.all(np.abs(v2) < EPS):
                continue

        n_hat[:] = np.cross(v1, v2)

        if not np.all(np.abs(n_hat) < EPS):
            break
    else:
        raise CoordinatesError("Unable to find a normal vector from set of points.")

    return n_hat / np.linalg.norm(n_hat)


@xyz_process
def get_perimeter(x, y, z=None):
    """
    Calculate the perimeter of a set of coordinates.

    Parameters
    ----------
    x: np.array
        The x coordinates
    y: np.array
        The y coordinates
    z: Union[None, np.array]
        The z coordinates

    Returns
    -------
    perimeter: float
        The perimeter of the coordinates
    """
    if z is None:
        return get_perimeter_2d(x, y)
    else:
        return get_perimeter_3d(x, y, z)


@nb.jit(cache=True, nopython=True)
def get_perimeter_2d(x, y):
    """
    Calculate the perimeter of a 2-D set of coordinates.

    Parameters
    ----------
    x: np.array
        The x coordinates
    y: np.array
        The y coordinates

    Returns
    -------
    perimeter: float
        The perimeter of the coordinates
    """
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    return np.sum(np.sqrt(dx**2 + dy**2))


@nb.jit(cache=True, nopython=True)
def get_perimeter_3d(x, y, z):
    """
    Calculate the perimeter of a set of 3-D coordinates.

    Parameters
    ----------
    x: np.array
        The x coordinates
    y: np.array
        The y coordinates
    z: np.array
        The z coordinates

    Returns
    -------
    perimeter: float
        The perimeter of the coordinates
    """
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    dz = z[1:] - z[:-1]
    return np.sum(np.sqrt(dx**2 + dy**2 + dz**2))


@xyz_process
def get_area(x, y, z=None):
    """
    Calculate the area inside a closed polygon with x, y coordinate vectors.
    `Link Shoelace method <https://en.wikipedia.org/wiki/Shoelace_formula>`_

    Parameters
    ----------
    x: np.array
        The first set of coordinates [m]
    y: np.array
        The second set of coordinates [m]
    z: Union[np.array, None]
        The third set of coordinates or None (for a 2-D polygon)

    Returns
    -------
    area: float
        The area of the polygon [m^2]
    """
    if z is None:
        return get_area_2d(x, y)
    else:
        return get_area_3d(x, y, z)


@nb.jit(cache=True, nopython=True)
def get_area_2d(x, y):
    """
    Calculate the area inside a closed polygon with x, y coordinate vectors.
    `Link Shoelace method <https://en.wikipedia.org/wiki/Shoelace_formula>`_

    Parameters
    ----------
    x: np.array
        The first set of coordinates [m]
    y: np.array
        The second set of coordinates [m]

    Returns
    -------
    area: float
        The area of the polygon [m^2]
    """
    # No np.roll in numba
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    x1 = np.append(x[-1], x[:-1])
    y1 = np.append(y[-1], y[:-1])
    return 0.5 * np.abs(np.dot(x, y1) - np.dot(y, x1))


@nb.jit(cache=True, nopython=True)
def get_area_3d(x, y, z):
    """
    Calculate the area inside a closed polygon.
    `Link Shoelace method <https://en.wikipedia.org/wiki/Shoelace_formula>`_

    Parameters
    ----------
    x: np.array
        The first set of coordinates [m]
    y: np.array
        The second set of coordinates [m]
    z: np.array
        The third set of coordinates [m]

    Returns
    -------
    area: float
        The area of the polygon [m^2]
    """
    if np.all(x == x[0]) and np.all(y == y[0]) and np.all(z == z[0]):
        # Basically a point, but avoid getting the normal vector..
        return 0

    v3 = get_normal_vector(x, y, z)
    m = np.zeros((3, len(x)))
    m[0, :] = x
    m[1, :] = y
    m[2, :] = z
    a = np.array([0.0, 0.0, 0.0])
    for i in range(len(z)):
        a += np.cross(m[:, i], m[:, (i + 1) % len(z)])
    a *= 0.5
    return abs(np.dot(a, v3))


@nb.jit(cache=True, nopython=True)
def check_ccw(x, z):
    """
    Check that a set of x, z coordinates are counter-clockwise.

    Parameters
    ----------
    x: np.array
        The x coordinates of the polygon
    z: np.array
        The z coordinates of the polygon

    Returns
    -------
    ccw: bool
        True if polygon counterclockwise
    """
    a = 0
    for n in range(len(x) - 1):
        a += (x[n + 1] - x[n]) * (z[n + 1] + z[n])
    return a < 0


def check_ccw_3d(x, y, z, normal):
    """
    Check if a set of coordinates is counter-clockwise w.r.t a normal vector.

    Parameters
    ----------
    x: np.array
        The first set of coordinates [m]
    y: np.array
        The second set of coordinates [m]
    z: np.array
        The third set of coordinates [m]
    normal: np.array
        The normal vector about which to check for CCW

    Returns
    -------
    ccw: bool
        Whether or not the set is CCW about the normal vector
    """
    # Translate to centroid
    dx, dy, dz = get_centroid_3d(x, y, z)
    x, y, z = x - dx, y - dy, z - dz
    # Rotate to x-y plane
    r = rotation_matrix_v1v2([0, 0, 1], normal)
    x, y, z = r.T @ np.array([x, y, z])
    # Check projected x-y is CCW
    return check_ccw(x, y)


@xyz_process
def get_centroid(x, y, z=None):
    """
    Calculate the centroid of a non-self-intersecting 2-D counter-clockwise polygon.

    Parameters
    ----------
    x: np.array
        x coordinates of the loop to calculate on
    y: np.array
        y coordinates of the loop to calculate on
    z: Union[None, np.array]

    Returns
    -------
    centroid: np.array
        The x, y, [z] coordinates of the centroid [m]
    """
    if z is None:
        return get_centroid_2d(x, y)
    else:
        return get_centroid_3d(x, y, z)


@nb.jit(cache=True, nopython=True)
def get_centroid_2d(x, z):
    """
    Calculate the centroid of a non-self-intersecting 2-D counter-clockwise polygon.

    Parameters
    ----------
    x: np.array
        x coordinates of the loop to calculate on
    z: np.array
        z coordinates of the loop to calculate on

    Returns
    -------
    centroid: List[float]
        The x, z coordinates of the centroid [m]
    """
    if not check_ccw(x, z):
        x = np.ascontiguousarray(x[::-1])
        z = np.ascontiguousarray(z[::-1])
    area = get_area_2d(x, z)

    cx, cz = 0, 0
    for i in range(len(x) - 1):
        a = x[i] * z[i + 1] - x[i + 1] * z[i]
        cx += (x[i] + x[i + 1]) * a
        cz += (z[i] + z[i + 1]) * a

    if area != 0:
        # Zero division protection
        cx /= 6 * area
        cz /= 6 * area

    return [cx, cz]


def get_centroid_3d(x, y, z):
    """
    Calculate the centroid of a non-self-intersecting counterclockwise polygon
    in 3-D.

    Parameters
    ----------
    x: Iterable
        The x coordinates
    y: Iterable
        The y coordinates
    z: Iterable
        The z coordinates

    Returns
    -------
    centroid: List[float]
        The x, y, z coordinates of the centroid [m]
    """
    cx, cy = get_centroid_2d(x, y)
    cx2, cz = get_centroid_2d(x, z)
    cy2, cz2 = get_centroid_2d(y, z)

    # The following is an "elegant" but computationally more expensive way of
    # dealing with the 0-area edge cases
    # (of which there are more than you think)
    cx = np.array([cx, cx2])
    cy = np.array([cy, cy2])
    cz = np.array([cz, cz2])

    def get_rational(i, array):
        """
        Gets rid of infinity and nan coordinates
        """
        args = np.argwhere(np.isfinite(array))
        if len(args) == 0:
            # 2-D shape with a simple axis offset
            # Get the first value of the coordinate set which is equal to the
            # offset
            return [x, y, z][i][0]
        elif len(args) == 1:
            return array[args[0][0]]
        else:
            if all(np.isclose(array, 0)):
                return 0
            elif any(np.isclose(array, 0)):
                # Occasionally the two c values are not the same, and one is 0
                return array[np.argmax(np.abs(array))]
            else:
                return array[0]

    return [get_rational(i, c) for i, c in enumerate([cx, cy, cz])]


def rotation_matrix_v1v2(v1, v2):
    """
    Get a rotation matrix based off two vectors.
    """
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    cos_angle = np.dot(v1, v2)
    d = np.cross(v1, v2)
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        matrix = np.identity(3) if cos_angle > 0.0 else -np.identity(3)
    else:
        d /= sin_angle

        eye = np.eye(3)
        ddt = np.outer(d, d)
        skew = np.array(
            [[0, d[2], -d[1]], [-d[2], 0, d[0]], [d[1], -d[0], 0]], dtype=np.float64
        )

        matrix = ddt + cos_angle * (eye - ddt) + sin_angle * skew

    return matrix


def principal_components(xyz_array):
    """
    Principal component analysis.
    """
    mean = np.mean(xyz_array, axis=1)
    xyz_shift = xyz_array - mean.reshape((3, 1))

    cov = np.cov(xyz_shift)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors


def _parse_to_xyz_array(xyz_array):
    """
    Make a 3, N xyz array out of just about anything.
    """
    if isinstance(xyz_array, np.ndarray):
        xyz_array = _parse_array(xyz_array)
    elif isinstance(xyz_array, dict):
        xyz_array = _parse_dict(xyz_array)
    elif isinstance(xyz_array, Iterable):
        # We temporarily set the dtype to object to avoid a VisibleDeprecationWarning
        xyz_array = np.array(xyz_array, dtype=object)
        xyz_array = _parse_array(xyz_array)
    else:
        raise CoordinatesError(f"Cannot instantiate Coordinates with: {type(xyz_array)}")
    return xyz_array


def _parse_array(xyz_array):
    try:
        xyz_array = np.array(np.atleast_2d(xyz_array), dtype=np.float64)
    except ValueError:
        raise CoordinatesError(
            "Cannot instantiate Coordinates with a ragged (3, N | M) array."
        )

    shape = xyz_array.shape
    if len(shape) > 2:
        raise NotImplementedError

    n, m = shape
    if n == 3:
        if m == 3:
            bluemira_warn(
                "You are creating Coordinates with a (3, 3) array, defaulting to (3, N)."
            )

    elif m == 3:
        xyz_array = xyz_array.T

    else:
        raise CoordinatesError(
            "Cannot instantiate Coordinates where either n or m != 3."
        )

    if not np.allclose([len(xyz_array[i]) for i in range(3)], len(xyz_array[0])):
        raise CoordinatesError(
            "Cannot instantiate Coordinates with a ragged (3, N | M) array."
        )

    return xyz_array


def _parse_dict(xyz_dict):
    x = np.atleast_1d(xyz_dict.get("x", 0))
    y = np.atleast_1d(xyz_dict.get("y", 0))
    z = np.atleast_1d(xyz_dict.get("z", 0))

    shape_lengths = np.array([len(c.shape) for c in [x, y, z]])

    if np.any(shape_lengths > 1):
        raise CoordinatesError(
            "Cannot instantiate Coordinates from dict with coordinate vectors that are not 1-D."
        )

    lengths = [len(c) for c in [x, y, z]]

    usable_lengths = []
    for length in lengths:
        if length != 1:
            usable_lengths.append(length)

    if not np.allclose(usable_lengths, usable_lengths[0]):
        raise CoordinatesError(
            "Cannot instantiate Coordinate from dict with a ragged set of vectors."
        )

    # Backfill single-value coordinates
    actual_length = usable_lengths[0]
    if len(x) == 1:
        x = x[0] * np.ones(actual_length)
    if len(y) == 1:
        y = y[0] * np.ones(actual_length)
    if len(z) == 1:
        z = z[0] * np.ones(actual_length)

    return np.array([x, y, z])


class Coordinates:
    """
    Coordinates object for storing ordered sets of coordinates.

    An array shape of (3, N) is enforced.

    Counter-clockwise direction can be set relative to a normal vector.

    Parameters
    ----------
    xyz_array: Union[np.ndarray, dict, Iterable[Iterable]]

    Notes
    -----
    This is a utility class for dealing with sets of coordinates in a number of different
    contexts. It should not be used for the creation of CAD geometries.
    """

    __slots__ = ("_array", "_is_planar", "_normal_vector")
    # =============================================================================
    # Instantiation
    # =============================================================================

    def __init__(self, xyz_array):
        self._array = _parse_to_xyz_array(xyz_array)
        self._is_planar = None
        self._normal_vector = None

    # =============================================================================
    # Checks
    # =============================================================================

    def _set_plane_props(self):
        """
        Set the planar properties of the Coordinates.
        """
        if self._is_planar is None and self._normal_vector is None:
            self._update_plane_props()

    def _update_plane_props(self):
        if len(self) > 3:
            eigenvalues, eigenvectors = principal_components(self._array)

            if np.isclose(eigenvalues[-1], 0.0):
                self._is_planar = True
            else:
                self._is_planar = False

            self._normal_vector = eigenvectors[:, -1]
        else:
            bluemira_warn("Cannot set planar properties on Coordinates with length < 3.")
            self._is_planar = False
            self._normal_vector = None

    @property
    def is_planar(self):
        """
        Whether or not the Coordinates are planar.
        """
        self._set_plane_props()
        return self._is_planar

    @property
    def normal_vector(self):
        """
        The normal vector of the best-fit plane of the Coordinates.
        """
        self._set_plane_props()
        return self._normal_vector

    def check_ccw(self, axis=None) -> bool:
        """
        Whether or not the Coordinates are ordered in the counter-clockwise direction
        about a specified axis. If None is specified, the Coordinates normal vector will
        be used.
        """
        if len(self) < 3:
            return False

        if axis is None:
            axis = self.normal_vector
        else:
            axis = np.array(axis, dtype=float)
            if not axis.size == 3:
                raise CoordinatesError("Base vector must be of size 3.")
            axis /= np.linalg.norm(axis)

        return check_ccw_3d(self.x, self.y, self.z, axis)

    def set_ccw(self, axis=None):
        """
        Set the Coordinates to be counter-clockwise about a specified axis. If None is
        specified, the Coordinates normal vector will be used.
        """
        if len(self) < 3:
            bluemira_warn("Cannot set Coordinates of length < 3 to CCW.")
            return

        if not self.check_ccw(axis=axis):
            self.reverse()

    # =============================================================================
    # Property access
    # =============================================================================

    @property
    def x(self):
        """
        The x coordinate vector
        """
        return self._array[0]

    @property
    def y(self):
        """
        The y coordinate vector
        """
        return self._array[1]

    @property
    def z(self):
        """
        The z coordinate vector
        """
        return self._array[2]

    @property
    def xy(self):
        """
        The x-y coordinate array
        """
        return self._array[[0, 1], :]

    @property
    def xz(self):
        """
        The x-z coordinate array
        """
        return self._array[[0, 2], :]

    @property
    def yz(self):
        """
        The y-z coordinate array
        """
        return self._array[[1, 2], :]

    @property
    def points(self):
        """
        A list of the individual points of the Coordinates.
        """
        return list(self.T)

    # =========================================================================
    # Conversions
    # =========================================================================

    def as_dict(self):
        """
        Cast the Coordinates as a dictionary.

        Returns
        -------
        d: dict
            Dictionary with {'x': [], 'y': [], 'z':[]}
        """
        return {"x": self.x, "y": self.y, "z": self.z}

    # =============================================================================
    # Useful properties
    # =============================================================================

    @property
    def closed(self) -> bool:
        """
        Whether or not this is a closed set of Coordinates.
        """
        if len(self) > 2:
            if np.allclose(self[:, 0], self[:, -1], rtol=0, atol=EPS):
                return True
        return False

    @property
    def length(self) -> float:
        """
        Perimeter length of the coordinates.
        """
        return get_perimeter_3d(*self._array)

    @property
    def center_of_mass(self) -> tuple:
        """
        Geometrical centroid of the Coordinates.
        """
        # [sic] coordinates do not have a "mass", but named such for consistency with
        # other geometry objects.
        return tuple(get_centroid_3d(*self._array))

    # =============================================================================
    # Array-like behaviour
    # =============================================================================

    @property
    def T(self):  # noqa :N802
        """
        Transpose of the Coordinates
        """
        return self._array.T

    @property
    def shape(self):
        """
        Shape of the Coordinates
        """
        return self._array.shape

    # =============================================================================
    # Modification
    # =============================================================================
    def reverse(self):
        """
        Reverse the direction of the Coordinates.
        """
        self._array = self._array[:, ::-1]

    def open(self):
        """
        Open the Coordinates (if they are closed)
        """
        if self.closed:
            self._array = self._array[:, :-1]

    def close(self):
        """
        Close the Coordinates (if they are open)
        """
        if not self.closed:
            self._array = np.vstack((self._array.T, self._array[:, 0])).T

    def rotate(
        self, base: tuple = (0, 0, 0), direction: tuple = (0, 0, 1), degree: float = 0.0
    ):
        """
        Rotate the Coordinates.

        Parameters
        ----------
        base: tuple (x,y,z)
            Origin location of the rotation
        direction: tuple (x,y,z)
            The direction vector
        degree: float
            rotation angle [degrees]
        """
        if degree == 0.0:
            return

        base = np.array(base, dtype=float)
        if not base.size == 3:
            raise CoordinatesError("Base vector must be of size 3.")

        direction = np.array(direction, dtype=float)
        if not direction.size == 3:
            raise CoordinatesError("Direction vector must be of size 3.")
        direction /= np.linalg.norm(direction)

        points = self._array - base.reshape(3, 1)
        quart = Quaternion(axis=direction, angle=np.deg2rad(degree))
        r_matrix = quart.rotation_matrix
        new_array = points.T @ r_matrix.T
        self._array = new_array.T

        self._update_plane_props()

    def translate(self, vector: tuple = (0, 0, 0)):
        """
        Translate this shape with the vector. This function modifies the self
        object.
        """
        vector = np.array(vector)
        if not vector.size == 3:
            raise CoordinatesError("Translation vector must be of size 3.")

        self._array += vector.reshape(3, 1)
        self._update_plane_props()

    # =============================================================================
    # Dunders (with different behaviour to array)
    # =============================================================================

    def __eq__(self, other):
        """
        Check the Coordinates for equality with other Coordinates.

        Parameters
        ----------
        other: Coordinates
            The other Coordinates to compare against

        Returns
        -------
        equal: bool
            Whether or not the Coordinates are identical

        Notes
        -----
        Coordinates with identical coordinates but different orderings will not be
        counted as identical.
        """
        if isinstance(other, self.__class__):
            return np.all(np.allclose(self._array, other._array, rtol=0, atol=EPS))
        return False

    def __len__(self):
        """
        The number of points in the Coordinates.
        """
        return self.shape[1]

    # =============================================================================
    # Array-like dunders
    # =============================================================================

    def __repr__(self):
        """
        Representation of the Coordinates.
        """
        r = self._array.__repr__()
        return f"{self.__class__.__name__}{r[5:]}"

    def __getitem__(self, *args, **kwargs):
        """
        Array-like indexing and slicing.
        """
        return self._array.__getitem__(*args, **kwargs)

    def __iter__(self):
        """
        Array-like unpacking.
        """
        return self._array.__iter__()


register_plotter(Coordinates, PointsPlotter)
