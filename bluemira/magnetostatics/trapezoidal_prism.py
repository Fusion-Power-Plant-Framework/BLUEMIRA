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
Analytical expressions for the field inside an arbitrarily shaped winding pack
of rectangular cross-section, following equations as described in:

https://onlinelibrary.wiley.com/doi/epdf/10.1002/jnm.594
including corrections from:
https://onlinelibrary.wiley.com/doi/abs/10.1002/jnm.675
"""
from typing import Union

import numba as nb
import numpy as np

from bluemira.base.constants import MU_0_4PI
from bluemira.magnetostatics.baseclass import RectangularCrossSectionCurrentSource
from bluemira.magnetostatics.error import MagnetostaticsError
from bluemira.magnetostatics.tools import process_xyz_array

__all__ = ["TrapezoidalPrismCurrentSource"]


@nb.jit(nopython=True, cache=True)
def primitive_sxn_bound(
    cos_theta: float, sin_theta: float, r: float, q: float, t: float
) -> float:
    """
    Function primitive of Bx evaluated at a bound.

    Parameters
    ----------
    cos_theta:
        The cosine of theta in radians
    sin_theta:
        The sine of theta in radians
    r:
        The r local coordinate value
    q:
        The q local coordinate value
    t:
        The t local coordinate value

    Returns
    -------
    The value of the primitive function at integral bound t

    Notes
    -----
    Uses corrected formulae available at:
    https://onlinelibrary.wiley.com/doi/abs/10.1002/jnm.675

    Singularities all resolve to: lim(ln(1)) --> 0
    """
    # First compute the divisors of each of the terms to determine if there
    # are singularities
    divisor_1 = cos_theta * np.sqrt(t**2 + q**2)
    divisor_2 = cos_theta * np.sqrt(r**2 * cos_theta**2 + q**2)
    divisor_3 = q * np.sqrt(
        t**2 + 2 * r * t * sin_theta * cos_theta + (r**2 + q**2) * cos_theta**2
    )

    # All singularities resolve to 0?
    result = 0
    if divisor_1 != 0:
        result += t * np.arcsinh((t * sin_theta + r * cos_theta) / divisor_1)
    if divisor_2 != 0:
        result += r * cos_theta * np.arcsinh((t + r * cos_theta * sin_theta) / divisor_2)
    if divisor_3 != 0:
        result += q * np.arctan((q**2 * sin_theta - t * r * cos_theta) / divisor_3)

    return result


@nb.jit(nopython=True, cache=True)
def primitive_sxn(theta: float, r: float, q: float, l1: float, l2: float) -> float:
    """
    Analytical integral of Bx function primitive.

    Parameters
    ----------
    theta:
        The angle in radians
    r:
        The r local coordinate value
    q:
        The q local coordinate value
    l1:
        The first local coordinate bound
    l2:
        The second local coordinate bound

    Returns
    -------
    The value of the integral of the Bx function primitive
    """
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    return primitive_sxn_bound(cos_theta, sin_theta, r, q, l2) - primitive_sxn_bound(
        cos_theta, sin_theta, r, q, l1
    )


@nb.jit(nopython=True, cache=True)
def primitive_szn_bound(
    cos_theta: float, sin_theta: float, r: float, ll: float, t: float
) -> float:
    """
    Function primitive of Bz evaluated at a bound.

    Parameters
    ----------
    cos_theta:
        The cosine of theta
    sin_theta:
        The sine of theta
    r:
        The r local coordinate value
    ll:
        The l local coordinate value
    t:
        The t local coordinate value

    Returns
    -------
    The value of the primitive function at integral bound t

    Notes
    -----
    Singularities all resolve to: lim(ln(1)) --> 0
    """
    sqrt_term = np.sqrt(
        t**2 * cos_theta**2
        + ll**2
        + 2 * r * ll * sin_theta * cos_theta
        + r**2 * cos_theta**2
    )

    # First compute the divisors of each of the terms to determine if there
    # are singularities
    divisor_1 = cos_theta * np.sqrt(t**2 + r**2 * cos_theta**2)
    divisor_2 = cos_theta * np.sqrt(t**2 + ll**2)
    divisor_3 = np.sqrt(
        ll**2 + 2 * ll * r * sin_theta * cos_theta + r**2 * cos_theta**2
    )
    divisor_4 = r * cos_theta * sqrt_term
    divisor_5 = ll * sqrt_term

    # All singularities resolve to 0?
    result = 0
    if divisor_1 != 0:
        result += (
            t * sin_theta * np.arcsinh((ll + r * sin_theta * cos_theta) / divisor_1)
        )
    if divisor_2 != 0:
        result -= t * np.arcsinh((ll * sin_theta + r * cos_theta) / divisor_2)
    if divisor_3 != 0:
        result -= r * cos_theta**2 * np.arcsinh((t * cos_theta) / divisor_3)
    if divisor_4 != 0:
        result -= (
            r
            * cos_theta
            * sin_theta
            * np.arctan((t * (ll + r * sin_theta * cos_theta)) / divisor_4)
        )
    if divisor_5 != 0:
        result += ll * np.arctan((t * (ll * sin_theta + r * cos_theta)) / divisor_5)

    return result


@nb.jit(nopython=True, cache=True)
def primitive_szn(theta: float, r: float, ll: float, q1: float, q2: float) -> float:
    """
    Analytical integral of Bz function primitive.

    Parameters
    ----------
    theta:
        The angle in radians
    r:
        The r local coordinate value
    ll:
        The l local coordinate value
    q1:
        The first local coordinate bound
    q2:
        The second local coordinate bound

    Returns
    -------
    The value of the integral of the Bx function primitive
    """
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    return primitive_szn_bound(cos_theta, sin_theta, r, ll, q2) - primitive_szn_bound(
        cos_theta, sin_theta, r, ll, q1
    )


@nb.jit(nopython=True, cache=True)
def Bx_analytical_prism(
    alpha: float,
    beta: float,
    l1: float,
    l2: float,
    q1: float,
    q2: float,
    r1: float,
    r2: float,
) -> float:
    """
    Calculate magnetic field in the local x coordinate direction due to a
    trapezoidal prism current source.

    Parameters
    ----------
    alpha:
        The first trapezoidal angle [rad]
    beta:
        The second trapezoidal angle [rad]
    l1:
        The local l1 coordinate [m]
    l2:
        The local l2 coordinate [m]
    q1:
        The local q1 coordinate [m]
    q2:
        The local q2 coordinate [m]
    r1:
        The local r1 coordinate [m]
    r2: float
        The local r2 coordinate [m]

    Returns
    -------
    The magnetic field response in the x coordinate direction
    """
    return (
        primitive_sxn(alpha, r1, q2, l1, l2)
        - primitive_sxn(alpha, r1, q1, l1, l2)
        + primitive_sxn(beta, r2, q2, l1, l2)
        - primitive_sxn(beta, r2, q1, l1, l2)
    )


@nb.jit(nopython=True, cache=True)
def Bz_analytical_prism(
    alpha: float,
    beta: float,
    l1: float,
    l2: float,
    q1: float,
    q2: float,
    r1: float,
    r2: float,
) -> float:
    """
    Calculate magnetic field in the local z coordinate direction due to a
    trapezoidal prism current source.

    Parameters
    ----------
    alpha:
        The first trapezoidal angle [rad]
    beta:
        The second trapezoidal angle [rad]
    l1:
        The local l1 coordinate [m]
    l2:
        The local l2 coordinate [m]
    q1:
        The local q1 coordinate [m]
    q2:
        The local q2 coordinate [m]
    r1:
        The local r1 coordinate [m]
    r2:
        The local r2 coordinate [m]

    Returns
    -------
    The magnetic field response in the z coordinate direction
    """
    return (
        primitive_szn(alpha, r1, l2, q1, q2)
        - primitive_szn(alpha, r1, l1, q1, q2)
        + primitive_szn(beta, r2, l2, q1, q2)
        - primitive_szn(beta, r2, l1, q1, q2)
    )


class TrapezoidalPrismCurrentSource(RectangularCrossSectionCurrentSource):
    """
    3-D trapezoidal prism current source with a rectangular cross-section and
    uniform current distribution.

    The current direction is along the local y coordinate.

    Parameters
    ----------
    origin:
        The origin of the current source in global coordinates [m]
    ds:
        The direction vector of the current source in global coordinates [m]
    normal:
        The normalised normal vector of the current source in global coordinates [m]
    t_vec:
        The normalised tangent vector of the current source in global coordinates [m]
    breadth:
        The breadth of the current source (half-width) [m]
    depth:
        The depth of the current source (half-height) [m]
    alpha:
        The first angle of the trapezoidal prism [°] [0, 180)
    beta:
        The second angle of the trapezoidal prism [°] [0, 180)
    current:
        The current flowing through the source [A]

    Notes
    -----
    Negative angles are allowed, but both angles must be either 0 or negative.
    """

    def __init__(
        self,
        origin: np.ndarray,
        ds: np.ndarray,
        normal: np.ndarray,
        t_vec: np.ndarray,
        breadth: float,
        depth: float,
        alpha: float,
        beta: float,
        current: float,
    ):
        alpha, beta = np.deg2rad(alpha), np.deg2rad(beta)
        self.origin = origin

        length = np.linalg.norm(ds)
        self._check_angle_values(alpha, beta)
        self._check_raise_self_intersection(length, breadth, alpha, beta)
        self._halflength = 0.5 * length
        # Normalised direction cosine matrix
        self.dcm = np.array([t_vec, ds / length, normal])
        self.length = 0.5 * (length - breadth * np.tan(alpha) - breadth * np.tan(beta))
        self.breadth = breadth
        self.depth = depth
        self.alpha = alpha
        self.beta = beta
        # Current density
        self.rho = current / (4 * breadth * depth)
        self.points = self._calculate_points()

    def _check_angle_values(self, alpha, beta):
        """
        Check that end-cap angles are acceptable.
        """
        sign_alpha = np.sign(alpha)
        sign_beta = np.sign(beta)
        one_zero = np.any(np.array([sign_alpha, sign_beta]) == 0.0)  # noqa: PLR2004
        if not one_zero and sign_alpha != sign_beta:
            raise MagnetostaticsError(
                f"{self.__class__.__name__} instantiation error: end-cap angles "
                f"must have the same sign {alpha=:.3f}, {beta=:.3f}."
            )
        if not (0 <= abs(alpha) < 0.5 * np.pi):
            raise MagnetostaticsError(
                f"{self.__class__.__name__} instantiation error: {alpha=:.3f} is outside"
                " bounds of [0, 180°)."
            )
        if not (0 <= abs(beta) < 0.5 * np.pi):
            raise MagnetostaticsError(
                f"{self.__class__.__name__} instantiation error: {beta=:.3f} is outside "
                "bounds of [0, 180°)."
            )

    def _check_raise_self_intersection(
        self, length: float, breadth: float, alpha: float, beta: float
    ):
        """
        Check for bad combinations of source length and end-cap angles.
        """
        a = np.tan(alpha) * breadth
        b = np.tan(beta) * breadth
        if (a + b) > length:
            raise MagnetostaticsError(
                f"{self.__class__.__name__} instantiation error: source length and "
                "angles imply a self-intersecting trapezoidal prism."
            )

    def _xyzlocal_to_rql(self, x_local, y_local, z_local):
        """
        Convert local x, y, z coordinates to working coordinates.
        """
        b = self.length
        c = self.depth
        d = self.breadth

        l1 = -d - x_local
        l2 = d - x_local
        q1 = -c - z_local
        q2 = c - z_local
        r1 = (d + x_local) * np.tan(self.alpha) + b - y_local
        r2 = (d + x_local) * np.tan(self.beta) + b + y_local
        return l1, l2, q1, q2, r1, r2

    def _BxByBz(self, point):
        """
        Calculate the field at a point in local coordinates.
        """
        l1, l2, q1, q2, r1, r2 = self._xyzlocal_to_rql(*point)
        bx = Bx_analytical_prism(self.alpha, self.beta, l1, l2, q1, q2, r1, r2)
        bz = Bz_analytical_prism(self.alpha, self.beta, l1, l2, q1, q2, r1, r2)
        return np.array([bx, 0, bz])

    @process_xyz_array
    def field(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Calculate the magnetic field at a point due to the current source.

        Parameters
        ----------
        x:
            The x coordinate(s) of the points at which to calculate the field
        y:
            The y coordinate(s) of the points at which to calculate the field
        z:
            The z coordinate(s) of the points at which to calculate the field

        Returns
        -------
        The magnetic field vector {Bx, By, Bz} in [T]
        """
        point = np.array([x, y, z])
        # Convert to local coordinates
        point = self._global_to_local([point])[0]
        # Evaluate field in local coordinates
        b_local = MU_0_4PI * self.rho * self._BxByBz(point)
        # Convert vector back to global coordinates
        return self.dcm.T @ b_local

    def _calculate_points(self):
        """
        Calculate extrema points of the current source for plotting and debugging.
        """
        b = self._halflength
        c = self.depth
        d = self.breadth
        # Lower rectangle
        p1 = np.array([-d, -b + d * np.tan(self.beta), -c])
        p2 = np.array([d, -b - d * np.tan(self.beta), -c])
        p3 = np.array([d, -b - d * np.tan(self.beta), c])
        p4 = np.array([-d, -b + d * np.tan(self.beta), c])

        # Upper rectangle
        p5 = np.array([-d, b - d * np.tan(self.alpha), -c])
        p6 = np.array([d, b + d * np.tan(self.alpha), -c])
        p7 = np.array([d, b + d * np.tan(self.alpha), c])
        p8 = np.array([-d, b - d * np.tan(self.alpha), c])

        points = [
            np.vstack([p1, p2, p3, p4, p1]),
            np.vstack([p5, p6, p7, p8, p5]),
            # Lines between rectangle corners
            np.vstack([p1, p5]),
            np.vstack([p2, p6]),
            np.vstack([p3, p7]),
            np.vstack([p4, p8]),
        ]

        return np.array([self._local_to_global(p) for p in points], dtype=object)
