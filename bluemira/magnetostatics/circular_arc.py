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
Analytical expressions for the field due to a circular current arc of
rectangular cross-section, following equations as described in:

https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1064259
"""
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.constants import MU_0_4PI
from bluemira.geometry._private_tools import make_circle_arc
from bluemira.magnetostatics.baseclass import RectangularCrossSectionCurrentSource
from bluemira.magnetostatics.tools import (
    integrate,
    jit_llc3,
    jit_llc4,
    process_xyz_array,
)

__all__ = ["CircularArcCurrentSource"]


# Full integrands free of singularities


@jit_llc4
def brc_integrand_full(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the Brc integrand without singularities.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    The result of the integrand at a single point
    """
    cos_psi = np.cos(psi)
    sqrt_term = np.sqrt(r_pc**2 - 2 * r_pc * r_j * cos_psi + r_j**2 + z_k**2)
    return cos_psi * sqrt_term + r_pc * cos_psi**2 * np.log(
        r_j - r_pc * cos_psi + sqrt_term
    )


@jit_llc4
def bzc_integrand_full_p1(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the Bzc integrand without singularities.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    The result of the integrand at a single point
    """
    cos_psi = np.cos(psi)
    sqrt_term = np.sqrt(r_pc**2 - 2 * r_pc * r_j * cos_psi + r_j**2 + z_k**2)
    return -z_k * np.log(r_j - r_pc * cos_psi + sqrt_term) - r_pc * cos_psi * np.log(
        -z_k + sqrt_term
    )


# Integrands to treat singularities


@jit_llc4
def brc_integrand_p1(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the first part of the Brc integrand (no singularities)

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    The result of the integrand at a single point
    """
    cos_psi = np.cos(psi)
    sqrt_term = np.sqrt(r_pc**2 - 2 * r_pc * r_j * cos_psi + r_j**2 + z_k**2)
    return cos_psi * sqrt_term


@jit_llc4
def bf1_r_pccos2_integrand(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the BF1(r_pc*cos(psi)^2) integrand

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    The result of the integrand at a single point
    """
    cos_psi = np.cos(psi)
    sqrt_term = np.sqrt(r_pc**2 - 2 * r_pc * r_j * cos_psi + r_j**2 + z_k**2)
    return r_pc * cos_psi**2 * np.log(r_j - r_pc * cos_psi + sqrt_term)


@jit_llc3
def bf1_r_pccos2_0_pi_integrand_p1(psi: float, r_pc: float, r_j: float) -> float:
    """
    Calculate the BF1(r_pc*cos(psi)^2) integrand for a 0 to pi integral

    From 0 to pi for r_j < r_pc and z_k == 0. Part 1

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    The result of the integrand at a single point for 0 to pi integral
    """
    cos_psi = np.cos(psi)
    return (
        r_pc
        * cos_psi**2
        * np.log(
            (r_pc * cos_psi - r_j)
            + np.sqrt((r_pc * cos_psi - r_j) ** 2 + r_pc**2 * np.sin(psi) ** 2)
        )
    )


@jit_llc3
def bf1_r_pccos2_0_pi_integrand_p2(psi: float, r_pc: float, r_j: float) -> float:
    """
    Calculate the BF1(r_pc*cos(psi)^2) integrand for a 0 to pi integral

    From 0 to pi for r_j < r_pc and z_k == 0. Part 2

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    The result of the integrand at a single point for 0 to pi integral
    """
    cos_psi = np.cos(psi)
    return (
        r_pc
        * cos_psi**2
        * np.log(
            (r_j - r_pc * cos_psi)
            + np.sqrt((r_j - r_pc * cos_psi) ** 2 + r_pc**2 * np.sin(psi) ** 2)
        )
    )


@jit_llc4
def bf1_zk_integrand(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the BF1(-z_k) integrand for a 0 to pi integral

    From 0 to pi for r_j < r_pc and z_k == 0.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    The result of the integrand at a single point
    """
    cos_psi = np.cos(psi)
    sqrt_term = np.sqrt(r_pc**2 - 2 * r_pc * r_j * cos_psi + r_j**2 + z_k**2)
    return -z_k * np.log(r_j - r_pc * cos_psi + sqrt_term)


@jit_llc4
def bf2_integrand(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the BF2 integrand.

    For r_j == r_pc and z_k >= 0.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    The result of the integrand at a single point
    """
    cos_psi = np.cos(psi)
    sqrt_term = np.sqrt(r_pc**2 - 2 * r_pc * r_j * cos_psi + r_j**2 + z_k**2)
    return -r_pc * cos_psi * np.log(-z_k + sqrt_term)


@jit_llc3
def bf2_0_pi_integrand(psi: float, r_pc: float, z_k: float) -> float:
    """
    Calculate the BF2 integrand for a 0 to pi integral

    From 0 to pi for r_j == r_pc and z_k >= 0.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    The result of the integrand at a single point for 0 to pi integral
    """
    cos_psi = np.cos(psi)
    return (
        r_pc * cos_psi * np.log(z_k + np.sqrt(2 * r_pc**2 * (1 - cos_psi) + z_k**2))
    )


@jit_llc4
def bf3_integrand(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the BF3 integrand

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    The result of the integrand at a single point

    Notes
    -----
    Treats the sin(psi) = 0 singularity
    """
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    if sin_psi != 0:
        sqrt_term = np.sqrt(r_pc**2 - 2 * r_pc * r_j * cos_psi + r_j**2 + z_k**2)
        return (
            r_pc
            * sin_psi
            * np.arctan((z_k * (r_j - r_pc * cos_psi)) / (r_pc * sin_psi * sqrt_term))
        )
    return 0


# More singularity treatments...


def bf1_r_pccos2_zk0_0_pi(r_pc: float, r_j: float) -> float:
    """
    Calculate the BF1(r_pc*cos(psi)^2) integral for 0 to pi.

    From 0 to pi for r_j <= r_pc and z_k == 0

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]

    Returns
    -------
    The result of the integral at a single point for 0 to pi integral
    """
    if r_pc == r_j:
        return r_pc * (0.5 * np.pi * np.log(r_pc) + 0.2910733)
    # r_j < r_pc
    result = 0.5 * np.pi * r_pc * (np.log(r_pc) - np.log(2) - 0.5)
    result -= integrate(bf1_r_pccos2_0_pi_integrand_p1, (r_pc, r_j), 0, 0.5 * np.pi)
    result += integrate(bf1_r_pccos2_0_pi_integrand_p2, (r_pc, r_j), 0.5 * np.pi, np.pi)
    return result


def bf2_rj_rpc_0_pi(r_pc: float, z_k: float) -> float:
    """
    Calculate the BF2 integrand for a 0 to pi integral

    From 0 to pi for r_j < r_pc and z_k == 0.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]

    Returns
    -------
    The result of the integral for 0 to pi
    """
    return np.pi * r_pc + integrate(bf2_0_pi_integrand, (r_pc, z_k), 0, np.pi)


# Primitive functions


def primitive_brc(
    r_pc: float, r_j: float, z_k: float, phi_pc: float, theta: float
) -> float:
    """
    Calculate the Brc primitives and treat singularities.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    phi_pc:
        Angle of the point at which to evaluate field [rad]
    theta:
        Azimuthal angle of the circular arc

    Returns
    -------
    The result of the Brc primitive
    """
    args = (r_pc, r_j, z_k)  # The function arguments for integration
    singularities = (z_k == 0) and (r_j <= r_pc) and (0 <= phi_pc <= theta)
    if not singularities:
        # No singularities
        return integrate(brc_integrand_full, args, -phi_pc, theta - phi_pc)

    # Treat singularities
    # Singularity free treatment of first term
    result = integrate(brc_integrand_p1, args, -phi_pc, theta - phi_pc)

    # Dodge singularities in second term
    if phi_pc == 0:
        if theta == np.pi:
            result += bf1_r_pccos2_zk0_0_pi(r_pc, r_j)

        if theta == 2 * np.pi:
            # cos(-psi)^2 == cos(psi)^2
            result += 2 * bf1_r_pccos2_zk0_0_pi(r_pc, r_j)
        else:
            result += bf1_r_pccos2_zk0_0_pi(r_pc, r_j)
            result -= integrate(bf1_r_pccos2_integrand, args, -theta, np.pi - theta)

    elif phi_pc == theta:
        if phi_pc == np.pi:
            result += bf1_r_pccos2_zk0_0_pi(r_pc, r_j)
        else:
            # cos(-psi)^2 == cos(psi)^2
            result += bf1_r_pccos2_zk0_0_pi(r_pc, r_j)
            result -= integrate(bf1_r_pccos2_integrand, args, np.pi, np.pi - theta)
    else:
        # cos(-psi)^2 == cos(psi)^2
        result += 2 * bf1_r_pccos2_zk0_0_pi(r_pc, r_j)
        result -= integrate(
            bf1_r_pccos2_integrand, args, phi_pc - theta, 2 * np.pi - theta
        )

    return result


def primitive_bzc(
    r_pc: float, r_j: float, z_k: float, phi_pc: float, theta: float
) -> float:
    """
    Calculate the Bzc primitives and treat singularities.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    phi_pc:
        Angle of the point at which to evaluate field [rad]
    theta:
        Azimuthal angle of the circular arc

    Returns
    -------
    The result of the Bzc primitive
    """
    args = (r_pc, r_j, z_k)  # The function arguments for integration
    bf1_singularities = (z_k == 0) and (r_j <= r_pc) and (0 <= phi_pc <= theta)
    bf2_singularities = (r_j == r_pc) and (z_k >= 0) and (0 <= phi_pc <= theta)
    bf3_singularities = r_pc == 0
    if not bf1_singularities and not bf2_singularities and not bf3_singularities:
        # No singularities (almost)
        return integrate(
            bzc_integrand_full_p1, args, -phi_pc, theta - phi_pc
        ) + integrate(bf3_integrand, args, -phi_pc, theta - phi_pc)

    # Treat singularities
    result = 0
    if bf1_singularities:
        # Treat BF1(-z_k)
        if phi_pc == 0 and theta != np.pi and theta != 2 * np.pi:
            # At pi and 2 * pi the BF1 integral is 0
            # Elsewhere:
            # result += 0 (the first part of BF1 is 0)
            result -= integrate(bf1_zk_integrand, args, -theta, np.pi - theta)

        if phi_pc == theta:
            # result += 0 (the first part of BF1 is 0)
            result -= integrate(bf1_zk_integrand, args, np.pi, np.pi - theta)

        else:
            # result += 0 (the first part of BF1 is 0)
            result -= integrate(
                bf1_zk_integrand, args, phi_pc - theta, 2 * np.pi - theta
            )

    else:
        # BF1 is normal
        result += integrate(bf1_zk_integrand, args, -phi_pc, theta - phi_pc)

    if bf2_singularities:
        # Treat BF2
        if phi_pc == 0:
            result += bf2_rj_rpc_0_pi(r_pc, z_k)
            result -= integrate(bf2_integrand, args, -theta, np.pi - theta)

        if phi_pc == theta:
            result += bf2_rj_rpc_0_pi(r_pc, z_k)
            result -= integrate(bf2_integrand, args, np.pi, np.pi - theta)

        else:
            result += 2 * bf2_rj_rpc_0_pi(r_pc, z_k)
            result -= integrate(bf2_integrand, args, phi_pc - theta, 2 * np.pi - theta)

    else:
        # BF2 is normal
        result += integrate(bf2_integrand, args, -phi_pc, theta - phi_pc)

    if r_pc != 0:
        # r_pc = 0, BF3 evaluates to 0
        result += integrate(bf3_integrand, args, -phi_pc, theta - phi_pc)

    return result


# Full field calculations in working coordinates


def Bx_analytical_circular(
    r1: float, r2: float, z1: float, z2: float, theta: float, r_p: float, theta_p: float
) -> float:
    """
    Calculate magnetic field in the local x coordinate direction due to a
    circular arc current source.

    Parameters
    ----------
    r1:
        Inner coil radius [m]
    r2:
        Outer coil radius [m]
    z1:
        The first modified z coordinate [m]
    z2:
        The second modified z coordinate [m]
    theta:
        Azimuthal angle of the circular arc [rad]
    r_p:
        The radius of the point at which to evaluate the field [m]
    theta_p:
        The angle of the point at which to evaluate the field [rad]

    Returns
    -------
    The magnetic field response in the x coordinate direction
    """
    return (
        primitive_brc(r_p, r1, z1, theta_p, theta)
        - primitive_brc(r_p, r1, z2, theta_p, theta)
        - primitive_brc(r_p, r2, z1, theta_p, theta)
        + primitive_brc(r_p, r2, z2, theta_p, theta)
    )


def Bz_analytical_circular(
    r1: float, r2: float, z1: float, z2: float, theta: float, r_p: float, theta_p: float
) -> float:
    """
    Calculate magnetic field in the local z coordinate direction due to a
    circular arc current source.

    Parameters
    ----------
    r1:
        Inner coil radius [m]
    r2:
        Outer coil radius [m]
    z1:
        The first modified z coordinate [m]
    z2:
        The second modified z coordinate [m]
    theta:
        Azimuthal angle of the circular arc [rad]
    r_p:
        The radius of the point at which to evaluate the field [m]
    theta_p:
        The angle of the point at which to evaluate the field [rad]

    Returns
    -------
    The magnetic field response in the z coordinate direction
    """
    return (
        primitive_bzc(r_p, r1, z1, theta_p, theta)
        - primitive_bzc(r_p, r1, z2, theta_p, theta)
        - primitive_bzc(r_p, r2, z1, theta_p, theta)
        + primitive_bzc(r_p, r2, z2, theta_p, theta)
    )


class CircularArcCurrentSource(RectangularCrossSectionCurrentSource):
    """
    3-D circular arc prism current source with a rectangular cross-section and
    uniform current distribution.

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
    radius:
        The radius of the circular arc from the origin [m]
    dtheta:
        The azimuthal width of the arc [°]
    current:
        The current flowing through the source [A]

    Notes
    -----
    The origin is at the centre of the circular arc, with the ds vector pointing
    towards the start of the circular arc.

    Cylindrical coordinates are used for calculations under the hood.
    """

    def __init__(
        self,
        origin: np.ndarray,
        ds: np.ndarray,
        normal: np.ndarray,
        t_vec: np.ndarray,
        breadth: float,
        depth: float,
        radius: float,
        dtheta: float,
        current: float,
    ):
        self.origin = origin
        self._breadth = breadth
        self.depth = depth
        self.length = 0.5 * (breadth + depth)  # For plotting only
        self._radius = radius
        self._update_r1r2()

        self.dtheta = np.deg2rad(dtheta)
        self.rho = current / (4 * breadth * depth)
        self.dcm = np.array([ds, normal, t_vec])
        self.points = self._calculate_points()

    @property
    def radius(self) -> float:
        """
        Get the radius of the CircularArcCurrentSource
        """
        return self._radius

    @radius.setter
    def radius(self, radius: float):
        """
        Set the radius.

        Parameters
        ----------
        radius:
            The radius of the CircularArcCurrentSource
        """
        self._radius = radius
        self._update_r1r2()

    @property
    def breadth(self) -> float:
        """
        Get the breadth of the CircularArcCurrentSource.
        """
        return self._breadth

    @breadth.setter
    def breadth(self, breadth: float):
        """
        Set the breadth of the CircularArcCurrentSource.

        Parameters
        ----------
        breadth:
            The breadth of the CircularArcCurrentSource
        """
        self._breadth = breadth
        self._update_r1r2()

    def _update_r1r2(self):
        """
        Update
        """
        self._r1 = self.radius - self.breadth
        self._r2 = self.radius + self.breadth

    @staticmethod
    def _local_to_cylindrical(point: np.ndarray) -> np.ndarray:
        """
        Convert from local to cylindrical coordinates.
        """
        x, y, z = point
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return np.array([rho, theta, z])

    def _cylindrical_to_working(self, zp: float) -> Tuple[float, float, float, float]:
        """
        Convert from local cylindrical coordinates to working coordinates.
        """
        z1 = zp + self.depth
        z2 = zp - self.depth
        return self._r1, self._r2, z1, z2

    def _BxByBz(self, rp: float, tp: float, zp: float) -> np.ndarray:
        """
        Calculate the field at a point in local coordinates.
        """
        r1, r2, z1, z2 = self._cylindrical_to_working(zp)
        bx = Bx_analytical_circular(r1, r2, z1, z2, self.dtheta, rp, tp)
        bz = Bz_analytical_circular(r1, r2, z1, z2, self.dtheta, rp, tp)
        return np.array([bx, 0, bz])

    @process_xyz_array
    def field(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
    ) -> np.ndarray:
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
        # Convert to local cylindrical coordinates
        point = self._global_to_local([point])[0]
        rp, tp, zp = self._local_to_cylindrical(point)
        # Calculate field in local coordinates
        b_local = MU_0_4PI * self.rho * self._BxByBz(rp, tp, zp)
        # Convert field to global coordinates
        return self.dcm.T @ b_local

    def _calculate_points(self):
        """
        Calculate extrema points of the current source for plotting and debugging.
        """
        r = self.radius
        a = self.breadth
        b = self.depth

        # Circle arcs
        n = 200
        theta = self.dtheta
        ones = np.ones(n)
        arc_1x, arc_1y = make_circle_arc(r - a, 0, 0, angle=theta, n_points=n)
        arc_2x, arc_2y = make_circle_arc(r + a, 0, 0, angle=theta, n_points=n)
        arc_3x, arc_3y = make_circle_arc(r + a, 0, 0, angle=theta, n_points=n)
        arc_4x, arc_4y = make_circle_arc(r - a, 0, 0, angle=theta, n_points=n)
        arc_1 = np.array([arc_1x, arc_1y, -b * ones]).T
        arc_2 = np.array([arc_2x, arc_2y, -b * ones]).T
        arc_3 = np.array([arc_3x, arc_3y, b * ones]).T
        arc_4 = np.array([arc_4x, arc_4y, b * ones]).T

        n_slices = int(2 + self.dtheta // (0.25 * np.pi))
        slices = np.linspace(0, n - 1, n_slices, endpoint=True, dtype=int)
        points = [arc_1, arc_2, arc_3, arc_4]

        # Rectangles
        points.extend(
            [
                np.vstack([arc_1[s], arc_2[s], arc_3[s], arc_4[s], arc_1[s]])
                for s in slices
            ]
        )

        return np.array([self._local_to_global(p) for p in points], dtype=object)

    def plot(self, ax: Optional[plt.Axes] = None, show_coord_sys: bool = False):
        """
        Plot the CircularArcCurrentSource.

        Parameters
        ----------
        ax: Union[None, Axes]
            The matplotlib axes to plot on
        show_coord_sys: bool
            Whether or not to plot the coordinate systems
        """
        super().plot(ax=ax, show_coord_sys=show_coord_sys)
        ax = plt.gca()
        theta = self.dtheta
        x, y = make_circle_arc(
            self.radius, 0, 0, angle=theta / 2, start_angle=theta / 4, n_points=200
        )
        centre_arc = np.array([x, y, np.zeros(200)]).T
        points = self._local_to_global(centre_arc)
        ax.plot(*points.T, color="r")
        ax.plot([points[-1][0]], [points[-1][1]], [points[-1][2]], marker="^", color="r")
