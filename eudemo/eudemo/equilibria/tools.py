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
"""Utility functions related to EUDEMO equilibria calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from bluemira.base.parameter_frame import ParameterFrame
    from bluemira.geometry.parameterisations import GeometryParameterisation
    from bluemira.geometry.wire import BluemiraWire

import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.optimisation.constraints import (
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
    PsiBoundaryConstraint,
)
from bluemira.equilibria.shapes import flux_surface_johner
from bluemira.geometry.coordinates import Coordinates, interpolate_points


def estimate_kappa95(A: float, m_s_limit: float) -> float:
    """
    Estimate the maximum kappa_95 for a given aspect ratio and margin to
    stability. It is always better to have as high a kappa_95 as possible, so
    we maximise it here, for a specified margin to stability value.

    Parameters
    ----------
    A:
        The aspect ratio of the plasma
    m_s_limit:
        The margin to stability (typically ~0.3)

    Returns
    -------
    The maximum elongation for the specified input values

    Notes
    -----
    The model used here is a 2nd order polynomial surface fit, generated using
    data from CREATE. A quadratic equation is then solved for kappa_95, based
    on the polynomial surface fit.
    The data are stored in: data/equilibria/vertical_stability_data.json
    For the A=2.6, m_s=0 case (a bit of an outlier), there is a fudging to cap the
    kappa_95 to ~1.8 (which is the recommended value). The fit otherwise overestimates
    kappa_95 in this corner of the space (kappa_95 ~ 1.81)
    This is only a crude model, and is only relevant for EU-DEMO-like machines.
    Furthermore, this is only for flat-top..! Ramp-up and ramp-down may be
    design driving. Exercise caution.
    \t:math:`m_{s} = a\\kappa_{95}^{2}+bA^{2}+c\\kappa A+d\\kappa+eA+f`\n
    \t:math:`\\kappa_{95}(A, m_{s}) = \\dfrac{-d-cA-\\sqrt{(c^{2}-4ab)A^{2}+(2dc-4ae)A+d^{2}-4af+4am_{s})}}{2a}`
    """  # noqa: W505, E501
    if not 2.6 <= A <= 3.6:  # noqa: PLR2004
        bluemira_warn(f"Kappa 95 estimate only valid for 2.6 <= A <= 3.6, not A = {A}")
    if not 0.0 <= m_s_limit <= 0.8655172413793104:  # noqa: PLR2004
        bluemira_warn(
            "Kappa 95 estimate only valid for 0.0 <= m_s <= 0.865, not m_s ="
            f" {m_s_limit}"
        )

    a = 3.68436807
    b = -0.27706527
    c = 0.87040251
    d = -18.83740952
    e = -0.27267618
    f = 20.5141261

    kappa_95 = (
        -d
        - c * A
        - np.sqrt(
            (c**2 - 4 * a * b) * A**2
            + (2 * d * c - 4 * a * e) * A
            + d**2
            - 4 * a * f
            + 4 * a * m_s_limit
        )
    ) / (2 * a)

    # We're going to trim kappa_95 to 1.8, which is the maximum of the data, keeping
    # the function smooth
    if kappa_95 > 1.77:  # noqa: PLR2004
        ratio = 1.77 / kappa_95
        corner_fudge = 0.3 * (kappa_95 - 1.77) / ratio
        kappa_95 = kappa_95 ** (ratio) + corner_fudge

    return kappa_95


def handle_lcfs_shape_input(
    param_cls: GeometryParameterisation,
    params: ParameterFrame,
    shape_config: Dict[str, float],
) -> Dict[str, float]:
    """
    Process the LCFS shape parameterisation inputs based on a parameterisation
    and a shape configuration.

    Parameters
    ----------
    param_cls:
        LCFS geometry parameterisation
    params:
        Parameters of the reactor
    shape_config:
        Dictionary with the various shape configuration keys, which can be specific
        to the geometry parameterisation

    Returns
    -------
    Input dictionary for the initialisation of the specified GeometryParameterisation
    """
    defaults = {
        "f_kappa_l": 1.0,
        "f_delta_l": 1.0,
    }
    shape_config = {**defaults, **shape_config}
    kappa_95 = params.kappa_95.value
    delta_95 = params.delta_95.value

    kappa_factor = shape_config.pop("f_kappa_l")
    delta_factor = shape_config.pop("f_delta_l")
    if "kappa_l" not in shape_config:
        shape_config["kappa_l"] = kappa_factor * kappa_95
    if "kappa_u" not in shape_config:
        shape_config["kappa_u"] = kappa_factor**0.5 * kappa_95
    if "delta_l" not in shape_config:
        shape_config["delta_l"] = delta_factor * delta_95
    if "delta_u" not in shape_config:
        shape_config["delta_u"] = delta_95

    input_dict = {
        "r_0": {"value": params.R_0.value},
        "a": {"value": params.R_0.value / params.A.value},
    }

    param_cls_instance = param_cls()

    for k, v in shape_config.items():
        if k in param_cls_instance.variables.names:
            input_dict[k] = {"value": v}
        else:
            bluemira_warn(
                f"Unknown shape parameter {k} for GeometryParameterisation:"
                f" {param_cls_instance.name}"
            )
    return input_dict


def make_grid(
    R_0: float, A: float, kappa: float, grid_settings: Dict[str, float]
) -> Grid:
    """
    Make a finite difference Grid for an Equilibrium.

    Parameters
    ----------
    R_0:
        Major radius
    A:
        Aspect ratio
    kappa:
        Elongation
    grid_settings:
        Dictionary of grid settings

    Returns
    -------
    Finite difference grid for an Equilibrium
    """
    defaults = {
        "grid_scale_x": 2.0,
        "grid_scale_z": 2.0,
        "nx": 65,
        "nz": 65,
    }
    grid_settings = {**defaults, **grid_settings}
    scale_x = grid_settings["grid_scale_x"]
    scale_z = grid_settings["grid_scale_z"]
    nx = grid_settings["nx"]
    nz = grid_settings["nz"]

    x_min, x_max = R_0 - scale_x * (R_0 / A), R_0 + scale_x * (R_0 / A)
    z_min, z_max = -scale_z * (kappa * R_0 / A), scale_z * (kappa * R_0 / A)
    return Grid(x_min, x_max, z_min, z_max, nx, nz)


class DivertorLegCalculator:
    """
    Straight line divertor leg mixin calculator.
    """

    @staticmethod
    def calc_line(p1, p2, n):
        """
        Calculate a linearly spaced series of points on a line between p1 and p2.
        """
        xn = np.linspace(p1[0], p2[0], int(n))
        zn = np.linspace(p1[1], p2[1], int(n))
        return xn, zn

    def calc_divertor_leg(self, x_point, angle, length, n, loc="lower", pos="outer"):
        """
        Calculate the position of a straight line divertor leg.
        """
        if loc not in ["upper", "lower"]:
            raise ValueError(
                f"Please specify loc: 'upper' or 'lower' X-point, not: {loc}"
            )
        if pos not in ["inner", "outer"]:
            raise ValueError(f"Please specify pos: 'inner' or 'outer' X leg, not: {pos}")

        loc_sign = 1 if loc == "upper" else -1
        pos_sign = 1 if pos == "outer" else -1

        angle = np.deg2rad(angle)
        x = x_point[0] + pos_sign * length * np.cos(angle)
        z = x_point[1] + loc_sign * length * np.sin(angle)

        return self.calc_line(x_point, (x, z), n)


class EUDEMOSingleNullConstraints(DivertorLegCalculator, MagneticConstraintSet):
    """
    Parameterised family of magnetic constraints for a typical EU-DEMO-like single
    null equilibrium.
    """

    def __init__(
        self,
        R_0: float,
        Z_0: float,
        A: float,
        kappa_u: float,
        kappa_l: float,
        delta_u: float,
        delta_l: float,
        psi_u_neg: float,
        psi_u_pos: float,
        psi_l_neg: float,
        psi_l_pos: float,
        div_l_ib: float,
        div_l_ob: float,
        psibval: float,
        psibtol: float = 1e-3,
        lower: float = True,
        n: int = 100,
    ):
        constraints = []
        f_s = flux_surface_johner(
            R_0,
            Z_0,
            R_0 / A,
            kappa_u,
            kappa_l,
            delta_u,
            delta_l,
            psi_u_neg,
            psi_u_pos,
            psi_l_neg,
            psi_l_pos,
            n=200,
        )

        arg_x = np.argmin(f_s.z) if lower else np.argmax(f_s.z)

        x_point = [f_s.x[arg_x], f_s.z[arg_x]]

        constraints = [FieldNullConstraint(*x_point)]

        f_s = Coordinates(interpolate_points(*f_s.xyz, n))

        x_s, z_s = f_s.x, f_s.z

        constraints.append(PsiBoundaryConstraint(x_s, z_s, psibval, tolerance=psibtol))

        x_leg1, z_leg1 = self.calc_divertor_leg(
            x_point, 50, div_l_ob, int(n / 10), loc="lower", pos="outer"
        )

        x_leg2, z_leg2 = self.calc_divertor_leg(
            x_point, 40, div_l_ib, int(n / 10), loc="lower", pos="inner"
        )

        x_legs = np.append(x_leg1, x_leg2)
        z_legs = np.append(z_leg1, z_leg2)
        constraints.append(
            PsiBoundaryConstraint(x_legs, z_legs, psibval, tolerance=psibtol)
        )

        super().__init__(constraints)


class EUDEMODoubleNullConstraints(DivertorLegCalculator, MagneticConstraintSet):
    """
    Parameterised family of magnetic constraints for a typical EU-DEMO-like double
    null equilibrium.
    """

    def __init__(
        self,
        R_0: float,
        Z_0: float,
        A: float,
        kappa: float,
        delta: float,
        psi_neg: float,
        psi_pos: float,
        div_l_ib: float,  # noqa: ARG002
        div_l_ob: float,  # noqa: ARG002
        psibval: float,
        n: int = 400,
    ):
        super().__init__()
        f_s = flux_surface_johner(
            R_0,
            Z_0,
            R_0 / A,
            kappa,
            kappa,
            delta,
            delta,
            psi_neg,
            psi_pos,
            psi_neg,
            psi_pos,
            n=200,
        )

        arg_xl = np.argmin(f_s.z)
        arg_xu = np.argmax(f_s.z)
        constraints = [
            FieldNullConstraint(f_s.x[arg_xl], f_s.z[arg_xl]),
            FieldNullConstraint(f_s.x[arg_xu], f_s.z[arg_xu]),
        ]
        f_s = Coordinates(interpolate_points(*f_s.xyz, n))
        x_s, z_s = f_s.x, f_s.z

        constraints.append(PsiBoundaryConstraint(x_s, z_s, psibval))

        super().__init__(constraints)


class ReferenceConstraints(MagneticConstraintSet):
    """
    Parameters
    ----------
    shape:
        Geometry from which to build the reference constraints for the equilibrium
    n_points:
        Number of points to use when creating the constraints
    """

    def __init__(self, shape: BluemiraWire, n_points: int):
        coords = shape.discretize(byedges=True, ndiscr=n_points)
        z_min = np.min(coords.z)
        z_max = np.max(coords.z)
        arg_xl = np.argmin(coords.z)
        arg_xu = np.argmax(coords.z)
        arg_xin = np.argmin(coords.x)

        if np.isclose(abs(z_min), z_max):
            # Double null
            constraints = [
                FieldNullConstraint(coords.x[arg_xl], coords.z[arg_xl]),
                FieldNullConstraint(coords.x[arg_xu], coords.z[arg_xu]),
            ]

        elif abs(z_min) > z_max:
            # Lower single null
            constraints = [
                FieldNullConstraint(
                    coords.x[arg_xl], coords.z[arg_xl], weights=n_points // 5
                ),
                # TODO: This is a hack so I can move on with my life. I'm not even sorry.
                FieldNullConstraint(
                    0.85 * coords.x[arg_xu],
                    1.35 * coords.z[arg_xu],
                    weights=n_points // 5,
                ),
            ]

        else:
            # Upper single null
            constraints = [
                FieldNullConstraint(coords.x[arg_xu], coords.z[arg_xu]),
            ]

        constraints.append(
            IsofluxConstraint(
                coords.x, coords.z, coords.x[arg_xin], coords.z[arg_xin], tolerance=1e-6
            )
        )

        super().__init__(constraints)
