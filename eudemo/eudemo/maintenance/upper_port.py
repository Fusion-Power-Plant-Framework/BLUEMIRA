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
Some crude EU-DEMO remote maintenance considerations.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np

from bluemira.base.constants import EPS
from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import make_polygon, slice_shape
from bluemira.optimisation import OptimisationProblem
from bluemira.optimisation.typing import ConstraintT
from eudemo.tools import get_inner_cut_point


class UpperPortOP(OptimisationProblem):
    """
    Collection of functions to use to minimise the upper port size.

    Parameters
    ----------
    bb:
        The xz-silhouette of the breeding blanket [m].
    c_rm:
        The required remote maintenance clearance [m].
    R_0:
        The tokamak major radius [m].
    tk_bb_ib:
        Blanket inboard thickness [m].
    tk_bb_ob:
        Blanket outboard thickness [m].
    bb_min_angle:
        Minimum blanket module angle [degrees].
    """

    def __init__(
        self,
        bb: BluemiraFace,
        c_rm: float,
        R_0: float,
        tk_bb_ib: float,
        tk_bb_ob: float,
        bb_min_angle: float,
    ):
        self.bb = bb
        self.c_rm = c_rm
        self.R_0 = R_0
        self.tk_bb_ib = tk_bb_ib
        self.bb_min_angle = bb_min_angle
        self.tk_bb_ob = tk_bb_ob
        self.r_ib_min = self.bb.bounding_box.x_min
        self.r_ob_max = self.bb.bounding_box.x_max
        self.gradient = np.array([-1, 1, 0, 1], dtype=float)

    def objective(self, x: np.ndarray) -> float:
        """The objective function of the optimisation."""
        return self.port_size(x)

    def df_objective(self, x: np.ndarray) -> np.ndarray:
        """The gradient of the objective function."""
        return self.df_port_size(x)

    def ineq_constraints(self) -> List[ConstraintT]:
        """Inequality constraints for the problem."""
        return [
            {"f_constraint": self.constrain_blanket_cut, "tolerance": np.full(3, 1e-6)}
        ]

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """The bounds for the optimisation parameters."""
        lower = [self.r_ib_min - self.c_rm, self.R_0, self.r_ib_min + self.tk_bb_ib, 0]
        upper = [
            self.R_0,
            self.r_ob_max + self.c_rm,
            self.r_ob_max - self.tk_bb_ob,
            self.bb_min_angle,
        ]
        return np.array(lower), np.array(upper)

    @staticmethod
    def port_size(x: np.ndarray) -> float:
        """Return the port size given parameterisation ``x``."""
        ri, ro, _, gamma = x
        return ro - ri + gamma

    def df_port_size(self, _: np.ndarray) -> np.ndarray:
        """
        Return the gradient of the port size.

        Parameters
        ----------
        _:
            The parameterisation of the port size. This is unused as the
            gradient is constant.

        Returns
        -------
        The gradient of the port size parameterisation, with shape (3,).
        """
        return self.gradient

    def constrain_blanket_cut(self, x: np.ndarray) -> np.ndarray:
        """
        Constrain the upper port size.

        This enforces 3 constraints:

        c1. The outboard blanket must fit through the port.
        c2. The inboard blanket must squeeze past the other blanket
            segment and out the port. Hence, this also enforces that
            the inboard blanket fits through the port.
        c3. There should be enough vertically accessible space on the
            inboard blanket to connect pipes and a remote attachment
            point.
        """
        ri, ro, ci, gamma = x
        co = self.get_outer_cut_point(ci, gamma)[0]
        c1 = (self.r_ob_max - co + self.c_rm) - (ro - co)
        c2 = (ci - self.r_ib_min) - (ro - ci + self.c_rm)
        c3 = (ri + 0.5 * abs(ci - self.r_ib_min)) - ci
        return np.array([c1, c2, c3])

    def get_outer_cut_point(self, ci: float, gamma: float):
        """
        Get the coordinate of the outer blanket cut point.

        The outer cut point radius of the cutting plane with the
        breeding blanket geometry.
        """
        intersection = get_inner_cut_point(self.bb, ci)
        x, y, z = intersection
        x2 = x - np.sin(np.deg2rad(gamma))
        y2 = y
        z2 = z + np.cos(np.deg2rad(gamma))
        angled_cut_plane = BluemiraPlane.from_3_points(
            intersection, [x2, y2, z2], [x, y + 1, z]
        )
        # Get the last intersection with the angled cut plane and the outer
        intersections = slice_shape(self.bb.boundary[0], angled_cut_plane)
        intersections = intersections[intersections[:, -1] > z + EPS]
        return min(intersections, key=lambda x: x[-1])


@dataclass
class UpperPortKOZDesignerParams(ParameterFrame):
    """Parameters required to run :class:`UpperPortKOZDesigner`."""

    tk_vv_double_wall: Parameter[float]
    """VV upper port wall thickness at radial ends [m]."""
    g_vv_ts: Parameter[float]
    """Gap between VV and TS [m]."""
    tk_ts: Parameter[float]
    """TS thickness [m]."""
    g_ts_tf: Parameter[float]
    """Gap between TS and TF (used for short gap to PF) [m]."""
    pf_s_g: Parameter[float]
    """Gap between PF coil and support [m]."""
    pf_s_tk_plate: Parameter[float]
    """PF coil support thickness [m]."""
    c_rm: Parameter[float]
    """Remote maintenance clearance [m]."""
    R_0: Parameter[float]
    """Major radius [m]."""
    bb_min_angle: Parameter[float]
    """Minimum blanket module angle [degrees]."""
    tk_bb_ib: Parameter[float]
    """Blanket inboard thickness [m]."""
    tk_bb_ob: Parameter[float]
    """Blanket outboard thickness [m]."""


class UpperPortKOZDesigner(Designer[Tuple[BluemiraFace, float, float]]):
    """Upper Port keep-out zone designer."""

    param_cls = UpperPortKOZDesignerParams
    params: UpperPortKOZDesignerParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
        build_config: Dict,
        blanket_face: BluemiraFace,
        upper_port_extrema=10,
    ):
        super().__init__(params, build_config)
        self.blanket_face = blanket_face
        self.opt_algorithm = self.build_config.get("opt_algorithm", "SLSQP")
        self.opt_conditions = {
            "max_eval": 1000,
            "ftol_rel": 1e-8,
            **self.build_config.get("opt_conditions", {}),
        }
        self.upper_port_extrema = upper_port_extrema

    def run(self):
        """Run the design problem to minimise the port size."""
        opt_problem = UpperPortOP(
            bb=self.blanket_face,
            c_rm=self.params.c_rm.value,
            R_0=self.params.R_0.value,
            tk_bb_ib=self.params.tk_bb_ib.value,
            tk_bb_ob=self.params.tk_bb_ob.value,
            bb_min_angle=(90 - self.params.bb_min_angle.value),
        )
        opt_result = opt_problem.optimise(
            # Initial guess at center of bounds
            x0=np.vstack(opt_problem.bounds()).mean(axis=0),
            algorithm=self.opt_algorithm,
            opt_conditions=self.opt_conditions,
        )
        r_up_inner, r_up_outer, r_cut, cut_angle = opt_result.x

        offset = (
            self.params.tk_vv_double_wall.value
            + self.params.g_vv_ts.value
            + self.params.tk_ts.value
            + self.params.g_ts_tf.value
            + self.params.pf_s_tk_plate.value
            + self.params.pf_s_g.value
        )
        r_up_inner -= offset
        r_up_outer += offset

        return (
            build_upper_port_zone(r_up_inner, r_up_outer, z_max=self.upper_port_extrema),
            r_cut,
            cut_angle,
        )


def build_upper_port_zone(
    r_up_inner: float, r_up_outer: float, z_max: float = 10, z_min: float = 0
) -> BluemiraFace:
    """
    Make the void geometry for the upper port in the poloidal plane.

    Parameters
    ----------
    r_up_inner:
        Inner radius of the upper port void space
    r_up_outer:
        Outer radius of the upper port void space
    z_max:
        Maximum vertical height of the upper port void space
    z_min:
        Minimum vertical height of the upper port void space

    Returns
    -------
    Face representing the upper port void space in the x-z plane
    """
    x = [r_up_inner, r_up_outer, r_up_outer, r_up_inner]
    z = [z_min, z_min, z_max, z_max]
    return BluemiraFace(make_polygon({"x": x, "y": 0, "z": z}, closed=True))
