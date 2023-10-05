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
First Wall Silhouette designer
"""
from dataclasses import dataclass
from typing import Dict, Type, Union

from bluemira.base.designer import Designer
from bluemira.base.error import DesignError
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.equilibria import Equilibrium
from bluemira.equilibria.find import find_OX_points, get_legs
from bluemira.geometry.optimisation import KeepOutZone, optimise_geometry
from bluemira.geometry.parameterisations import GeometryParameterisation, PolySpline
from bluemira.geometry.tools import convex_hull_wires_2d, make_polygon, offset_wire
from bluemira.geometry.wire import BluemiraWire
from bluemira.utilities.tools import get_class_from_module


@dataclass
class WallSilhouetteDesignerParams(ParameterFrame):
    """Parameters for running the `WallSilhouetteDesigner`."""

    R_0: Parameter[float]  # major radius
    kappa_95: Parameter[float]  # 95th percentile plasma elongation
    r_fw_ib_in: Parameter[float]  # inboard first wall inner radius
    r_fw_ob_in: Parameter[float]  # inboard first wall outer radius
    A: Parameter[float]  # aspect ratio
    tk_sol_ib: Parameter[float]
    fw_psi_n: Parameter[float]
    div_L2D_ib: Parameter[float]
    div_L2D_ob: Parameter[float]


class WallSilhouetteDesigner(Designer[GeometryParameterisation]):
    """
    Designs the first wall silhouette to inform the divertor and IVC design areas

    Parameters
    ----------
    params:
        Wall silhouette designer parameters
    build_config:
        configuration of the design
    equilibrium:
        The equilibrium to design around

    """

    param_cls = WallSilhouetteDesignerParams

    def __init__(
        self,
        params: Union[ParameterFrame, Dict],
        build_config: Dict,
        equilibrium: Equilibrium,
    ) -> None:
        super().__init__(params, build_config)

        self.parameterisation_cls: Type[
            GeometryParameterisation
        ] = get_class_from_module(
            self.build_config["param_class"],
            default_module="bluemira.geometry.parameterisations",
        )

        self.variables_map = self.build_config.get("variables_map", {})
        self.file_path = self.build_config.get("file_path", None)

        self.problem_settings = self.build_config.get("problem_settings", {})
        self.opt_config = self.build_config.get("optimisation_settings", {})
        self.algorithm_name = self.opt_config.get("algorithm_name", "SLSQP")
        self.opt_conditions = self.opt_config.get("conditions", {"max_eval": 100})
        self.opt_parameters = self.opt_config.get("parameters", {})

        self.equilibrium = equilibrium

    def execute(self) -> GeometryParameterisation:
        """
        Execute method of WallSilhouetteDesigner
        """
        result = super().execute()

        # TODO move to plasma component manager
        _, x_points = find_OX_points(
            self.equilibrium.x, self.equilibrium.z, self.equilibrium.psi()
        )

        if result.create_shape().bounding_box.z_min >= x_points[0].z:
            raise DesignError("First wall boundary does not enclose separatrix x-point.")
        return result

    def mock(self) -> GeometryParameterisation:
        """
        Mock method of WallSilhouetteDesigner
        """
        return self._get_parameterisation()

    def read(self) -> GeometryParameterisation:
        """
        Read method of WallSilhouetteDesigner
        """
        if not self.file_path:
            raise ValueError(
                f"Cannot execute {type(self).__name__} in 'read' mode: no file path"
                " specified."
            )
        return self.parameterisation_cls.from_json(file=self.file_path)

    def run(self) -> GeometryParameterisation:
        """
        Optimise the shape using the provided parameterisation and optimiser.
        """
        parameterisation = self._get_parameterisation()

        def f_objective(geom: GeometryParameterisation) -> float:
            """Objective function to minimise a shape's length."""
            return geom.create_shape().length

        bluemira_print("Solving WallSilhouette optimisation")
        bluemira_debug(
            "Setting up design problem with:\n"
            f"algorithm_name: {self.algorithm_name}\n"
            f"n_variables: {parameterisation.variables.n_free_variables}\n"
            f"opt_conditions: {self.opt_conditions}\n"
            f"opt_parameters: {self.opt_parameters}"
        )

        if self.problem_settings != {}:
            bluemira_debug(
                f"Applying non-default settings to problem: {self.problem_settings}"
            )
        result = optimise_geometry(
            parameterisation,
            algorithm=self.algorithm_name,
            f_objective=f_objective,
            opt_conditions=self.opt_conditions,
            opt_parameters=self.opt_parameters,
            keep_out_zones=[
                KeepOutZone(
                    self._make_wall_keep_out_zone(),
                    n_discr=self.problem_settings.get("n_koz_points", 100),
                )
            ],
        )
        return result.geom

    def _get_parameterisation(self) -> GeometryParameterisation:
        return self.parameterisation_cls(self._derive_shape_params())

    def _derive_shape_params(self) -> Dict:
        shape_params = {}
        for key, val in self.variables_map.items():
            if isinstance(val, str):
                val = getattr(self.params, val).value  # noqa: PLW2901

            if isinstance(val, dict):
                if isinstance(val["value"], str):
                    val["value"] = getattr(self.params, val["value"]).value
            else:
                val = {"value": val}  # noqa: PLW2901

            shape_params[key] = val

        if issubclass(self.parameterisation_cls, PolySpline):
            height_value = self._derive_polyspline_height()
            shape_params["height"] = {
                "value": height_value,
                "lower_bound": 0.8 * height_value,
                "upper_bound": 2.0 * height_value,
            }

        return shape_params

    def _derive_polyspline_height(self) -> float:
        """Derive the PolySpline height from relevant parameters."""
        r_minor = self.params.R_0.value / self.params.A.value
        return (self.params.kappa_95.value * r_minor) * 2

    def _make_wall_keep_out_zone(self) -> BluemiraWire:
        """
        Create a "keep-out zone" to be used as a constraint in the
        wall shape optimiser.
        """
        geom_offset = self.params.tk_sol_ib.value
        psi_n = self.params.fw_psi_n.value
        geom_offset = 0.2  # TODO: Unpin
        psi_n = 1.05  # TODO: Unpin
        geom_offset_zone = self._make_geometric_keep_out_zone(geom_offset)
        flux_surface_zone = self._make_flux_surface_keep_out_zone(psi_n)
        leg_zone = self._make_divertor_leg_keep_out_zone(
            self.params.div_L2D_ib.value, self.params.div_L2D_ob.value
        )
        return convex_hull_wires_2d(
            [geom_offset_zone, flux_surface_zone, leg_zone], ndiscr=200, plane="xz"
        )

    def _make_geometric_keep_out_zone(self, offset: float) -> BluemiraWire:
        """
        Make a "keep-out zone" from a geometric offset of the LCFS.
        """
        lcfs = make_polygon(self.equilibrium.get_LCFS().xyz, closed=True)
        return offset_wire(lcfs, offset, join="arc")

    def _make_flux_surface_keep_out_zone(self, psi_n: float) -> BluemiraWire:
        """
        Make a "keep-out zone" from an equilibrium's flux surface.
        """
        # TODO: This is currently called three times once here, once above
        # and once for setup of the remaining ivc
        o_points, _ = find_OX_points(
            self.equilibrium.x, self.equilibrium.z, self.equilibrium.psi()
        )
        flux_surface_zone = self.equilibrium.get_flux_surface(psi_n)
        # Chop the flux surface to only take the upper half
        indices = flux_surface_zone.z >= o_points[0][1]
        return make_polygon(flux_surface_zone.xyz[:, indices], closed=True)

    def _make_divertor_leg_keep_out_zone(
        self, leg_length_ib_2D, leg_length_ob_2D
    ) -> BluemiraWire:
        """
        Make a "keep-out zone" from an equilibrium's divertor legs
        """
        # TODO move to plasma component manager
        legs = get_legs(self.equilibrium, n_layers=1, dx_off=0.0)

        ib_leg = make_polygon(legs["lower_inner"][0].xyz)
        ob_leg = make_polygon(legs["lower_outer"][0].xyz)

        return make_polygon(
            [
                ib_leg.value_at(distance=leg_length_ib_2D),
                ob_leg.value_at(distance=leg_length_ob_2D),
            ],
            closed=False,
        )
