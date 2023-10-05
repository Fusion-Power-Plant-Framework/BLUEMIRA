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
"""Designer for PF coils and its parameters."""

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Union

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.designer import Designer
from bluemira.base.look_and_feel import bluemira_print
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.file import EQDSKInterface
from bluemira.equilibria.optimisation.constraints import (
    CoilFieldConstraints,
    CoilForceConstraints,
    FieldNullConstraint,
    IsofluxConstraint,
    PsiConstraint,
)
from bluemira.equilibria.optimisation.problem import PulsedNestedPositionCOP
from bluemira.equilibria.run import (
    BreakdownCOPSettings,
    EQSettings,
    OptimisedPulsedCoilsetDesign,
    PositionSettings,
)
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.wire import BluemiraWire
from bluemira.utilities.tools import get_class_from_module, json_writer
from eudemo.equilibria.tools import make_grid
from eudemo.model_managers import EquilibriumManager
from eudemo.pf_coils.tools import make_coil_mapper, make_coilset, make_pf_coil_path


@dataclass
class PFCoilsDesignerParams(ParameterFrame):
    """Parameters for :class:`PFCoilsDesigner`."""

    # TODO(hsaunders1904): docstrings for these parameters?
    A: Parameter[float]
    B_0: Parameter[float]
    B_premag_stray_max: Parameter[float]
    beta_p: Parameter[float]
    C_Ejima: Parameter[float]
    CS_bmax: Parameter[float]
    CS_jmax: Parameter[float]
    delta: Parameter[float]
    F_cs_sepmax: Parameter[float]
    F_cs_ztotmax: Parameter[float]
    F_pf_zmax: Parameter[float]
    g_cs_mod: Parameter[float]
    I_p: Parameter[float]
    kappa: Parameter[float]
    l_i: Parameter[float]
    n_CS: Parameter[int]
    n_PF: Parameter[int]
    PF_bmax: Parameter[float]
    PF_jmax: Parameter[float]
    R_0: Parameter[float]
    r_cs_in: Parameter[float]
    tau_flattop: Parameter[float]
    tk_cs_casing: Parameter[float]
    tk_cs_insulation: Parameter[float]
    tk_pf_casing: Parameter[float]
    tk_pf_insulation: Parameter[float]
    pf_s_tk_plate: Parameter[float]
    pf_s_g: Parameter[float]
    tk_cs: Parameter[float]
    tk_sol_ib: Parameter[float]
    v_burn: Parameter[float]


class PFCoilsDesigner(Designer[CoilSet]):
    """
    Design a set of PF Coils for EUDEMO.

    Parameters
    ----------
    params:
        A `PFCoilsDesignerParams` instance, or a dictionary or other
        `ParameterFrame` that can be converted to a
        `PFCoilDesignerParams` instance.
    reference_equilibrium:
        Reference equilibrium to attempt to match during the design
    build_config:
        Build configuration dictionary for the PFCoilsDesigner
    tf_coil_boundary:
        Wire giving the outline of outer edge of the reactor's TF coils.
    keep_out_zones:
        Faces representing keep-out-zones for the PF coil geometry.
    """

    param_cls = PFCoilsDesignerParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
        build_config: Dict,
        equilibrium_manager: EquilibriumManager,
        tf_coil_boundary: BluemiraWire,
        keep_out_zones: Iterable[BluemiraFace],
    ):
        super().__init__(params, build_config)
        self.ref_eq = equilibrium_manager.get_state(equilibrium_manager.REFERENCE)
        self.tf_coil_boundary = tf_coil_boundary
        self.keep_out_zones = keep_out_zones
        self.file_path = self.build_config.get("file_path", None)
        self.eq_manager = equilibrium_manager

    def read(self) -> CoilSet:
        """Read in a coilset."""
        if self.file_path is None:
            raise ValueError("No file path to read from!")

        with open(self.file_path) as file:
            data = json.load(file)

        # TODO: Load up equilibria from files and add states to manager

        eqdsk = EQDSKInterface(**data[next(iter(data))])
        return CoilSet.from_group_vecs(eqdsk)

    def run(self) -> CoilSet:
        """Create and run the design optimisation problem.

        Create an initial coilset, grid and equilibria profile, and use
        these to solve an :class:`OptimisedPulsedCoilsetDesign` problem.
        """
        coilset = self._make_coilset()
        coil_mapper = self._make_coil_mapper(coilset)

        grid = make_grid(
            self.params.R_0.value,
            self.params.A.value,
            self.params.kappa.value,
            self.build_config.get("grid_settings", {}),
        )
        profiles = deepcopy(self.ref_eq.profiles)
        constraints = self._make_opt_constraints(coilset)
        opt_problem = self._make_pulsed_coilset_opt_problem(
            coilset, grid, profiles, coil_mapper, constraints
        )
        bluemira_print(f"Solving design problem: {opt_problem.__class__.__name__}")
        result = opt_problem.optimise(verbose=self.build_config.get("verbose", False))
        self._save_equilibria(opt_problem)
        if self.build_config.get("plot", False):
            opt_problem.plot()
            plt.show()

        return result

    def _save_equilibria(self, opt_problem):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_dict = {}
        for k, v in opt_problem.snapshots.items():
            if k in [opt_problem.SOF, opt_problem.EOF]:
                result_dict[k] = v.eq.to_dict()
                result_dict[k]["name"] = f"bluemira {timestamp} {k}"
            self.eq_manager.add_state(k, v)

        json_writer(result_dict, self.file_path)

    def _make_pulsed_coilset_opt_problem(
        self, coilset, grid, profiles, position_mapper, constraints
    ):
        breakdown_defaults = {
            "param_class": (
                "bluemira.equilibria.optimisation.problem::OutboardBreakdownZoneStrategy"
            ),
            "problem_class": "bluemira.equilibria.optimisation.problem::BreakdownCOP",
            "optimisation_settings": {
                "algorithm_name": "COBYLA",
                "conditions": {
                    "max_eval": 5000,
                    "ftol_rel": 1e-10,
                },
            },
            "B_stray_con_tol": 1e-6,
            "n_B_stray_points": 10,
        }
        breakdown_settings = {
            **breakdown_defaults,
            **self.build_config.get("breakdown_settings", {}),
        }

        eq_defaults = {
            "problem_class": (
                "bluemira.equilibria.optimisation.problem::TikhonovCurrentCOP"
            ),
            "convergence_class": "bluemira.equilibria.solve::DudsonConvergence",
            "conv_limit": 1e-4,
            "gamma": 1e-12,
            "relaxation": 0.2,
            "peak_PF_current_factor": 1.5,
            "optimisation_settings": {
                "algorithm_name": "SLSQP",
                "conditions": {
                    "max_eval": 5000,
                    "ftol_rel": 1e-6,
                },
            },
        }
        eq_settings = {
            **eq_defaults,
            **self.build_config.get("equilibrium_settings", {}),
        }
        eq_converger = get_class_from_module(eq_settings["convergence_class"])

        pos_defaults = {
            "optimisation_settings": {
                "algorithm_name": "COBYLA",
                "conditions": {
                    "max_eval": 200,
                    "ftol_rel": 1e-6,
                    "xtol_rel": 1e-6,
                },
            },
        }
        pos_settings = {**pos_defaults, **self.build_config.get("position_settings", {})}

        return OptimisedPulsedCoilsetDesign(
            self.params,
            coilset,
            position_mapper,
            grid,
            current_opt_constraints=[constraints["psi_inner"]],
            coil_constraints=constraints["coil_field"],
            equilibrium_constraints=[constraints["isoflux"], constraints["x_point"]],
            profiles=profiles,
            breakdown_settings=BreakdownCOPSettings(
                strategy=get_class_from_module(breakdown_settings["param_class"]),
                problem=get_class_from_module(breakdown_settings["problem_class"]),
                algorithm=breakdown_settings["optimisation_settings"]["algorithm_name"],
                opt_conditions=breakdown_settings["optimisation_settings"]["conditions"],
                B_stray_con_tol=breakdown_settings["B_stray_con_tol"],
                n_B_stray_points=breakdown_settings["n_B_stray_points"],
            ),
            equilibrium_settings=EQSettings(
                problem=get_class_from_module(eq_settings["problem_class"]),
                convergence=eq_converger(eq_settings["conv_limit"]),
                algorithm=eq_settings["optimisation_settings"]["algorithm_name"],
                opt_conditions=eq_settings["optimisation_settings"]["conditions"],
                gamma=eq_settings["gamma"],
                relaxation=eq_settings["relaxation"],
                peak_PF_current_factor=eq_settings["peak_PF_current_factor"],
            ),
            position_settings=PositionSettings(
                problem=PulsedNestedPositionCOP,
                algorithm=pos_settings["optimisation_settings"]["algorithm_name"],
                opt_conditions=pos_settings["optimisation_settings"]["conditions"],
            ),
            limiter=None,
        )

    def _make_opt_constraints(self, coilset):
        # TODO: Make LCFS constraints from fixed boundary k_95 / d_95 optimisation
        kappa = self.params.kappa.value
        kappa_ul_tweak = 0.085
        kappa_u = (1 - kappa_ul_tweak) * kappa
        kappa_l = (1 + kappa_ul_tweak) * kappa
        lcfs_parameterisation = JohnerLCFS(
            {
                "r_0": {"value": self.params.R_0.value},
                "z_0": {"value": 0.0},
                "a": {"value": self.params.R_0.value / self.params.A.value},
                "kappa_u": {"value": kappa_u},
                "kappa_l": {"value": kappa_l},
                "delta_u": {"value": self.params.delta.value},
                "delta_l": {"value": self.params.delta.value},
                "phi_u_neg": {"value": 0.0},
                "phi_u_pos": {"value": 0.0},
                "phi_l_neg": {"value": 45.0},
                "phi_l_pos": {"value": 30.0},
            }
        )
        lcfs = lcfs_parameterisation.create_shape().discretize(byedges=True, ndiscr=50)
        x_lcfs, z_lcfs = lcfs.x, lcfs.z
        arg_inner = np.argmin(x_lcfs)
        arg_xp = np.argmin(z_lcfs)

        isoflux = IsofluxConstraint(
            x_lcfs,
            z_lcfs,
            x_lcfs[arg_inner],
            z_lcfs[arg_inner],
            tolerance=1.0,
            constraint_value=0.0,
        )
        psi_inner = PsiConstraint(
            x_lcfs[arg_inner], z_lcfs[arg_inner], target_value=0.0, tolerance=1e-3
        )
        x_point = FieldNullConstraint(x_lcfs[arg_xp], z_lcfs[arg_xp], tolerance=1e-4)
        coil_field_constraints = [
            CoilFieldConstraints(coilset, coilset.b_max, tolerance=1e-6),
            CoilForceConstraints(
                coilset,
                self.params.F_pf_zmax.value,
                self.params.F_cs_ztotmax.value,
                self.params.F_cs_sepmax.value,
                tolerance=1e-3,
            ),
        ]
        return {
            "isoflux": isoflux,
            "psi_inner": psi_inner,
            "x_point": x_point,
            "coil_field": coil_field_constraints,
        }

    def _make_coil_mapper(self, coilset):
        # Get an offset from the TF that corresponds to a PF coil half-width of a
        # current equal to Ip
        # TODO may need to thread this better!
        peak_PF_current_factor = 1.5
        offset_value = 0.5 * np.sqrt(
            peak_PF_current_factor * self.params.I_p.value / self.params.PF_jmax.value
        )
        offset_value += np.sqrt(2) * (
            self.params.tk_pf_casing.value
            + self.params.tk_pf_insulation.value
            + self.params.pf_s_g.value
            + self.params.pf_s_tk_plate.value
        )
        pf_coil_path = make_pf_coil_path(self.tf_coil_boundary, offset_value)
        pf_coils = coilset.get_coiltype("PF")._coils
        return make_coil_mapper(pf_coil_path, self.keep_out_zones, pf_coils)

    def _make_coilset(self):
        return make_coilset(
            tf_boundary=self.tf_coil_boundary,
            R_0=self.params.R_0.value,
            kappa=self.params.kappa.value,
            delta=self.params.delta.value,
            r_cs=self.params.r_cs_in.value + 0.5 * self.params.tk_cs.value,
            tk_cs=0.5 * self.params.tk_cs.value,
            g_cs=self.params.g_cs_mod.value,
            tk_cs_ins=self.params.tk_cs_insulation.value,
            tk_cs_cas=self.params.tk_cs_casing.value,
            n_CS=self.params.n_CS.value,
            n_PF=self.params.n_PF.value,
            CS_jmax=self.params.CS_jmax.value,
            CS_bmax=self.params.CS_bmax.value,
            PF_jmax=self.params.PF_jmax.value,
            PF_bmax=self.params.PF_bmax.value,
        )
