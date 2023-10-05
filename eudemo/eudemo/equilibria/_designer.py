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
Designer for an `Equilibrium` solving an unconstrained Tikhnov current
gradient coil-set optimisation problem.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.designer import Designer
from bluemira.base.file import get_bluemira_path, get_bluemira_root
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.codes.plasmod.equilibrium_2d_coupling import solve_transport_fixed_boundary
from bluemira.codes.wrapper import transport_code_solver
from bluemira.equilibria import Equilibrium
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemGradShafranovFixedBoundary,
    FixedBoundaryEquilibrium,
)
from bluemira.equilibria.fem_fixed_boundary.file import save_fixed_boundary_to_file
from bluemira.equilibria.fem_fixed_boundary.utilities import get_mesh_boundary
from bluemira.equilibria.file import EQDSKInterface
from bluemira.equilibria.optimisation.problem import (
    UnconstrainedTikhonovCurrentGradientCOP,
)
from bluemira.equilibria.profiles import BetaLiIpProfile, CustomProfile, Profile
from bluemira.equilibria.solve import DudsonConvergence, PicardIterator
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.parameterisations import GeometryParameterisation, PrincetonD
from bluemira.geometry.tools import make_circle, make_polygon, offset_wire
from bluemira.geometry.wire import BluemiraWire
from bluemira.utilities.tools import get_class_from_module
from eudemo.equilibria._equilibrium import (
    EquilibriumParams,
    ReferenceEquilibriumParams,
    make_equilibrium,
    make_reference_equilibrium,
)
from eudemo.equilibria.tools import (
    EUDEMOSingleNullConstraints,
    ReferenceConstraints,
    handle_lcfs_shape_input,
)


@dataclass
class EquilibriumDesignerParams(ParameterFrame):
    """Parameters for running the `UnconstrainedTikhonovSolver`."""

    A: Parameter[float]
    B_0: Parameter[float]
    beta_p: Parameter[float]
    CS_bmax: Parameter[float]
    CS_jmax: Parameter[float]
    delta: Parameter[float]
    delta_95: Parameter[float]
    div_L2D_ib: Parameter[float]
    div_L2D_ob: Parameter[float]
    g_cs_mod: Parameter[float]
    I_p: Parameter[float]
    kappa: Parameter[float]
    kappa_95: Parameter[float]
    l_i: Parameter[float]
    n_CS: Parameter[int]
    n_PF: Parameter[int]
    PF_bmax: Parameter[float]
    PF_jmax: Parameter[float]
    q_95: Parameter[float]
    R_0: Parameter[float]
    r_cs_in: Parameter[float]
    r_tf_in_centre: Parameter[float]
    r_tf_out_centre: Parameter[float]
    shaf_shift: Parameter[float]
    tk_cs_casing: Parameter[float]
    tk_cs_insulation: Parameter[float]
    tk_cs: Parameter[float]


class EquilibriumDesigner(Designer[Equilibrium]):
    """
    Solves an unconstrained Tikhnov current gradient coil-set
    optimisation problem, outputting an `Equilibrium`.

    Parameters
    ----------
    params:
        The parameters for the solver, the dictionary or frame must
        contain all the parameters present in
        `UnconstrainedTikhonovSolverParams`.
    build_config:
        The config for the solver. Optional keys:
        - `read_file_path`: str
            the path to an eqdsk file to read the equilibrium from,
            required in `read` mode.
        - `plot_optimisation`: bool
            set to `True` to plot the iterations in the optimisation,
            only used in `run` mode
    """

    params: EquilibriumDesignerParams
    param_cls = EquilibriumDesignerParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
        build_config: Optional[Dict] = None,
    ):
        super().__init__(params, build_config)
        self.file_path = self.build_config.get("file_path", None)
        self.plot_optimisation = self.build_config.get("plot_optimisation", False)
        if self.run_mode == "read" and self.file_path is None:
            raise ValueError(
                f"Cannot execute {type(self).__name__} in 'read' mode: "
                "'file_path' missing from build config."
            )

    def run(self) -> Equilibrium:
        """Run the designer's optimisation problem."""
        eq = self._make_equilibrium()
        opt_problem = self._make_opt_problem(eq)
        iterator_program = PicardIterator(
            eq,
            opt_problem,
            convergence=DudsonConvergence(),
            relaxation=0.2,
            fixed_coils=True,
            plot=self.plot_optimisation,
        )
        self._result = iterator_program()
        self._update_params_from_eq(eq)
        return eq

    def read(self) -> Equilibrium:
        """Load an equilibrium from a file."""
        eq = Equilibrium.from_eqdsk(self.file_path)
        self._update_params_from_eq(eq)
        return eq

    def _update_params_from_eq(self, eq: Equilibrium):
        plasma_dict = eq.analyse_plasma()
        new_values = {
            "beta_p": plasma_dict["beta_p"],
            "delta_95": plasma_dict["delta_95"],
            "delta": plasma_dict["delta"],
            "I_p": plasma_dict["Ip"],
            "kappa_95": plasma_dict["kappa_95"],
            "kappa": plasma_dict["kappa"],
            "l_i": plasma_dict["li"],
            "q_95": plasma_dict["q_95"],
            "shaf_shift": np.hypot(plasma_dict["dx_shaf"], plasma_dict["dz_shaf"]),
        }
        self.params.update_values(new_values, source=type(self).__name__)

    def _make_equilibrium(self) -> Equilibrium:
        """
        Make a reference MHD equilibrium for the plasma.
        """
        return make_equilibrium(
            EquilibriumParams.from_frame(self.params),
            _make_tf_boundary(
                self.params.r_tf_in_centre.value,
                self.params.r_tf_out_centre.value,
                self.params.delta_95.value,
            ),
            self.build_config.get("grid_settings", {}),
        )

    def _make_opt_problem(self, eq: Equilibrium):
        """
        Create the `UnconstrainedTikhonovCurrentGradientCOP` optimisation problem.
        """
        kappa = 1.12 * self.params.kappa_95.value
        kappa_ul_tweak = 0.05
        kappa_u = (1 - kappa_ul_tweak) * kappa
        kappa_l = (1 + kappa_ul_tweak) * kappa

        eq_targets = EUDEMOSingleNullConstraints(
            R_0=self.params.R_0.value,
            Z_0=0.0,
            A=self.params.A.value,
            kappa_u=kappa_u,
            kappa_l=kappa_l,
            delta_u=self.params.delta_95.value,
            delta_l=self.params.delta_95.value,
            psi_u_neg=0.0,
            psi_u_pos=0.0,
            psi_l_neg=60.0,
            psi_l_pos=30.0,
            div_l_ib=self.params.div_L2D_ib.value,
            div_l_ob=self.params.div_L2D_ob.value,
            psibval=0.0,
            psibtol=1.0e-3,
            lower=True,
            n=100,
        )
        return UnconstrainedTikhonovCurrentGradientCOP(
            eq.coilset, eq, eq_targets, gamma=1e-8
        )


def _make_tf_boundary(
    r_tf_in_centre: float, r_tf_out_centre: float, delta_95: float
) -> BluemiraWire:
    """
    Make an initial TF coil shape to guide an equilibrium calculation.
    """
    rin, rout = r_tf_in_centre, r_tf_out_centre
    # TODO: Handle other TF coil parameterisations?
    shape = PrincetonD({"x1": {"value": rin}, "x2": {"value": rout}, "dz": {"value": 0}})
    tf_boundary = shape.create_shape()
    if delta_95 < 0:  # Negative triangularity
        tf_boundary.rotate(tf_boundary.center_of_mass, direction=(0, 1, 0), degree=180)
    tf_boundary = offset_wire(tf_boundary, -0.5)
    x, z = _flatten_shape(*tf_boundary.discretize(200, byedges=True).xz)
    return make_polygon({"x": x, "z": z})


def _flatten_shape(x, z):
    """
    Flattens a shape by dragging the lowest and highest point to the minimum
    radius point.
    """
    amin, amax = np.argmin(z), np.argmax(z)
    num_elements = amax - amin + 2

    xx = np.empty(num_elements)
    xx[0] = np.min(x)
    xx[1:-1] = x[amin:amax]
    xx[-1] = xx[0]

    zmin, zmax = z[amin], z[amax]
    zz = np.empty(num_elements)
    zz[0] = zmin
    zz[1:-1] = z[amin:amax]
    zz[-1] = zmax

    return xx, zz


def get_plasmod_binary_path():
    """
    Get the path to the PLASMOD binary.
    """
    if plasmod_binary := shutil.which("plasmod"):
        PLASMOD_PATH = Path(plasmod_binary).parent
    else:
        PLASMOD_PATH = Path(Path(get_bluemira_root()).parent, "plasmod/bin")
    return Path(PLASMOD_PATH, "plasmod").as_posix()


@dataclass
class FixedEquilibriumDesignerParams(ParameterFrame):
    """Parameters for running the fixed boundary equilibrium solver."""

    A: Parameter[float]
    B_0: Parameter[float]
    delta: Parameter[float]
    delta_95: Parameter[float]
    I_p: Parameter[float]
    kappa: Parameter[float]
    kappa_95: Parameter[float]
    q_95: Parameter[float]
    R_0: Parameter[float]
    r_cs_in: Parameter[float]
    tk_cs: Parameter[float]
    v_burn: Parameter[float]
    P_fus: Parameter[float]

    # PLASMOD parameters
    q_control: Parameter[float]
    e_nbi: Parameter[float]
    f_ni: Parameter[float]
    T_e_ped: Parameter[float]


class FixedEquilibriumDesigner(Designer[Tuple[Coordinates, CustomProfile]]):
    """
    Solves a transport <-> fixed boundary equilibrium problem to convergence,
    returning a `FixedBoundaryEquilibrium`.

    Parameters
    ----------
    params:
        The parameters for the solver
    build_config:
        The config for the solver.
    """

    params: FixedEquilibriumDesignerParams
    param_cls = FixedEquilibriumDesignerParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
        build_config: Optional[Dict] = None,
    ):
        super().__init__(params, build_config)
        self.file_path = self.build_config.get("file_path", None)
        if self.run_mode == "read" and self.file_path is None:
            raise ValueError(
                f"Cannot execute {type(self).__name__} in 'read' mode: "
                "'file_path' missing from build config."
            )

    def run(self) -> Tuple[Coordinates, CustomProfile]:
        """
        Run the FixedEquilibriumDesigner.
        """
        # Get geometry parameterisation
        geom_parameterisation = self._get_geometry_parameterisation()

        # Get PLASMOD solver
        transport_solver = self._get_transport_solver()

        # Get fixed boundary equilibrium solver
        fem_fixed_be_solver = self._get_fixed_equilibrium_solver()

        # Solve converged transport - fixed boundary equilibrium
        defaults = {
            "lcar_mesh": 0.3,
            "max_iter": 15,
            "iter_err_max": 1e-3,
            "relaxation": 0.0,
            "plot": False,
        }
        settings = self.build_config.get("transport_eq_settings", {})
        settings = {**defaults, **settings}
        fixed_equilibrium: FixedBoundaryEquilibrium = solve_transport_fixed_boundary(
            geom_parameterisation,
            transport_solver,
            fem_fixed_be_solver,
            kappa95_t=self.params.kappa_95.value,  # Target kappa_95
            delta95_t=self.params.delta_95.value,  # Target delta_95
            **settings,
        )
        if self.file_path is not None:
            save_fixed_boundary_to_file(
                self.file_path,
                f"Transport-fixed-boundary-solve {fem_fixed_be_solver.iter_err_max:.3e}",
                fixed_equilibrium,
                65,
                127,
            )

        xbdry, zbdry = get_mesh_boundary(fixed_equilibrium.mesh)
        lcfs_coords = Coordinates({"x": xbdry, "y": 0, "z": zbdry})
        lcfs_coords.close()
        profiles = CustomProfile(
            fixed_equilibrium.pprime,
            fixed_equilibrium.ffprime,
            R_0=fixed_equilibrium.R_0,
            B_0=fixed_equilibrium.B_0,
            I_p=fixed_equilibrium.I_p,
        )
        return lcfs_coords, profiles

    def read(self) -> Tuple[Coordinates, CustomProfile]:
        """
        Read in a fixed boundary equilibrium
        """
        data = EQDSKInterface.from_file(self.file_path)
        lcfs_coords = Coordinates({"x": data.xbdry, "y": 0, "z": data.zbdry})
        lcfs_coords.close()

        profiles = CustomProfile(
            data.pprime,
            data.ffprime,
            R_0=data.xcentre,
            B_0=data.bcentre,
            I_p=data.cplasma,
        )
        return lcfs_coords, profiles

    def _get_geometry_parameterisation(self):
        param_cls: Type[GeometryParameterisation] = get_class_from_module(
            self.build_config["param_class"], default_module="bluemira.equilibria.shapes"
        )
        shape_config = self.build_config.get("shape_config", {})
        input_dict = handle_lcfs_shape_input(param_cls, self.params, shape_config)
        return param_cls(input_dict)

    def _get_transport_solver(self):
        defaults = {
            "i_impmodel": "PED_FIXED",
            "i_modeltype": "GYROBOHM_2",
            "i_equiltype": "q95_sawtooth",
            "i_pedestal": "SAARELMA",
            "isawt": "FULLY_RELAXED",
        }
        problem_settings = self.build_config.get("plasmod_settings", defaults)
        problem_settings["amin"] = self.params.R_0.value / self.params.A.value
        problem_settings["pfus_req"] = (
            self.params.P_fus.value / 1e6
        )  # TODO: Move into PLASMOD params
        problem_settings["q_control"] = (
            self.params.q_control.value / 1e6
        )  # TODO: Move into PLASMOD params
        problem_settings["volume_in"] = -2500.0
        problem_settings["v_loop"] = -1.0e-6

        plasmod_build_config = {
            "problem_settings": problem_settings,
            "mode": "run",
            "binary": get_plasmod_binary_path(),
            "directory": get_bluemira_path("", subfolder="generated_data"),
        }

        return transport_code_solver(
            params=self.params, build_config=plasmod_build_config, module="PLASMOD"
        )

    def _get_fixed_equilibrium_solver(self):
        eq_settings = self.build_config.get("fixed_equilibrium_settings", {})
        defaults = {
            "p_order": 2,
            "max_iter": 30,
            "iter_err_max": 1e-4,
            "relaxation": 0.05,
        }
        eq_settings = {**defaults, **eq_settings}
        return FemGradShafranovFixedBoundary(**eq_settings)


@dataclass
class DummyFixedEquilibriumDesignerParams(ParameterFrame):
    """
    Parameter frame for the dummy equilibrium designer
    """

    R_0: Parameter[float]
    B_0: Parameter[float]
    I_p: Parameter[float]
    l_i: Parameter[float]
    beta_p: Parameter[float]
    A: Parameter[float]
    delta: Parameter[float]
    delta_95: Parameter[float]
    kappa: Parameter[float]
    kappa_95: Parameter[float]


class DummyFixedEquilibriumDesigner(Designer[Tuple[Coordinates, Profile]]):
    """
    Dummy equilibrium designer that produces a LCFS shape and a profile
    object to be used in later reference free boundary equilibrium
    designers.
    """

    params: DummyFixedEquilibriumDesignerParams
    param_cls = DummyFixedEquilibriumDesignerParams

    def __init__(self, params, build_config):
        super().__init__(params, build_config)
        if self.build_config["run_mode"] != "run":
            bluemira_warn(
                f"This designer {type(self).__name__} can only be run in run mode."
            )
            self.build_config["run_mode"] = "run"

    def run(self) -> Tuple[Coordinates, Profile]:
        """
        Run the DummyFixedEquilibriumDesigner.
        """
        param_cls = self.build_config.get(
            "param_class", "bluemira.equilibria.shapes.JohnerLCFS"
        )
        param_cls = get_class_from_module(param_cls)
        shape_config = self.build_config.get("shape_config", {})
        input_dict = handle_lcfs_shape_input(param_cls, self.params, shape_config)
        lcfs_parameterisation = param_cls(input_dict)

        default_settings = {
            "n_points": 200,
            "li_rel_tol": 0.01,
            "li_min_iter": 2,
        }
        settings = self.build_config.get("settings", {})
        settings = {**default_settings, **settings}
        lcfs_coords = lcfs_parameterisation.create_shape().discretize(
            byedges=True, ndiscr=settings["n_points"]
        )

        profiles = BetaLiIpProfile(
            self.params.beta_p.value,
            self.params.l_i.value,
            self.params.I_p.value,
            R_0=self.params.R_0.value,
            B_0=self.params.B_0.value,
            li_rel_tol=settings["li_rel_tol"],
            li_min_iter=settings["li_min_iter"],
        )
        return lcfs_coords, profiles


@dataclass
class ReferenceFreeBoundaryEquilibriumDesignerParams(ParameterFrame):
    """Parameters for running the fixed boundary equilibrium solver."""

    A: Parameter[float]
    B_0: Parameter[float]
    I_p: Parameter[float]
    kappa: Parameter[float]
    R_0: Parameter[float]
    r_cs_in: Parameter[float]
    g_cs_mod: Parameter[float]
    tk_cs_casing: Parameter[float]
    tk_cs_insulation: Parameter[float]

    tk_cs: Parameter[float]
    tk_bb_ob: Parameter[float]
    tk_vv_out: Parameter[float]

    n_CS: Parameter[int]
    n_PF: Parameter[int]

    # Updated parameters
    delta_95: Parameter[float]
    delta: Parameter[float]
    kappa_95: Parameter[float]
    q_95: Parameter[float]
    beta_p: Parameter[float]
    l_i: Parameter[float]
    shaf_shift: Parameter[float]


class ReferenceFreeBoundaryEquilibriumDesigner(Designer[Equilibrium]):
    """
    Solves a free boundary equilibrium from a LCFS shape and profiles.

    Some coils are positioned at sensible locations to try and get an initial
    free boundary equilibrium in order to be able to draw an initial first wall
    shape.

    Parameters
    ----------
    params:
        The parameters for the solver
    build_config:
        The config for the solver.
    lcfs_coords:
        Coordinates for the desired LCFS shape
    profiles:
        Profile object describing the equilibrium profiles
    """

    params: ReferenceFreeBoundaryEquilibriumDesignerParams
    param_cls = ReferenceFreeBoundaryEquilibriumDesignerParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
        build_config: Optional[Dict] = None,
        lcfs_coords: Optional[Coordinates] = None,
        profiles: Optional[Profile] = None,
    ):
        super().__init__(params, build_config)
        self.file_path = self.build_config.get("file_path", None)
        self.lcfs_coords = lcfs_coords
        self.profiles = profiles

        if self.run_mode == "read" and self.file_path is None:
            raise ValueError(
                f"Cannot execute {type(self).__name__} in 'read' mode: "
                "'file_path' missing from build config."
            )
        self.opt_problem = None

        if self.run_mode == "run" and (
            (self.lcfs_coords is None) or (self.profiles is None)
        ):
            raise ValueError(
                f"Cannot execute {type(self).__name__} in 'run' mode without "
                "input LCFS shape or profiles."
            )

    def run(self) -> Equilibrium:
        """
        Run the FreeBoundaryEquilibriumFromFixedDesigner.
        """
        lcfs_shape = make_polygon(self.lcfs_coords, closed=True)

        # Make dummy tf coil boundary
        tf_coil_boundary = self._make_tf_boundary(lcfs_shape)

        defaults = {
            "relaxation": 0.02,
            "coil_discretisation": 0.3,
            "gamma": 1e-8,
            "iter_err_max": 1e-2,
            "max_iter": 30,
        }
        settings = self.build_config.get("settings", {})
        settings = {**defaults, **settings}

        eq = make_reference_equilibrium(
            ReferenceEquilibriumParams.from_frame(self.params),
            tf_coil_boundary,
            lcfs_shape,
            self.profiles,
            self.build_config.get("grid_settings", {}),
        )
        # TODO: Check coil discretisation is sensible when size not set...
        discretisation = settings.pop("coil_discretisation")
        # eq.coilset.discretisation = settings.pop("coil_discretisation")
        eq.coilset.get_coiltype("CS").discretisation = discretisation

        self.opt_problem = self._make_fbe_opt_problem(
            eq, lcfs_shape, len(self.lcfs_coords.x), settings.pop("gamma")
        )

        iter_err_max = settings.pop("iter_err_max")
        max_iter = settings.pop("max_iter")
        settings["maxiter"] = max_iter  # TODO: Standardise name in PicardIterator
        iterator_program = PicardIterator(
            eq,
            self.opt_problem,
            convergence=DudsonConvergence(iter_err_max),
            plot=self.build_config.get("plot", False),
            fixed_coils=True,
            **settings,
        )
        self._result = iterator_program()

        if self.build_config.get("plot", False):
            _, ax = plt.subplots()
            eq.plot(ax=ax)
            eq.coilset.plot(ax=ax, label=True)
            ax.plot(self.lcfs_coords.x, self.lcfs_coords.z, "", marker="o")
            self.opt_problem.targets.plot(ax=ax)
            plt.show()

        self._update_params_from_eq(eq)

        return eq

    def read(self) -> Equilibrium:
        """Load an equilibrium from a file."""
        eq = Equilibrium.from_eqdsk(self.file_path)
        self._update_params_from_eq(eq)
        return eq

    def _make_tf_boundary(
        self,
        lcfs_shape: BluemiraWire,
    ) -> BluemiraWire:
        coords = lcfs_shape.discretize(byedges=True, ndiscr=200)
        xu_arg = np.argmax(coords.z)
        xl_arg = np.argmin(coords.z)
        xz_min, z_min = coords.x[xl_arg], coords.z[xl_arg]
        xz_max, z_max = coords.x[xu_arg], coords.z[xu_arg]
        x_circ = min(xz_min, xz_max)
        z_circ = z_max - abs(z_min)
        r_circ = 0.5 * (z_max + abs(z_min))

        offset_value = self.params.tk_bb_ob.value + self.params.tk_vv_out.value + 2.5
        semi_circle = make_circle(
            r_circ + offset_value,
            center=(x_circ, 0, z_circ),
            start_angle=-90,
            end_angle=90,
            axis=(0, 1, 0),
        )

        xs, zs = semi_circle.start_point().xz.T[0]
        xe, ze = semi_circle.end_point().xz.T[0]
        r_cs_out = self.params.r_cs_in.value + self.params.tk_cs.value

        lower_wire = make_polygon({"x": [r_cs_out, xs], "y": [0, 0], "z": [zs, zs]})
        upper_wire = make_polygon({"x": [xe, r_cs_out], "y": [0, 0], "z": [ze, ze]})

        return BluemiraWire([lower_wire, semi_circle, upper_wire])

    @staticmethod
    def _make_fbe_opt_problem(
        eq: Equilibrium, lcfs_shape: BluemiraWire, n_points: int, gamma: float
    ):
        """
        Create the `UnconstrainedTikhonovCurrentGradientCOP` optimisation problem.
        """
        eq_targets = ReferenceConstraints(lcfs_shape, n_points)
        return UnconstrainedTikhonovCurrentGradientCOP(
            eq.coilset, eq, eq_targets, gamma=gamma
        )

    def _update_params_from_eq(self, eq: Equilibrium):
        plasma_dict = eq.analyse_plasma()
        new_values = {
            "beta_p": plasma_dict["beta_p"],
            "delta_95": plasma_dict["delta_95"],
            "delta": plasma_dict["delta"],
            "I_p": plasma_dict["Ip"],
            "kappa_95": plasma_dict["kappa_95"],
            "kappa": plasma_dict["kappa"],
            "l_i": plasma_dict["li"],
            "q_95": plasma_dict["q_95"],
            "shaf_shift": np.hypot(plasma_dict["dx_shaf"], plasma_dict["dz_shaf"]),
        }
        self.params.update_values(new_values, source=type(self).__name__)
