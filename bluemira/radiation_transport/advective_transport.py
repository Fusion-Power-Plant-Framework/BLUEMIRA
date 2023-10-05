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
A simplified 2-D solver for calculating charged particle heat loads.
"""

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from bluemira.base.constants import EPS
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.display.plotter import plot_coordinates
from bluemira.equilibria.find import find_flux_surface_through_point
from bluemira.equilibria.flux_surfaces import OpenFluxSurface
from bluemira.geometry.coordinates import Coordinates, coords_plane_intersect
from bluemira.geometry.plane import BluemiraPlane
from bluemira.radiation_transport.error import AdvectionTransportError

__all__ = ["ChargedParticleSolver"]


class ChargedParticleSolver:
    """
    A simplified charged particle transport model along open field lines.

    Parameters
    ----------
    config: Dict[str, float]
        The parameters for running the transport model. See
        :class:`ChargedParticleSolverParams` for available parameters
        and their defaults.
    equilibrium: Equilibrium
        The equilibrium defining flux surfaces.
    dx_mp: float (optional)
        The midplane spatial resolution between flux surfaces [m]
        (default: 0.001).
    """

    def __init__(self, config: Dict[str, float], equilibrium, dx_mp: float = 0.001):
        self.eq = equilibrium
        self.params = self._make_params(config)
        self._check_params()
        self.dx_mp = dx_mp

        # Constructors
        self.first_wall = None
        self.flux_surfaces_ob_lfs = None
        self.flux_surfaces_ob_hfs = None
        self.flux_surfaces_ib_lfs = None
        self.flux_surfaces_ib_hfs = None
        self.x_sep_omp = None
        self.x_sep_imp = None
        self.result = None

        # Pre-processing
        o_points, _ = self.eq.get_OX_points()
        self._o_point = o_points[0]
        z = self._o_point.z
        self._yz_plane = BluemiraPlane.from_3_points([0, 0, z], [1, 0, z], [1, 1, z])

    @property
    def flux_surfaces(self):
        """
        All flux surfaces in the ChargedParticleSolver.

        Returns
        -------
        flux_surfaces: List[PartialOpenFluxSurface]
        """
        flux_surfaces = []
        for group in [
            self.flux_surfaces_ob_lfs,
            self.flux_surfaces_ob_hfs,
            self.flux_surfaces_ib_lfs,
            self.flux_surfaces_ib_hfs,
        ]:
            if group:
                flux_surfaces.extend(group)
        return flux_surfaces

    def _check_params(self):
        """
        Check input fractions for validity.
        """
        # Check lower power fractions
        lower_power = self.params.f_lfs_lower_target + self.params.f_hfs_lower_target
        upper_power = self.params.f_lfs_upper_target + self.params.f_hfs_upper_target
        power_sum = lower_power + upper_power

        if not np.isclose(power_sum, 1.0, rtol=EPS, atol=1e-9):
            raise AdvectionTransportError(
                f"Total power fractions should sum to 1, not : {power_sum}"
            )

        zero_in_one_direction = np.any(
            np.isclose([lower_power, upper_power], 0, rtol=EPS, atol=1e-9)
        )
        if self.eq.is_double_null and zero_in_one_direction:
            bluemira_warn(
                "A DN equilibrium was detected but your power distribution"
                " is 0 in either the lower or upper directions."
            )
        elif not self.eq.is_double_null and not zero_in_one_direction:
            bluemira_warn(
                "A SN equilibrium was detected but you power distribution is not 0"
                " in either the lower or upper directions."
            )

    @staticmethod
    def _process_first_wall(first_wall):
        """
        Force working first wall geometry to be closed and counter-clockwise.
        """
        first_wall = deepcopy(first_wall)

        if not first_wall.check_ccw(axis=[0, 1, 0]):
            bluemira_warn(
                "First wall should be oriented counter-clockwise. Reversing it."
            )
            first_wall.reverse()

        if not first_wall.closed:
            bluemira_warn("First wall should be a closed geometry. Closing it.")
            first_wall.close()
        return first_wall

    @staticmethod
    def _get_arrays(flux_surfaces):
        """
        Get arrays of flux surface values.
        """
        x_mp = np.array([fs.x_start for fs in flux_surfaces])
        z_mp = np.array([fs.z_start for fs in flux_surfaces])
        x_fw = np.array([fs.x_end for fs in flux_surfaces])
        z_fw = np.array([fs.z_end for fs in flux_surfaces])
        alpha = np.array([fs.alpha for fs in flux_surfaces])
        return x_mp, z_mp, x_fw, z_fw, alpha

    def _get_sep_out_intersection(self, outboard=True):
        """
        Find the middle and maximum outboard mid-plane psi norm values
        """
        yz_plane = self._yz_plane
        o_point = self._o_point
        separatrix = self.eq.get_separatrix()

        if not isinstance(separatrix, Coordinates):
            sep1_intersections = coords_plane_intersect(separatrix[0], yz_plane)
            sep2_intersections = coords_plane_intersect(separatrix[1], yz_plane)
            sep1_arg = np.argmin(np.abs(sep1_intersections.T[0] - o_point.x))
            sep2_arg = np.argmin(np.abs(sep2_intersections.T[0] - o_point.x))
            x_sep1_mp = sep1_intersections.T[0][sep1_arg]
            x_sep2_mp = sep2_intersections.T[0][sep2_arg]
            if outboard:
                x_sep_mp = x_sep1_mp if x_sep1_mp > x_sep2_mp else x_sep2_mp
            else:
                x_sep_mp = x_sep1_mp if x_sep1_mp < x_sep2_mp else x_sep2_mp
        else:
            sep_intersections = coords_plane_intersect(separatrix, yz_plane)
            sep_arg = np.argmin(np.abs(sep_intersections.T[0] - o_point.x))
            x_sep_mp = sep_intersections.T[0][sep_arg]

        out_intersections = coords_plane_intersect(self.first_wall, yz_plane)
        if outboard:
            x_out_mp = np.max(out_intersections.T[0])
        else:
            x_out_mp = np.min(out_intersections.T[0])

        return x_sep_mp, x_out_mp

    def _make_flux_surfaces(self, x, z):
        """
        Make individual PartialOpenFluxSurfaces through a point.
        """
        coords = find_flux_surface_through_point(
            self.eq.x, self.eq.z, self.eq.psi(), x, z, self.eq.psi(x, z)
        )
        coords = Coordinates({"x": coords[0], "z": coords[1]})
        f_s = OpenFluxSurface(coords)
        lfs, hfs = f_s.split(self._o_point, plane=self._yz_plane)
        return lfs, hfs

    def _make_flux_surfaces_ob(self):
        """
        Make the flux surfaces on the outboard.
        """
        self.x_sep_omp, x_out_omp = self._get_sep_out_intersection(outboard=True)

        self.flux_surfaces_ob_lfs = []
        self.flux_surfaces_ob_hfs = []

        x = self.x_sep_omp + self.dx_mp
        while x < x_out_omp - EPS:
            lfs, hfs = self._make_flux_surfaces(x, self._o_point.z)
            self.flux_surfaces_ob_lfs.append(lfs)
            self.flux_surfaces_ob_hfs.append(hfs)
            x += self.dx_mp

    def _make_flux_surfaces_ib(self):
        """
        Make the flux surfaces on the inboard.
        """
        self.x_sep_imp, x_out_imp = self._get_sep_out_intersection(outboard=False)

        self.flux_surfaces_ib_lfs = []
        self.flux_surfaces_ib_hfs = []
        x = self.x_sep_imp - self.dx_mp
        while x > x_out_imp + EPS:
            lfs, hfs = self._make_flux_surfaces(x, self._o_point.z)
            self.flux_surfaces_ib_lfs.append(lfs)
            self.flux_surfaces_ib_hfs.append(hfs)
            x -= self.dx_mp

    def _clip_flux_surfaces(self, first_wall):
        """
        Clip the flux surfaces to a first wall. Catch the cases where no intersections
        are found.
        """
        for group in [
            self.flux_surfaces_ob_lfs,
            self.flux_surfaces_ob_hfs,
            self.flux_surfaces_ib_lfs,
            self.flux_surfaces_ib_hfs,
        ]:
            if group:
                for i, flux_surface in enumerate(group):
                    flux_surface.clip(first_wall)
                    if flux_surface.alpha is None:
                        # No intersection detected between flux surface and first wall
                        # Drop the flux surface from the group
                        group.pop(i)

    def analyse(self, first_wall):
        """
        Perform the calculation to obtain charged particle heat fluxes on the
        the specified first_wall

        Parameters
        ----------
        first_wall: Coordinates
            The closed first wall geometry on which to calculate the heat flux

        Returns
        -------
        x: np.array
            The x coordinates of the flux surface intersections
        z: np.array
            The z coordinates of the flux surface intersections
        heat_flux: np.array
            The perpendicular heat fluxes at the intersection points [MW/m^2]
        """
        self.first_wall = self._process_first_wall(first_wall)

        if self.eq.is_double_null:
            x, z, hf = self._analyse_DN(first_wall)
        else:
            x, z, hf = self._analyse_SN(first_wall)

        self.result = x, z, hf
        return x, z, hf

    def _analyse_SN(self, first_wall):
        """
        Calculation for the case of single nulls.
        """
        self._make_flux_surfaces_ob()

        # Find the intersections of the flux surfaces with the first wall
        self._clip_flux_surfaces(first_wall)

        x_omp, z_omp, x_lfs_inter, z_lfs_inter, alpha_lfs = self._get_arrays(
            self.flux_surfaces_ob_lfs
        )
        _, _, x_hfs_inter, z_hfs_inter, alpha_hfs = self._get_arrays(
            self.flux_surfaces_ob_hfs
        )

        # Calculate values at OMP
        dx_omp = x_omp - self.x_sep_omp
        Bp_omp = self.eq.Bp(x_omp, z_omp)
        Bt_omp = self.eq.Bt(x_omp)
        B_omp = np.hypot(Bp_omp, Bt_omp)

        # Parallel power at the outboard midplane
        q_par_omp = self._q_par(x_omp, dx_omp, B_omp, Bp_omp)

        # Calculate values at intersections
        Bp_lfs = self.eq.Bp(x_lfs_inter, z_lfs_inter)
        Bp_hfs = self.eq.Bp(x_hfs_inter, z_hfs_inter)

        # Calculate parallel power at the intersections
        # Note that flux expansion terms cancel down to this
        q_par_lfs = q_par_omp * Bp_lfs / B_omp
        q_par_hfs = q_par_omp * Bp_hfs / B_omp

        # Calculate perpendicular heat fluxes
        heat_flux_lfs = self.params.f_lfs_lower_target * q_par_lfs * np.sin(alpha_lfs)
        heat_flux_hfs = self.params.f_hfs_lower_target * q_par_hfs * np.sin(alpha_hfs)

        # Correct power (energy conservation)
        q_omp_int = 2 * np.pi * np.sum(q_par_omp / (B_omp / Bp_omp) * self.dx_mp * x_omp)
        f_correct_power = self.params.P_sep_particle / q_omp_int
        return (
            np.append(x_lfs_inter, x_hfs_inter),
            np.append(z_lfs_inter, z_hfs_inter),
            f_correct_power * np.append(heat_flux_lfs, heat_flux_hfs),
        )

    def _analyse_DN(self, first_wall):
        """
        Calculation for the case of double nulls.
        """
        self._make_flux_surfaces_ob()
        self._make_flux_surfaces_ib()

        # Find the intersections of the flux surfaces with the first wall
        self._clip_flux_surfaces(first_wall)

        (
            x_omp,
            z_omp,
            x_lfs_down_inter,
            z_lfs_down_inter,
            alpha_lfs_down,
        ) = self._get_arrays(self.flux_surfaces_ob_lfs)
        _, _, x_lfs_up_inter, z_lfs_up_inter, alpha_lfs_up = self._get_arrays(
            self.flux_surfaces_ob_hfs
        )
        (
            x_imp,
            z_imp,
            x_hfs_down_inter,
            z_hfs_down_inter,
            alpha_hfs_down,
        ) = self._get_arrays(self.flux_surfaces_ib_lfs)
        _, _, x_hfs_up_inter, z_hfs_up_inter, alpha_hfs_up = self._get_arrays(
            self.flux_surfaces_ib_hfs
        )

        # Calculate values at OMP
        dx_omp = x_omp - self.x_sep_omp
        Bp_omp = self.eq.Bp(x_omp, z_omp)
        Bt_omp = self.eq.Bt(x_omp)
        B_omp = np.hypot(Bp_omp, Bt_omp)

        # Calculate values at IMP
        dx_imp = abs(x_imp - self.x_sep_imp)
        Bp_imp = self.eq.Bp(x_imp, z_imp)
        Bt_imp = self.eq.Bt(x_imp)
        B_imp = np.hypot(Bp_imp, Bt_imp)

        # Parallel power at the outboard and inboard midplane
        q_par_omp = self._q_par(x_omp, dx_omp, B_omp, Bp_omp)
        q_par_imp = self._q_par(x_imp, dx_imp, B_imp, Bp_imp, outboard=False)

        # Calculate poloidal field at intersections
        Bp_lfs_down = self.eq.Bp(x_lfs_down_inter, z_lfs_down_inter)
        Bp_lfs_up = self.eq.Bp(x_lfs_up_inter, z_lfs_up_inter)
        Bp_hfs_down = self.eq.Bp(x_hfs_down_inter, z_hfs_down_inter)
        Bp_hfs_up = self.eq.Bp(x_hfs_up_inter, z_hfs_up_inter)

        # Calculate parallel power at the intersections
        # Note that flux expansion terms cancel down to this
        q_par_lfs_down = q_par_omp * Bp_lfs_down / B_omp
        q_par_lfs_up = q_par_omp * Bp_lfs_up / B_omp
        q_par_hfs_down = q_par_imp * Bp_hfs_down / B_imp
        q_par_hfs_up = q_par_imp * Bp_hfs_up / B_imp

        # Calculate perpendicular heat fluxes
        heat_flux_lfs_down = (
            self.params.f_lfs_lower_target * q_par_lfs_down * np.sin(alpha_lfs_down)
        )
        heat_flux_lfs_up = (
            self.params.f_lfs_upper_target * q_par_lfs_up * np.sin(alpha_lfs_up)
        )
        heat_flux_hfs_down = (
            self.params.f_hfs_lower_target * q_par_hfs_down * np.sin(alpha_hfs_down)
        )
        heat_flux_hfs_up = (
            self.params.f_hfs_upper_target * q_par_hfs_up * np.sin(alpha_hfs_up)
        )

        # Correct power (energy conservation)
        q_omp_int = 2 * np.pi * np.sum(q_par_omp * Bp_omp / B_omp * self.dx_mp * x_omp)
        q_imp_int = 2 * np.pi * np.sum(q_par_imp * Bp_imp / B_imp * self.dx_mp * x_imp)

        total_power = self.params.P_sep_particle
        f_outboard = self.params.f_lfs_lower_target + self.params.f_lfs_upper_target
        f_inboard = self.params.f_hfs_lower_target + self.params.f_hfs_upper_target
        f_correct_lfs_down = (
            total_power * self.params.f_lfs_lower_target / f_outboard
        ) / q_omp_int
        f_correct_lfs_up = (
            total_power * self.params.f_lfs_upper_target / f_outboard
        ) / q_omp_int
        f_correct_hfs_down = (
            total_power * self.params.f_hfs_lower_target / f_inboard
        ) / q_imp_int
        f_correct_hfs_up = (
            total_power * self.params.f_hfs_upper_target / f_inboard
        ) / q_imp_int

        return (
            np.concatenate(
                [x_lfs_down_inter, x_lfs_up_inter, x_hfs_down_inter, x_hfs_up_inter]
            ),
            np.concatenate(
                [z_lfs_down_inter, z_lfs_up_inter, z_hfs_down_inter, z_hfs_up_inter]
            ),
            np.concatenate(
                [
                    f_correct_lfs_down * heat_flux_lfs_down,
                    f_correct_lfs_up * heat_flux_lfs_up,
                    f_correct_hfs_down * heat_flux_hfs_down,
                    f_correct_hfs_up * heat_flux_hfs_up,
                ]
            ),
        )

    def _q_par(self, x, dx, B, Bp, outboard=True):
        """
        Calculate the parallel power at the midplane.
        """
        p_sol_near = self.params.P_sep_particle * self.params.f_p_sol_near
        p_sol_far = self.params.P_sep_particle * (1 - self.params.f_p_sol_near)
        if outboard:
            lq_near = self.params.fw_lambda_q_near_omp
            lq_far = self.params.fw_lambda_q_far_omp
        else:
            lq_near = self.params.fw_lambda_q_near_imp
            lq_far = self.params.fw_lambda_q_far_imp
        return (
            (
                p_sol_near * np.exp(-dx / lq_near) / lq_near
                + p_sol_far * np.exp(-dx / lq_far) / lq_far
            )
            * B
            / (Bp * 2 * np.pi * x)
        )

    def plot(self, ax: Axes = None, show=False):
        """
        Plot the ChargedParticleSolver results.
        """
        if ax is None:
            _, ax = plt.subplots()

        plot_coordinates(self.first_wall, ax=ax, linewidth=0.5, fill=False)
        separatrix = self.eq.get_separatrix()

        if isinstance(separatrix, Coordinates):
            separatrix = [separatrix]

        for sep in separatrix:
            plot_coordinates(sep, ax=ax, linewidth=0.2)

        for f_s in self.flux_surfaces:
            plot_coordinates(f_s.coords, ax=ax, linewidth=0.01)

        cm = ax.scatter(
            self.result[0],
            self.result[1],
            c=self.result[2],
            s=10,
            zorder=40,
            cmap="plasma",
        )
        f = ax.figure
        f.colorbar(cm, label="MW/m²")
        if show:
            plt.show()
        return ax

    @staticmethod
    def _make_params(config):
        """Convert the given params to ``ChargedParticleSolverParams``"""
        if isinstance(config, dict):
            try:
                return ChargedParticleSolverParams(**config)
            except TypeError:
                unknown = [
                    k for k in config if k not in fields(ChargedParticleSolverParams)
                ]
                raise TypeError(
                    f"Unknown config parameter(s) {str(unknown)[1:-1]}"
                ) from None
        elif isinstance(config, ChargedParticleSolverParams):
            return config
        else:
            raise TypeError(
                "Unsupported type: 'config' must be a 'dict', or "
                "'ChargedParticleSolverParams' instance; found "
                f"'{type(config).__name__}'."
            )


@dataclass
class ChargedParticleSolverParams:
    P_sep_particle: float = 150
    """Separatrix power [MW]."""

    f_p_sol_near: float = 0.5
    """Near scrape-off layer power rate [dimensionless]."""

    fw_lambda_q_near_omp: float = 0.003
    """Lambda q near SOL at the outboard [m]."""

    fw_lambda_q_far_omp: float = 0.05
    """Lambda q far SOL at the outboard [m]."""

    fw_lambda_q_near_imp: float = 0.003
    """Lambda q near SOL at the inboard [m]."""

    fw_lambda_q_far_imp: float = 0.05
    """Lambda q far SOL at the inboard [m]."""

    f_lfs_lower_target: float = 0.9
    """Fraction of SOL power deposited on the LFS lower target [dimensionless]."""

    f_hfs_lower_target: float = 0.1
    """Fraction of SOL power deposited on the HFS lower target [dimensionless]."""

    f_lfs_upper_target: float = 0
    """
    Fraction of SOL power deposited on the LFS upper target (DN only)
    [dimensionless].
    """

    f_hfs_upper_target: float = 0
    """
    Fraction of SOL power deposited on the HFS upper target (DN only)
    [dimensionless].
    """
