# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
TF coil case class
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from bluemira.base.look_and_feel import bluemira_warn, bluemira_print
from bluemira.magnets.conductor import Conductor
from bluemira.magnets.materials import Material
from bluemira.magnets.utils import parall_k, serie_k
from bluemira.magnets.winding_pack import WindingPack


class CaseTF:
    """TF case class"""

    def __init__(
        self,
        Ri: float,  # noqa: N803
        dy_ps: float,
        dy_vault: float,
        theta_TF: float,
        mat_case: Material,
        WPs: list[WindingPack],  # noqa: N803
        name: str = "",
    ):
        """
        Case structure for TF coils

        Parameters
        ----------
        Ri:
            external radius of the coil
        dy_ps:
            radial thickness of the case cap
        dy_vault:
            radial thickness of the vault
        theta_TF:
            toroidal angle of a TF coil
        mat_case:
            material of the case
        WPs:
            list of winding packs associated with the case
        name:
            string identifier
        """
        self.name = name
        self.dy_ps = dy_ps
        self.dy_vault = dy_vault
        self.theta_TF = theta_TF
        self._rad_theta_TF = np.radians(theta_TF)
        self.Ri = Ri
        self.mat_case = mat_case
        self.WPs = WPs

    @property
    def dx_i(self):
        """Toroidal length of the coil case at its maximum radial position [m]"""
        return 2 * self.Ri * np.tan(self._rad_theta_TF / 2)

    @property
    def dx_ps(self):
        """Average toroidal length of the ps plate [m]"""
        return (self.Ri + (self.Ri - self.dy_ps)) * np.tan(self._rad_theta_TF / 2)

    @property
    def n_conductors(self):
        """Total number of conductors in the winding pack."""
        return sum([w.n_conductors for w in self.WPs])

    def max_Iop(self, B, T, T_margin):  # noqa: N803, N802
        """Maximum operational current (equal to the critical current into the
        superconductor
        """
        return self.WPs[0].conductor.cable.sc_strand.Ic(B=B, T=T, T_margin=T_margin)

    @property
    def R_wp_i(self):  # noqa: N802
        """Maximum radial position for each winding pack"""
        dy_wp_cumsum = np.cumsum(np.array([0] + [w.dy for w in self.WPs]))
        return np.array([self.Ri - self.dy_ps - y for y in dy_wp_cumsum[0:-1]])

    @property
    def R_wp_k(self):  # noqa: N802
        """Minimum radial position for each winding pack"""
        return self.R_wp_i - np.array([w.dy for w in self.WPs])

    @property
    def Rk(self):  # noqa: N802
        """Minimum radial position of case"""
        return self.R_wp_k[-1] - self.dy_vault

    @property
    def dx_k(self):
        """Toroidal length of the case at its minimum radial position"""
        return 2 * self.Rk * np.tan(self._rad_theta_TF / 2)

    @property
    def dx_vault(self):
        """Average toroidal length of the vault"""
        return (self.R_wp_k[-1] + self.Rk) * np.tan(self._rad_theta_TF / 2)

    @property
    def area(self):
        """Total case area (winding packs included)"""
        return (self.dx_i + self.dx_k) * (self.Ri - self.Rk) / 2

    @property
    def area_case_jacket(self):
        """Total case jacket area (total case area - winding packs area)"""
        total_wp_area = np.sum([w.conductor.area * w.nx * w.ny for w in self.WPs])
        return self.area - total_wp_area

    @property
    def area_wps(self):
        """Tatal winding pack area"""
        return np.sum([w.area for w in self.WPs])

    @property
    def area_wps_jacket(self):
        """Tatal jacket area in the winding packs"""
        return np.sum([w.conductor.area_jacket * w.nx * w.ny for w in self.WPs])

    def Kx_ps(self, **kwargs):  # noqa: N802
        """Equivalent radial mechanical stiffness of ps"""
        return self.mat_case.E(**kwargs) * self.dy_ps / self.dx_ps

    def Kx_lat(self, **kwargs):  # noqa: N802
        """Equivalent radial stiffness of the lateral case part connected to each
        winding pack
        """
        dx_lat = np.array([
            (self.R_wp_i[i] + self.R_wp_k[i]) / 2 * np.tan(self._rad_theta_TF / 2)
            - w.dx / 2
            for i, w in enumerate(self.WPs)
        ])
        dy_lat = np.array([w.dy for w in self.WPs])
        return self.mat_case.E(**kwargs) * dy_lat / dx_lat

    def Kx_vault(self, **kwargs):  # noqa: N802
        """Equivalent radial stiffness of the vault"""
        return self.mat_case.E(**kwargs) * self.dy_vault / self.dx_vault

    def Kx(self, **kwargs):  # noqa: N802
        """Total equivalent radial stiffness of the case"""
        temp = [
            serie_k([
                self.Kx_lat(**kwargs)[i],
                w.Kx(**kwargs),
                self.Kx_lat(**kwargs)[i],
            ])
            for i, w in enumerate(self.WPs)
        ]
        return parall_k([self.Kx_ps(**kwargs), self.Kx_vault(**kwargs), *temp])

    def Ky_ps(self, **kwargs):  # noqa: N802
        """Equivalent toroidal stiffness of ps"""
        return self.mat_case.E(**kwargs) * self.dx_ps / self.dy_ps

    def Ky_lat(self, **kwargs):  # noqa: N802
        """Equivalent toroidal stiffness of the lateral case part connected to each
        winding
        """
        dx_lat = np.array([
            (self.R_wp_i[i] + self.R_wp_k[i]) / 2 * np.tan(self._rad_theta_TF / 2)
            - w.dx / 2
            for i, w in enumerate(self.WPs)
        ])
        dy_lat = np.array([w.dy for w in self.WPs])
        return self.mat_case.E(**kwargs) * dx_lat / dy_lat

    def Ky_vault(self, **kwargs):  # noqa: N802
        """Equivalent toroidal stiffness of the vault"""
        return self.mat_case.E(**kwargs) * self.dx_vault / self.dy_vault

    def Ky(self, **kwargs):  # noqa: N802
        """Total equivalent toroidal stiffness of the case"""
        temp = [
            parall_k([
                self.Ky_lat(**kwargs)[i],
                w.Ky(**kwargs),
                self.Ky_lat(**kwargs)[i],
            ])
            for i, w in enumerate(self.WPs)
        ]
        return serie_k([self.Ky_ps(**kwargs), self.Ky_vault(**kwargs), *temp])

    def _tresca_stress(self, pm: float, fz: float, **kwargs):
        """Procedure that calculate Tresca principal stress on the case

        Parameters
        ----------
            pm:
                radial magnetic pressure
            fz:
                vertical tension acting on the case
            kwargs:
                additional arguments necessary to calculate the structural properties
                of the case

        """
        # The maximum principal stress acting on the case nose is the compressive
        # hoop stress generated in the equivalent shell from the magnetic pressure. From
        # the Shell theory, for an isotropic continuous shell with a thickness ratio:
        beta = self.Rk / (self.Rk + self.dy_vault)
        # the maximum hoop stress, corrected to account for the presence of the WP, is
        # placed at the innermost radius of the case as:
        sigma_theta = (
            2.0 / (1 - beta**2) * pm * self.Kx_vault(**kwargs) / self.Kx(**kwargs)
        )

        # In addition to the radial centripetal force, the second in-plane component
        # to be accounted is the vertical force acting on the TFC inner-leg.
        # t_z = 0.5*np.log(self.Ri / Re) * MU_0_4PI * (360. / self.theta_TF) * I ** 2

        # As conservative approximation, the vertical force is considered to act only
        # on jackets and vault
        sigma_z = fz / (self.area_case_jacket + self.area_wps_jacket)
        return sigma_theta + sigma_z

    def optimize_vault_radial_thickness(
        self,
        pm: float,
        fz: float,
        T: float,  # noqa: N803
        B: float,
        allowable_sigma: float,
        bounds: np.array = None,
    ):
        """
        Optimize the vault radial thickness of the case

        Parameters
        ----------
        pm :
            The magnetic pressure applied along the radial direction (Pa).
        f_z :
            The force applied in the z direction, perpendicular to the case
            cross-section (N).
        T :
            The operating temperature (K).
        B :
            The operating magnetic field (T).
        allowable_sigma :
            The allowable stress (Pa) for the jacket material.
        bounds :
            Optional bounds for the jacket thickness optimization (default is None).

        Returns
        -------
            The result of the optimization process containing information about the
            optimal vault thickness.

        Raises
        ------
        ValueError
            If the optimization process did not converge.
        """
        bluemira_print(f"pm: {pm}, fz: {fz}, T: {T}, B: {B}, allowable_sigma: {allowable_sigma}")
        bluemira_print(f"bounds: {bounds}")

        method = None
        if bounds is not None:
            method = "bounded"

        result = minimize_scalar(
            fun=self._sigma_difference,
            args=(pm, fz, T, B, allowable_sigma),
            bounds=bounds,
            method=method,
            options={"xatol": 1e-4},
        )

        if not result.success:
            raise ValueError("dy_vault optimization did not converge.")
        self.dy_vault = result.x
        # print(f"Optimal dy_vault: {self.dy_vault}")
        # print(f"Tresca sigma: {self._tresca_stress(pm, fz, T=T, B=B) / 1e6} MPa")

        return result

    def _sigma_difference(
        self,
        dy_vault: float,
        pm: float,
        fz: float,
        T: float,
        B: float,
        allowable_sigma: float,
    ):
        """
        Fitness function for the optimization problem. It calculates the absolute
        difference between
        the Tresca stress and the allowable stress.

        Parameters
        ----------
        dy_vault :
            The thickness of the vault in the direction perpendicular to the
            applied pressure(m).
        pm :
            The magnetic pressure applied along the radial direction (Pa).
        fz :
            The force applied in the z direction, perpendicular to the case
            cross-section (N).
        T :
            The temperature (K) at which the conductor operates.
        B :
            The magnetic field (T) at which the conductor operates.
        allowable_sigma :
            The allowable stress (Pa) for the vault material.

        Returns
        -------
            The absolute difference between the calculated Tresca stress and the
            allowable stress (Pa).

        Notes
        -----
            This function modifies the case's vault thickness
            using the value provided in jacket_thickness.
        """
        self.dy_vault = dy_vault
        sigma = self._tresca_stress(pm, fz, T=T, B=B)
        # bluemira_print(f"sigma: {sigma}, allowable_sigma: {allowable_sigma}, diff: {sigma - allowable_sigma}")
        return abs(sigma - allowable_sigma)

    def optimize_jacket_and_vault(
        self,
        pm: float,
        fz: float,
        T: float,  # noqa: N803
        B: float,
        allowable_sigma: float,
        bounds_cond_jacket: np.array = None,
        bounds_dy_vault: np.array = None,
        layout: str = "auto",
        wp_reduction_factor: float = 0.8,
        min_gap_x: float = 0.05,
        n_layers_reduction: int = 4,
        max_niter: int = 10,
        eps: float = 1e-8,
        n_conds: int = None,
    ):
        if n_conds is None:
            n_conds = self.n_conductors
        conductor = self.WPs[0].conductor
        tot_err = 100 * eps
        i = 0
        while i < max_niter and tot_err > eps:
            i += 1
            print(f"Internal optimazion - iteration {i}")
            print(
                f"before optimization: conductor jacket area = {conductor.area_jacket}"
            )
            cond_area_jacket0 = conductor.area_jacket
            t_z_cable_jacket = (
                fz
                * self.area_wps_jacket
                / (self.area_case_jacket + self.area_wps_jacket)
                / self.n_conductors
            )
            conductor.optimize_jacket_conductor(
                pm, t_z_cable_jacket, T, B, allowable_sigma, bounds_cond_jacket
            )
            print(t_z_cable_jacket)
            print(f"after optimization: conductor jacket area = {conductor.area_jacket}")
            err_conductor_area_jacket = (
                abs(conductor.area_jacket - cond_area_jacket0) / cond_area_jacket0
            )

            self.rearrange_conductors_in_wp(
                n_conds,
                conductor,
                self.R_wp_i[0],
                wp_reduction_factor,
                min_gap_x,
                n_layers_reduction,
                layout=layout,
            )
            case_dy_vault0 = self.dy_vault
            print(f"before optimization: case dy_vault = {self.dy_vault}")
            self.optimize_vault_radial_thickness(
                pm=pm, fz=fz, T=T, B=B, allowable_sigma=allowable_sigma, bounds=bounds_dy_vault
            )

            print(f"after optimization: case dy_vault = {self.dy_vault}")
            delta_case_dy_vault = abs(self.dy_vault - case_dy_vault0)
            err_dy_vault = delta_case_dy_vault / self.dy_vault
            tot_err = err_dy_vault + err_conductor_area_jacket

            print(f"err_dy_jacket = {err_conductor_area_jacket}")
            print(f"err_dy_vault = {err_dy_vault}")
            print(f"tot_err = {tot_err}")

    def plot(self, ax=None, *, show: bool = False, homogenized: bool = False):
        """
        Schematic plot of the case cross-section.

        Parameters
        ----------
        ax:
            Matplotlib Axis on which the plot shall be displayed. If None,
            a new figure is created
        show:
            if True, the plot is displayed
        homogenized:
            if True, the winding pack is homogenized (default is False)
        """
        if ax is None:
            _, ax = plt.subplots()
            ax.set_aspect("equal", adjustable="box")

        p0 = np.array([-self.dx_i / 2, self.Ri])
        p1 = np.array([self.dx_i / 2, self.Ri])
        p2 = np.array([self.dx_k / 2, self.Rk])
        p3 = np.array([-self.dx_k / 2, self.Rk])

        points_ext = np.vstack((p0, p1, p2, p3, p0))

        ax.plot(points_ext[:, 0], points_ext[:, 1], "r")
        for i, w in enumerate(self.WPs):
            xc_w = 0
            yc_w = self.R_wp_i[i] - w.dy / 2
            ax = w.plot(xc=xc_w, yc=yc_w, ax=ax, homogenized=homogenized)

        if show:
            plt.show()

        return ax

    def rearrange_conductors_in_wp(
        self,
        n_conductors: int,
        cond: Conductor,
        R_wp_i: float,  # noqa: N803
        wp_reduction_factor: float,
        min_gap_x: float,
        n_layers_reduction: int,
        layout: str = "auto",
    ):
        """
        Rearrange the total number of conductors into the TF coil case considering
        a specific conductor

        Parameters
        ----------
        n_conductors:
            number of supercoductors
        cond:
            type of conductor
        R_wp_i:
            initial radial distance at which the first winding pack is placed
        dx_WP:
            toroidal length of the first winding pack
        min_gap_x:
            minimum toroidal distance between winding pack and tf coils lateral faces
        n_layers_reduction:
            number of turns to be removed when calculating a new pancake
        layout:
            type of layout for the conductor placement. Available types are "single", "layer", "pancake".
                - "auto" means that the conductors are placed in order
                    to only verify the geometrical constraints
                - "layer" adds the odd constrain on ny (ny % 2 = 0).
                - "pancake" adds the even constrain on nx (nx % 2 = 0)

        Returns
        -------
            np.array: number of turns and layers for each "pancake"

        Note
        ----
            The final number of allocated superconductors could slightly differ from
            the one defined in n_conductors due to the necessity to close the final
            layer.
        """
        dx_WP = self.dx_i * wp_reduction_factor

        WPs = []  # noqa: N806
        # number of conductors to be allocated
        remaining_conductors = n_conductors
        # maximum number of internal iterations
        i_max = 50
        i = 0
        while i < i_max and remaining_conductors > 0:
            i += 1
            # print(f"Rearrange conductors in WP - iteration: {i}")
            # print(f"remaining_conductors: {remaining_conductors}")

            # maximum toroidal dimension of the WP most outer pancake
            # dx_WP = 2 * (R_wp_i * np.tan(self._rad_theta_TF / 2) - dx0_wp)

            # maximum number of turns on the considered WP
            if i == 1:
                n_layers_max = int(math.floor(dx_WP / cond.dx))
                if layout == "pancake":
                    n_layers_max = int(math.floor(dx_WP / cond.dx / 2.0) * 2)
                    if n_layers_max == 0:
                        n_layers_max = 2
            else:
                n_layers_max -= n_layers_reduction

            if n_layers_max < 1:
                raise ValueError(
                    f"n_layers_max: {n_layers_max} < 1. There is not enough space to "
                    f"allocate all the conductors"
                )

            dx_WP = n_layers_max * cond.dx  # noqa: N806

            gap_0 = R_wp_i * np.tan(self._rad_theta_TF / 2) - dx_WP / 2
            gap_1 = min_gap_x

            max_dy = (gap_0 - gap_1) / np.tan(self._rad_theta_TF / 2)
            n_turns_max = min(
                int(np.floor(max_dy / cond.dy)),
                int(np.ceil(remaining_conductors / n_layers_max)),
            )
            if layout == "layer":
                n_turns_max = min(
                    int(np.floor(max_dy / cond.dy / 2.0) * 2),
                    int(np.ceil(remaining_conductors / n_layers_max / 2.0) * 2),
                )
                if n_turns_max == 0:
                    n_turns_max = 2

            if n_turns_max < 1:
                raise ValueError(
                    f"n_turns_max: {n_turns_max} < 1. There is not enough space to "
                    f"allocate all the conductors"
                )

            WPs.append(WindingPack(conductor=cond, nx=n_layers_max, ny=n_turns_max))

            remaining_conductors -= n_layers_max * n_turns_max

            if remaining_conductors < 0:
                bluemira_warn(
                    f"{abs(remaining_conductors)}/{n_layers_max * n_turns_max} have been added"
                    f" to complete the last winding pack (nx={n_layers_max}, ny={n_turns_max})."
                )

            R_wp_i -= n_turns_max * cond.dy  # noqa: N806
            # dx_WP = dx_WP - n_layers_reduction * cond.dx
            # print(f"remaining_conductors: {remaining_conductors}")

        self.WPs = WPs
