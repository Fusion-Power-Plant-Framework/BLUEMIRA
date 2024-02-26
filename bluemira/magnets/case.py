import math
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from bluemira.magnets.base import StructuralComponent, parall_k, serie_k
from bluemira.magnets.conductor import Conductor
from bluemira.magnets.materials import Material
from bluemira.magnets.winding_pack import WindingPack


class CaseTF(StructuralComponent):
    def __init__(
            self,
            Ri: float,
            dy_ps: float,
            dy_vault: float,
            theta_TF: float,
            mat_case: Material,
            WPs: List[WindingPack],
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
        return 2 * (self.Ri + self.Ri - self.dy_ps) / 2 * np.tan(self._rad_theta_TF / 2)

    @property
    def R_wp_i(self):
        """Maximum radial position for each winding pack"""
        dy_wp_cumsum = np.cumsum(np.array([0] + [w.dy for w in self.WPs]))
        return np.array([self.Ri - self.dy_ps - y for y in dy_wp_cumsum[0:-1]])

    @property
    def R_wp_k(self):
        """Minimum radial position for each winding pack"""
        return self.R_wp_i - np.array([w.dy for w in self.WPs])

    @property
    def Rk(self):
        """Minimum radial position of case"""
        return self.R_wp_k[-1] - self.dy_vault

    @property
    def dx_k(self):
        """Toroidal length of the case at its minimum radial position"""
        return 2 * self.Rk * np.tan(self._rad_theta_TF / 2)

    @property
    def dx_vault(self):
        """Average toroidal length of the vault"""
        return 2 * (self.R_wp_k[-1] + self.Rk) / 2 * np.tan(self._rad_theta_TF / 2)

    def Kx_ps(self, **kwargs):
        """Equivalent radial mechanical stiffness of ps"""
        return self.mat_case.ym(**kwargs) * self.dx_ps / self.dy_ps

    def Ky_ps(self, **kwargs):
        """Equivalent toroidal stiffness of ps"""
        return self.mat_case.ym(**kwargs) * self.dy_ps / self.dx_ps

    def Kx_lat(self, **kwargs):
        """Equivalent radial stiffness of the lateral case part connected to each winding pack"""
        dx_lat = np.array([
            (self.R_wp_i[i] + self.R_wp_k[i]) / 2 * np.tan(self._rad_theta_TF / 2)
            - w.dx / 2
            for i, w in enumerate(self.WPs)
        ])
        dy_lat = np.array([w.dy for w in self.WPs])
        return self.mat_case.ym(**kwargs) * dx_lat / dy_lat

    def Ky_lat(self, **kwargs):
        """Equivalent toroidal stiffness of the lateral case part connected to each winding"""
        dx_lat = np.array([
            (self.R_wp_i[i] + self.R_wp_k[i]) / 2 * np.tan(self._rad_theta_TF / 2)
            - w.dx / 2
            for i, w in enumerate(self.WPs)
        ])
        dy_lat = np.array([w.dy for w in self.WPs])
        return self.mat_case.ym(**kwargs) * dy_lat / dx_lat

    def Kx_vault(self, **kwargs):
        """Equivalent radial stiffness of the vault"""
        return self.mat_case.ym(**kwargs) * self.dx_vault / self.dy_vault

    def Ky_vault(self, **kwargs):
        """Equivalent toroidal stiffness of the vault"""
        return self.mat_case.ym(**kwargs) * self.dy_vault / self.dx_vault

    def Kx(self, **kwargs):
        """Total equivalent radial stiffness of the case"""
        temp = [
            parall_k([
                self.Kx_lat(**kwargs)[i],
                w.Kx(**kwargs),
                self.Kx_lat(**kwargs)[i],
            ])
            for i, w in enumerate(self.WPs)
        ]
        return serie_k([self.Kx_ps(**kwargs), self.Kx_vault(**kwargs)] + temp)

    def Ky(self, **kwargs):
        """Total equivalent toroidal stiffness of the case"""
        temp = [
            serie_k([self.Ky_lat(**kwargs)[i], w.Ky(**kwargs), self.Ky_lat(**kwargs)[i]])
            for i, w in enumerate(self.WPs)
        ]
        return parall_k([self.Ky_ps(**kwargs), self.Ky_vault(**kwargs)] + temp)

    def Xx(self, **kwargs):
        Kx_wp = parall_k([w.Kx(**kwargs) for w in self.WPs])
        return Kx_wp / self.Kx(**kwargs)

    def Yy(self, **kwargs):
        return self.Ky_vault(**kwargs) / self.Ky(**kwargs)

    def plot(self, ax=None, show: bool = False, homogenized: bool = False):
        if ax is None:
            _, ax = plt.subplots()

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

    def rearrange_conductors_in_wp_type1(
            self,
            n_conductors: int,
            cond: Conductor,
            R_wp_i: float,
            dx0_wp: float,
            min_gap_x: float,
            n_layers_reduction: int,
    ):
        """
        Rearrange the total number of conductors into the TF coil case considering a specific conductor

        Parameters
        ----------
        n_conductors:
            number of supercoductors
        cond:
            type of conductor
        R_wp_i:
            initial radial distance at which the first winding pack is placed
        dx0_wp:
            toroidal distance between inner leg lateral face and most outer layer in the winding pack
        min_gap_x:
            minimum toroidal distance between winding pack and tf coils lateral faces
        n_layers_reduction:
            number of turns to be removed when calculating a new pancake

        Returns
        -------
            np.array: number of turns and layers for each "pancake"

        Note
        ----
            The final number of allocated superconductors could slightly differ from the one defined
            in n_conductors due to the necessity to close the final layer.
        """
        WPs = []
        # number of conductors to be allocated
        remaining_conductors = n_conductors
        # maximum number of internal iterations
        i_max = 50
        i = 0
        while i < i_max and remaining_conductors > 0:
            i = i + 1
            print(f"iteration: {i}")
            print(f"remaining_conductors: {remaining_conductors}")

            # maximum toroidal dimension of the WP most outer pancake
            dx_WP = 2 * (R_wp_i * np.tan(self._rad_theta_TF / 2) - dx0_wp)

            # maximum number of turns on the considered pancake
            n_layers_max = int(math.floor(dx_WP / cond.dx))

            if n_layers_max < 1:
                raise ValueError(
                    f"n_layers_max: {n_layers_max} < 1. There is not enough space to allocate all the conductors"
                )

            dx_WP = n_layers_max * cond.dx

            gap_0 = R_wp_i * np.tan(self._rad_theta_TF / 2) - dx_WP / 2
            gap_1 = min_gap_x

            max_dy = (gap_0 - gap_1) / np.tan(self._rad_theta_TF / 2)
            n_turns_max = min(
                int(np.floor(max_dy / cond.dy)),
                int(np.ceil(remaining_conductors / n_layers_max)),
            )

            if n_turns_max < 1:
                raise ValueError(
                    f"n_turns_max: {n_turns_max} < 1. There is not enough space to allocate all the conductors"
                )

            WPs.append(WindingPack(conductor=cond, nl=n_layers_max, nt=n_turns_max))

            remaining_conductors = remaining_conductors - (n_layers_max * n_turns_max)

            if remaining_conductors < 0:
                print(
                    f"WARNING: {abs(remaining_conductors)} have been added to complete the last layer."
                )

            R_wp_i = R_wp_i - n_turns_max * cond.dy
            dx0_wp = (
                    R_wp_i * np.tan(self._rad_theta_TF / 2)
                    - (dx_WP - n_layers_reduction * cond.dx) / 2
            )

            print(f"remaining_conductors: {remaining_conductors}")

        self.WPs = WPs
