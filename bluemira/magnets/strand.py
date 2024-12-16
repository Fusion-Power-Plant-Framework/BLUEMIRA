# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Strand class"""

import numpy as np
import matplotlib.pyplot as plt
from bluemira.base.look_and_feel import bluemira_error
from bluemira.magnets.materials import Copper100, Material, Nb3Sn, NbTi
from bluemira.magnets.utils import parall_r, serie_r
from bluemira.geometry.tools import make_circle
from bluemira.geometry.face import BluemiraFace
from bluemira import display
from bluemira.display.plotter import PlotOptions


class Strand:
    """
    Represents a strand with a circular cross-section.
    """
    # Todo: discuss if it could be worth to consider the Strand as a PhysicalComponent

    def __init__(
            self,
            materials: list[Material],
            percentage: np.ndarray,
            d_strand: float = 0.82e-3,
    ):
        """
        Initialize a Strand instance.

        Parameters
        ----------
        materials:
            List of materials inside the strand.
        percentage:
            Percentage of each material (with the same ordering of materials). The sum of all percentages
            must be equal to 1.
        d_strand:
            Strand diameter in meters.
        """
        self._materials = None
        self._percentage = None
        self._d_strand = d_strand
        self._shape = None

        self.materials = materials
        self.percentage = percentage
        self.d_strand = d_strand

    @property
    def materials(self):
        return self._materials

    @materials.setter
    def materials(self, value: list[Material]):
        if len(value) < 1:
            msg = f"At least one material must be provided."
            bluemira_error(msg)
            raise ValueError(msg)
        self._materials = value

    @property
    def percentage(self):
        return self._percentage

    @percentage.setter
    def percentage(self, value: np.ndarray):
        if len(value) != len(self.materials):
            msg = f"Percentage and Materials must have the same length."
            bluemira_error(msg)
            raise ValueError(msg)

        if all(v>0 for v in value) and sum(value) != 1.0:
            msg = f"Percentages must be positive and their sum must be equal to 1.0."
            bluemira_error(msg)
            raise ValueError(msg)

        self._percentage = value

    @property
    def d_strand(self):
        return self._d_strand

    @d_strand.setter
    def d_strand(self, value: float):
        if value < 0.0:
            msg = f"Strand diameter must be positive."
            bluemira_error(msg)
            raise ValueError(msg)
        self._d_strand = value

    @property
    def area(self) -> float:
        """Returns the area of the strand cross-section in square meters."""
        return np.pi * self.d_strand ** 2 / 4

    def E(self) -> float:
        """
        Young's modulus (dummy value)

        Returns
        -------
        Young's modulus in Pascals.
        """
        return 0.1e9

    def erho(self, **kwargs) -> float:
        """
        Calculates the equivalent resistivity based on the parallel connection of strand components.

        Parameters
        ----------
        **kwargs:
            Additional parameters for calculating resistivity.

        Returns
        -------
        Equivalent resistivity in Ohm meters.
        """
        resistances = [
            x.erho(**kwargs) / self.area / self.percentage[i]
            for i, x in enumerate(self.materials)
        ]
        res_tot = parall_r(resistances)
        return res_tot * self.area

    def cp_v(self, **kwargs) -> float:
        """
        Calculates the equivalent specific heat based on the series connection of
        strand components.

        Parameters
        ----------
        **kwargs:
            Additional parameters for calculating specific heat.

        Returns
        -------
        Equivalent specific heat in Joules per Kelvin per meter.
        """
        specific_heat = [
            x.cp_v(**kwargs) * self.percentage[i] for i, x in enumerate(self.materials)
        ]
        return serie_r(specific_heat)

    def _create_shape(self):
        self._shape = BluemiraFace([make_circle(self.d_strand)])

    @property
    def shape(self):
        return self._create_shape()

    def plot(self, ax=None, *, show: bool = True, **kwargs,):
        # Todo: plot approach to be discussed (also in view of using PhysicalComponents).
        plot_options = PlotOptions()
        plot_options.view = "xy"
        ax = display.plot_2d(self.shape, options = plot_options, ax=ax, show=show, **kwargs)
        return ax


class SuperconductingStrand(Strand):
    """
    Represents a superconducting strand with a circular cross-section.
    """

    def __init__(
            self,
            materials: list[Material],
            percentage: np.array,
            d_strand: float = 0.82e-3,
    ):
        """
        Initialize a Strand instance.

        Parameters
        ----------
        materials:
            List of materials inside the strand. First material is the
            superconducting material.
        percentage:
            Percentage of each material (with the same ordering of materials).
        d_strand:
            Strand diameter in meters.
        """
        super().__init__(materials=materials, percentage=percentage, d_strand=d_strand)

    @property
    def sc_area(self):
        return self.area * self.percentage[0]

    def Jc(
            self, B: float, T: float, strain: float = 0.55, T_margin: float = 1.5,
            **kwargs
    ) -> float:
        """
        Returns the critical current density.

        Parameters
        ----------
        B:
            Operating magnetic field in Teslas.
        T:
            Operating temperature in Kelvins.
        esps:
            Total applied measured strain in percentage. Default is 0.55.
        T_margin:
            Strand temperature margin in operation in Kelvins. Default is 1.5.
        **kwargs:
            Additional parameters.

        Returns
        -------
        Critical current density in Amperes/m².
        """
        return 0

    def Ic(
            self, B: float, T: float, strain: float = 0.55, T_margin: float = 1.5,
            **kwargs
    ) -> float:
        """
        Returns the critical current.

        Parameters
        ----------
        B:
            Operating magnetic field in Teslas.
        T:
            Operating temperature in Kelvins.
        esps:
            Total applied measured strain in percentage. Default is 0.55.
        T_margin:
            Strand temperature margin in operation in Kelvins. Default is 1.5.
        **kwargs:
            Additional parameters.

        Returns
        -------
        Critical current density in Amperes.
        """
        return self.Jc(B=B, T=T, strain=strain, T_margin=T_margin) * self.sc_area


    def plot_Ic_B(self, B: np.ndarray, T: float, strain: float = 0.55, T_margin: float = 1.5, ax = None, show: bool = True):
        """
            Plot the critical current in a range of B
        """
        if ax is None:
            fig, ax = plt.subplots()

        Ic_sc = self.Ic(B=B , T=T, T_margin=T_margin)
        ax.plot(B, Ic_sc)
        # Adding the plot title and axis labels
        plt.title(f"Critical current for {self.__class__.__name__}\n"
                  f"T = {T} [K], Tmargin = {T_margin} [K], strain = {strain}")  # Title
        plt.xlabel('B [T]')            # X-axis label
        plt.ylabel('Ic [A]')           # Y-axis label
        # Enabling the grid
        plt.grid(True)
        if show:
            plt.show()
        return ax

class WireNb3Sn(SuperconductingStrand):
    """Represents an Nb3Sn strand made of 50% Copper100 and 50% Nb3Sn."""

    # superconducting parameters for the calculation of Ic
    c_ = 1.0
    Ca1 = 50.06
    Ca2 = 0.00
    eps_0a = 0.00312
    eps_m = -0.00059
    Bc20max = 33.24
    Tc0max = 16.34
    p = 0.593
    q = 2.156
    C = 83075 * 1e6

    def __init__(self, d_strand: float = 0.82e-3):
        """
        Initialize a WireNb3Sn instance.

        Parameters
        ----------
        d_strand:
            Strand diameter in meters. Default is 0.82e-3.
        """
        copper_100 = Copper100()
        mat_Nb3Sn = Nb3Sn()  # noqa: N806
        # materials: first material is the SC, then the other materials
        materials = [mat_Nb3Sn, copper_100]
        percentage = [0.5, 0.5]
        super().__init__(materials=materials, percentage=percentage, d_strand=d_strand)

    def Jc(
            self, B: float, T: float, strain: float = 0.55, T_margin: float = 1.5,
            **kwargs
    ) -> float:
        """
        Returns the strand critical current density.

        Parameters
        ----------
        B:
            Operating magnetic field in Teslas.
        T:
            Operating temperature in Kelvins.
        eps:
            Total applied measured strain in percentage. Default is 0.55.
        T_margin:
            Strand temperature margin in operation in Kelvins. Default is 1.5.

        Returns
        -------
        Strand current density in Amperes per square meter.
        """
        T_ = T + T_margin  # noqa: N806
        int_eps = -strain / 100
        eps_sh = self.Ca2 * self.eps_0a / (np.sqrt(self.Ca1 ** 2 - self.Ca2 ** 2))
        s_eps = 1 + (
                self.Ca1
                * (
                        np.sqrt(eps_sh ** 2 + self.eps_0a ** 2)
                        - np.sqrt((int_eps - eps_sh) ** 2 + self.eps_0a ** 2)
                )
                - self.Ca2 * int_eps
        ) / (1 - self.Ca1 * self.eps_0a)
        Bc0_eps = self.Bc20max * s_eps
        Tc0_eps = self.Tc0max * (s_eps) ** (1 / 3)  # noqa: N806
        t = T_ / Tc0_eps
        BcT_eps = Bc0_eps * (1 - t ** (1.52))
        b = B / BcT_eps
        hT = (1 - t ** (1.52)) * (1 - t ** 2)  # noqa: N806
        fPb = (b ** self.p) * (1 - b) ** self.q  # noqa: N806
        return self.c_ * (self.C / B) * s_eps * fPb * hT


class Wire_NbTi(SuperconductingStrand):  # noqa: N801
    """Represents an NbTi strand."""

    # superconducting parameters for the calculation of Ic
    n = 1.7
    Bc20_T = 15.19
    Tc0_K = 8.907
    C0 = 3.00e04
    C1 = 0.45
    a1 = 3.2
    b1 = 2.43
    C2 = 0.55
    a2 = 0.65
    b2 = 2
    g1 = 1.8
    g2 = 1.8

    def __init__(self, d_strand: float = 0.82e-3):
        """
        Initialize a Wire_NbTi instance.

        Parameters
        ----------
        d_strand:
            Strand diameter in meters. Default is 0.82e-3.
        """
        copper_100 = Copper100()
        mat_NbTi = NbTi()  # noqa: N806
        materials = [mat_NbTi, copper_100]
        percentage = [0.5, 0.5]
        super().__init__(materials=materials, percentage=percentage, d_strand=d_strand)

    def Jc(self, B: float, T: float, T_margin: float = 1.5, **kwargs):  # noqa:
        # ARG002, N803
        """
        NbTi critical current density.

        Parameters
        ----------
        B:
            Operating magnetic field in Teslas.
        T:
            Operating temperature in Kelvins.
        T_margin:
            Strand temperature margin in operation in Kelvins. Default is 1.5.

        Returns
        -------
        Critical current in Amperes per square meter.

        References
        ----------
        - Pinning Properties of Commercial Nb-Ti Wires Described by a 2-Components Model,
        Luigi Muzzi, Gianluca De Marzi, et al.
        - Fit data from DTT TF strand.
        """
        t = (T + T_margin) / self.Tc0_K
        b = B / self.Bc20_T
        tt = 1 - t ** self.n
        G = (self.a1 / (self.a1 + self.b1)) ** self.a1  # noqa: N806
        GG = (self.b1 / (self.a1 + self.b1)) ** self.b1  # noqa: N806
        GGG = G * GG  # noqa: N806
        F = (self.a2 / (self.a2 + self.b2)) ** self.a2
        FF = (self.b2 / (self.a2 + self.b2)) ** self.b2  # noqa: N806
        FFF = F * FF  # noqa: N806
        Jc = (  # noqa: N806
                     self.C0
                     * self.C1
                     / (B * GGG)
                     * (b / tt) ** self.a1
                     * (1 - b / tt) ** self.b1
                     * tt ** self.g1
                     + self.C0
                     * self.C2
                     / (B * FFF)
                     * (b / tt) ** self.a2
                     * (1 - b / tt) ** self.b2
                     * tt ** self.g2
             ) * 1e6
        return Jc  # noqa: RET504
