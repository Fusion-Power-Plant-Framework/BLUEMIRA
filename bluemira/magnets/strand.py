from typing import List, Union

import numpy as np

from bluemira.magnets.base import parall_r, serie_r
from bluemira.magnets.materials import Copper100, Material, Nb3Sn, NbTi


class Strand:
    def __init__(
            self,
            materials: List[Material],
            percentage: Union[np.array, List[float]],
            d_strand: float = 0.82e-3,
    ):
        """
        Class that represents a strand

        Parameters
        ----------
        materials:
            list of materials inside the strand
        percentage:
            percentage of each material (with the same ordering of materials)
        d_strand:
            strand diameter [m]
        """
        self.materials = materials
        self.percentage = percentage
        self.d_strand = d_strand

    @property
    def area(self) -> float:
        return np.pi * self.d_strand ** 2 / 4

    def ym(self, T: float) -> float:
        """
        Young moduli for the considered strand

        Parameters
        ----------
        kwargs:
            Other Parameters that shall be used to calculate the Young moduli (e.g. Temperature)

        Returns
        -------
            float [Pa]

        Note
        ----

            Dummy value due missing data for the strand
        """
        return 0.1e9

    def res(self, **kwargs) -> float:
        """
        Strand's equivalent resistivity, compute the parallel between strand's components

        Parameters
        ----------
        Return
        ------
            float [Ohm m]
        """
        resistances = [
            x.res(**kwargs) / self.area / self.percentage[i]
            for i, x in enumerate(self.materials)
        ]
        res_tot = parall_r(resistances)
        return res_tot * self.area

    def cp_v(self, **kwargs) -> float:
        """
        Strand's equivalent Specific Heat, compute the series between strand's components

        Parameters
        ----------
        Return
        ------
            float [J/K/m]
        """
        specific_heat = [
            x.cp_v(**kwargs) * self.percentage[i] for i, x in enumerate(self.materials)
        ]
        return serie_r(specific_heat)

    def Ic(
            self, B: float, T: float, strain: float = 0.55, T_margin: float = 1.5
    ) -> float:
        """
        Critical current from Jc(B,T,strain)

        Parameters
        ----------
        T:
            Operating temperature [K]
        B:
            Operating magnetic field [T]
        strain:
            total applied measured strain [%]
        T_margin:
            Strand temperature margin in operation [K]

        Return
        ------
            float [A]
        """
        return 0


class Wire_Nb3Sn(Strand):
    def __init__(self, d_strand: float = 0.82e-3):
        """
        Nb3Sn strand

        Parameters
        ----------
            d_strand:
                strand diameter [m]
        """
        copper_100 = Copper100()
        mat_Nb3Sn = Nb3Sn()
        materials = [copper_100, mat_Nb3Sn]
        percentage = [0.5, 0.5]
        super().__init__(materials=materials, percentage=percentage, d_strand=d_strand)

    def Ic(
            self, B: float, T: float, strain: float = 0.55, T_margin: float = 1.5
    ) -> float:
        """
        Nb3Sn critical current from Jc(B,T,strain) Parameterization for the ITER Nb3Sn Production

        Ref
        ---
        fit from IEEE TRANSACTIONS ON APPLIED SUPERCONDUCTIVITY, VOL. 19, NO. 3, JUNE 2009
        Luca Bottura and Bernardo Bordini

        note
        ----
        fit parameters form WST strand,A. Nijhuis, “TF conductor samples strand thermo mechanical critical
        performances tests”, https://idm.euro-fusion.org/?uid=2M5SMM v1.0.

        Parameters
        ----------
        T:
            Operating temperature [K]
        B:
            Operating magnetic field [T]
        strain:
            total applied measured strain [%]
        T_margin:
            Strand temperature margin in operation [K]
        Return
        ------
            float [A]
        """
        d_ = self.d_strand * 1e3
        CunonCu = self.percentage[0] / self.percentage[1]
        strand_A = np.pi * d_ ** 2 / (4 * (1 + CunonCu))
        c_ = 1.0
        Ca1 = 50.06
        Ca2 = 0.00
        eps_0a = 0.00312
        eps_m = -0.00059
        Bc20max = 33.24
        Tc0max = 16.34
        p = 0.593
        q = 2.156
        C = 83075 * strand_A  # [AT]
        T_ = T + T_margin
        int_eps = -strain / 100 + eps_m
        eps_sh = Ca2 * eps_0a / (np.sqrt(Ca1 ** 2 - Ca2 ** 2))
        s_eps = 1 + (
                Ca1
                * (
                        np.sqrt(eps_sh ** 2 + eps_0a ** 2)
                        - np.sqrt((int_eps - eps_sh) ** 2 + eps_0a ** 2)
                )
                - Ca2 * int_eps
        ) / (1 - Ca1 * eps_0a)
        Bc0_eps = Bc20max * s_eps
        Tc0_eps = Tc0max * (s_eps) ** (1 / 3)
        t = T_ / Tc0_eps
        BcT_eps = Bc0_eps * (1 - t ** (1.52))
        b = B / BcT_eps
        hT = (1 - t ** (1.52)) * (1 - t ** 2)
        fPb = (b ** p) * (1 - b) ** q
        return c_ * (C / B) * s_eps * fPb * hT

    def Je(self, Ic: float):
        """
        Strand current density

        Parameters
        ----------
        Ic:
            Critical current [A]

        Return
        ------
             float [A/m2]
        """
        return Ic / self.area


class Wire_NbTi(Strand):
    def __init__(self, d_strand: float = 0.82e-3):
        """
        NbTi strand

        Parameters
        ----------
        d_strand:
            strand diameter [m]
        """
        copper_100 = Copper100()
        mat_NbTi = NbTi()
        materials = [copper_100, mat_NbTi]
        percentage = [0.5, 0.5]
        super().__init__(materials=materials, percentage=percentage, d_strand=d_strand)

    def Ic(self, B: float, T: float, T_margin: float = 1.5, **kwargs):
        """
        NbTi critical current

        Ref
        ---
        Pinning Properties of Commercial Nb-Ti Wires Described by a 2-Components Model,
        Luigi Muzzi, Gianluca De Marzi, et al.

        Note
        ----
        Fit data from DTT TF strand

        Parameters
        ----------
        T:
            Operating temperature [K]
        B:
            Operating magnetic field [T]
        T_margin:
            Strand temperature margin in operation [K]
        Return
        ------
            float [A]
        """
        d_ = self.d_strand * 1e3
        CunonCu = self.percentage[0] / self.percentage[1]
        n = 1.7
        Bc20_T = 15.19
        Tc0_K = 8.907
        t = (T + T_margin) / Tc0_K
        tt = 1 - t ** n
        b = B / Bc20_T
        C0 = 3.00e04
        C1 = 0.45
        a1 = 3.2
        b1 = 2.43
        C2 = 0.55
        a2 = 0.65
        b2 = 2
        g1 = 1.8
        g2 = 1.8
        G = (a1 / (a1 + b1)) ** a1
        GG = (b1 / (a1 + b1)) ** b1
        GGG = G * GG
        F = (a2 / (a2 + b2)) ** a2
        FF = (b2 / (a2 + b2)) ** b2
        FFF = F * FF
        Jc = (
                C0 * C1 / (B * GGG) * (b / tt) ** a1 * (1 - b / tt) ** b1 * tt ** g1
                + C0 * C2 / (B * FFF) * (b / tt) ** a2 * (1 - b / tt) ** b2 * tt ** g2
        )
        return Jc * np.pi * d_ ** 2 / (4 * (1 + CunonCu))

    def Je(self, Ic: float):
        """
        Strand current density

        Parameters
        ----------
        Ic:
            Critical current [A]

        Return
        ------
             float [A/m2]
        """
        return Ic / self.area
