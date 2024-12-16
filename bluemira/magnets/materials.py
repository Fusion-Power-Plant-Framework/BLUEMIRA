# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Material database used for magnets
"""

import math
from abc import ABC, abstractmethod


class ABCMaterial(ABC):
    """Abstract Material class"""

    @abstractmethod
    def density(self, **kwargs):
        """Density"""
        pass

    @abstractmethod
    def E(self, **kwargs):  # noqa: N802
        """Young's module"""
        pass

    @abstractmethod
    def erho(self, **kwargs):
        """Electrical Resistivity"""
        pass

    @abstractmethod
    def cp_v(self, **kwargs):
        """Specific heat (constant volume)"""
        pass


class Material(ABCMaterial):
    """Generic Material"""

    def density(self, **kwargs):  # noqa: ARG002
        """Density"""
        return 0

    def E(self, **kwargs):  # noqa: N802, ARG002
        """Young's module"""
        return 1e-6

    def erho(self, **kwargs):  # noqa: ARG002
        """Electrical Resistivity"""
        return 1e6

    def cp_v(self, **kwargs):  # noqa: ARG002
        """Specific heat (constant volume)"""
        return 0


class AISI_316LN(Material):  # noqa: N801
    """
    Stainless steel (316 LN, with extra low carbon content, aged)

    Ref
    ---
    EFDA Material Data Compilation for Superconductor Simulation
    P. Bauer, H. Rajainmaki, E. Salpietro
    EFDA CSU, Garching, 04/18/07

    """

    def density(self, **kwargs):  # noqa: ARG002
        """Material density [kg/m³]"""
        return 7890.0

    def E(self, T: float, **kwargs):  # noqa: N802, N803, ARG002
        """
        Young's module

        Ref
        ---
        Data fits from N. Mitchell, “Finite element simulations of elasto-plastic
        processes” Cryogenics 45 (2005) 501–515

        Parameters
        ----------
        T:
            Operating temperature [K]

        Returns
        -------
            float [Pa]
        """
        if T > 173:
            E = 200.4 * 1e9 - 8.1221e-2 * (T - 273) * 1e9
        else:
            E = 208.5 * 1e9
        return E

    def erho(self, T: float, **kwargs):  # noqa: N803
        """
        Electrical Resistivity

        Ref
        ---
        Fit of data quoted by J. Davis in ITER Material Properties Handbook,
        ITER Document No. S 74 RE 1

        Parameters
        ----------
        T:
            Operating temperature [K]

        Return
        ------
            float [Ohm m]
        """
        return (
                76.2063 + (0.071375 * (T - 273.15)) - (2.3109 * 1e-5 * (T - 273.15) ** 2)
        ) * 1e-8


class Copper100(Material):
    def __init__(self):
        """
        Copper (OFHC, high purity RRR~100, annealed)

        Ref
        ---
        EFDA Material Data Compilation for Superconductor Simulation
        P. Bauer, H. Rajainmaki, E. Salpietro
        EFDA CSU, Garching, 04/18/07

        """
        # cp300: Specific heat known data-point at 300 K [J/K/m³]
        self.cp300 = 3.454e6
        # gamma: Debye fit parameter for Cp calculation [J/K**2/kg]
        self.gamma = 0.011
        # beta: Grueneisen fit parameter for Cp calculation [J/K**4/kg]
        self.beta = 0.0011
        # RRR: Residual-resistance ratio (ratio of the resistivity of a material at
        # room temperature and at 0 K)
        self.RRR = 100

    def density(self, **kwargs):
        """Material density [kg/m³]"""
        return 8960

    def erho(self, T: float, B: float, **kwargs):
        """
        Electrical Resistivity

        Ref
        ---
        NIST MONOGRAPH 177 J.Simon, E.S.Drexler and R.P.Reed, "Properties of
        Copper and Copper Alloys at Cryogenic Temperatures", 850 pages, February 1992,
        U.S. Government Printing Office, Washington, DC 20402-9325

        Parameters
        ----------
        T:
            Operating temperature [K]
        B:
            Operating magnetic field [T]

        Return
        ------
            float  [Ohm m]
        """
        rho1 = (1.171 * (10 ** -17) * (T ** 4.49)) / (
                1 + (4.5 * (10 ** -7) * (T ** 3.35) * (math.exp(-((50.0 / T) ** 6.428))))
        )
        rho2 = (
                (1.69 * (10 ** -8) / self.RRR)
                + rho1
                + 0.4531 * ((1.69 * (10 ** -8) * rho1) / (
                    self.RRR * rho1 + 1.69 * (10 ** -8)))
        )
        A = math.log10(1.553 * (10 ** -8) * B / rho2)
        a = (
                -2.662
                + (0.3168 * A)
                + (0.6229 * (A ** 2))
                - (0.1839 * (A ** 3))
                + (0.01827 * (A ** 4))
        )

        return rho2 * (1 + (10 ** a))

    def cp_v(self, T: float, **kwargs):
        """
        Specific heat

        Ref
        ---
        L. Dresner, “Stability of Superconductors”, Plenum Press, NY, 1995

        Note
        ----
        The specific heat over the whole temperature range is obtained through
        fitting the function to the known low temperature and high temperature data.

        Parameters
        ----------
        T:
            Operating temperature [K]
        Return
        ------
            float [J/K/m³]
        """
        density = self.density(T=T, **kwargs)
        c_plow = (self.beta * (T ** 3)) + (self.gamma * T)
        return 1 / ((1 / self.cp300) + (1 / (c_plow * density)))


class Copper300(Material):
    def __init__(self):
        """
        Copper (OFHC, high purity RRR~100, annealed)

        Ref
        ---
        EFDA Material Data Compilation for Superconductor Simulation
        P. Bauer, H. Rajainmaki, E. Salpietro
        EFDA CSU, Garching, 04/18/07

        """
        # cp300: Specific heat known data-point at 300 K [J/K/m³]
        self.cp300 = 3.454e6
        # gamma: Debye fit parameter for Cp calculation [J/K**2/kg]
        self.gamma = 0.011
        # beta: Grueneisen fit parameter for Cp calculation [J/K**4/kg]
        self.beta = 0.0011
        # RRR: Residual-resistance ratio (ratio of the resistivity of a material at
        # room temperature and at 0 K)
        self.RRR = 300

    def density(self, **kwargs):
        """Material density [kg/m³]"""
        return 8960

    def erho(self, T: float, B: float, **kwargs):
        """
        Electrical Resistivity

        Ref
        ---
        NIST MONOGRAPH 177 J.Simon, E.S.Drexler and R.P.Reed, "Properties of
        Copper and Copper Alloys at Cryogenic Temperatures", 850 pages, February 1992,
        U.S. Government Printing Office, Washington, DC 20402-9325

        Parameters
        ----------
        T:
            Operating temperature [K]
        B:
            Operating magnetic field [T]

        Return
        ------
            float  [Ohm m]
        """
        rho1 = (1.171 * (10 ** -17) * (T ** 4.49)) / (
                1 + (4.5 * (10 ** -7) * (T ** 3.35) * (math.exp(-((50 / T) ** 6.428))))
        )
        rho2 = (
                (1.69 * (10 ** -8) / self.RRR)
                + rho1
                + 0.4531 * ((1.69 * (10 ** -8) * rho1) / (
                    self.RRR * rho1 + 1.69 * (10 ** -8)))
        )
        A = math.log10(1.553 * (10 ** -8) * B / rho2)
        a = (
                -2.662
                + (0.3168 * A)
                + (0.6229 * (A ** 2))
                - (0.1839 * (A ** 3))
                + (0.01827 * (A ** 4))
        )
        return rho2 * (1 + (10 ** a))

    def cp_v(self, T: float, **kwargs):
        """
        Specific heat

        Ref
        ---
        L. Dresner, “Stability of Superconductors”, Plenum Press, NY, 1995

        Note
        ----
        The specific heat over the whole temperature range is obtained through
        fitting the function to the known low temperature and high temperature data.

        Parameters
        ----------
        T:
            Operating temperature [K]
        Return
        ------
            float [J/K/m³]
        """
        density = self.density(T=T, **kwargs)
        c_plow = (self.beta * (T ** 3)) + (self.gamma * T)
        return 1 / ((1 / self.cp300) + (1 / (c_plow * density)))


class Nb3Sn(Material):
    def __init__(self):
        """
        Nb3Sn superconductor (polycrystalline, normal state)

        Ref
        ---
        EFDA Material Data Compilation for Superconductor Simulation
        P. Bauer, H. Rajainmaki, E. Salpietro
        EFDA CSU, Garching, 04/18/07


        """
        # cp300: Specific heat known data-point at 300 K [J/K/m³]
        self.cp300 = 210
        # gamma: Debye fit parameter for Cp calculation [J/K**2/kg]
        self.gamma = 0.1
        # beta: Grueneisen fit parameter for Cp calculation [J/K**4/kg]
        self.beta = 0.001

    def density(self, **kwargs):
        """Material density [kg/m³]"""
        return 8040

    def cp_v(self, T: float, **kwargs):
        """
        Nb3Sn specific heat

        Ref
        ---
        ITER DRG1 Annex, Superconducting Material Database, Article 5, N 11 FDR
        42 01-07-05 R 0.1, Thermal, Electrical and Mechanical Properties of Materials
        at Cryogenic Temperatures

        Note
        ----
            Most specific heat data are listed in J/K/kg units. To represent the data
            in the J/K/m3 units a density of 8040 kg/m3 was assumed.

        Parameters
        ----------
        T:
            Operating temperature [K]

        Return
        ------
            float [J/K/m³]
        """
        density = self.density(T=T, **kwargs)
        cp_low_NC = (self.beta * (T ** 3)) + (self.gamma * T)
        cp_Nb3Sn = 1 / ((1 / self.cp300) + (1 / cp_low_NC))
        return cp_Nb3Sn * density

    def erho(self, T: float, **kwargs):
        """
        Electrical Resistivity

        Ref
        ---
        Data collected by L. Rossi, M. Sorbi, “MATPRO: A Computer Library of Material
        Property at Cryogenic Temperature”, INFN/TC-06/02, CARE-Note-2005-018-HHH

        Parameters
        ----------
        T:
            Operating temperature [K]

        Return
        ------
            float  [Ohm m]
        """
        return (-1e-4 * T ** 2 + 0.0938 * T + 22.601) * 1e-8


class NbTi(Material):
    def __init__(self):
        """
        NbTi(46.5%) (polycrystalline, normal state)

        Ref
        ---
        EFDA Material Data Compilation for Superconductor Simulation
        P. Bauer, H. Rajainmaki, E. Salpietro
        EFDA CSU, Garching, 04/18/07

        """
        # cp300: Specific heat known data-point at 300 K [J/K/m³]
        self.cp300 = 400
        # gamma: Debye fit parameter for Cp calculation [J/K**2/kg]
        self.gamma = 0.145
        # beta: Grueneisen fit parameter for Cp calculation [J/K**4/kg]
        self.beta = 0.0023

    def density(self, **kwargs):
        """Material density [kg/m³]"""
        return 6000

    def cp_v(self, T: float, **kwargs):
        """
        NbTi Specific Heat

        Ref
        ---
        Elrod S.A. Miller J.R., Dresner L., “The specific heat of NbTi from 0-7T between
        4.2 and 20K”, Advances in Cryogenic Engineering Materials, Vol. 28, 1981

        Note
        ----
            Most specific heat data are listed in J/K/kg units. To represent the data
            in the J/K/m3 units a density of 8570 kg/m3 was assumed.

        Parameters
        ----------
        T:
            Operating temperature [K]

        Return
        ------
            float [J/K/m³]
        """
        density = self.density(T=T, **kwargs)
        cp_low_NC = (self.beta * (T ** 3)) + (self.gamma * T)
        cp_NbTi = 1 / ((1 / self.cp300) + (1 / cp_low_NC))
        return cp_NbTi * density

    def erho(self, T: float, **kwargs):
        """
        Electrical Resistivity

        Ref
        ---
        data-point discussed in Larbalestier’s article in 'Superconducting Material
        Properties' is 50 mWm, at 10 K. This value is 2 times higher than those from
        MATPRO. We therefore suggest the use of the CRYOCOMP resistivities, which can
        be fitted with the following polynomial

        Parameters
        ----------
        T:
            Operating temperature [K]

        Return
        ------
            float  [Ohm m]
        """
        return (0.0558 * T + 55.668) * 1e-8


class DummyInsulator(Material):
    """A dummy insulator"""

    def erho(self, **kwargs):
        return 1e6

    def E(self, **kwargs):
        """Young's module"""
        return 12e9
