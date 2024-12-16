# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Cable class"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from bluemira.base.look_and_feel import bluemira_error, bluemira_print
from bluemira.magnets.materials import Material
from bluemira.magnets.strand import Strand, SuperconductingStrand
from bluemira.magnets.utils import parall_r, serie_r


class ABCCable(Material, ABC):
    def __init__(
        self,
        sc_strand: SuperconductingStrand,
        stab_strand: Strand,
        n_sc_strand: int,
        n_stab_strand: int,
        d_cooling_channel: float,
        void_fraction: float = 0.725,
        cos_theta: float = 0.97,
        name: str = "",
    ):
        """
        Representation of a cable. Only the x-dimension of the cable is given as
        input. The y-dimension is calculated on the basis of the cable design.

        Notes
        -----
        Cooling material not implemented.

        Parameters
        ----------
        sc_strand:
            strand of the superconductor
        stab_strand:
            strand of the stabilizer
        d_cooling_channel:
            diameter of the cooling channel
        n_sc_strand:
            number of superconducting strands
        n_stab_strand:
            number of stabilizer strands
        void_fraction:
            void fraction defined as material_volume/total_volume
        cos_theta:
            corrective factor that consider the twist of the cable
        name:
            cable string identifier
        """
        # initialize private variables
        self._d_cooling_channel = None
        self._void_fraction = None
        self._n_sc_strand = None
        self._n_stab_strand = None
        self._cos_theta = None
        self._shape = None

        # assign
        self.name = name
        self.sc_strand = sc_strand
        self.stab_strand = stab_strand
        self.void_fraction = void_fraction
        self.d_cooling_channel = d_cooling_channel
        self.n_sc_strand = n_sc_strand
        self.n_stab_strand = n_stab_strand
        self.cos_theta = cos_theta

    @property
    @abstractmethod
    def dx(self):
        pass

    @property
    @abstractmethod
    def dy(self):
        pass

    @property
    def aspect_ratio(self):
        return self.dx / self.dy

    @property
    def n_sc_strand(self):
        """Number of stabilizing strands"""
        return self._n_sc_strand

    @n_sc_strand.setter
    def n_sc_strand(self, value: int):
        if value < 0:
            msg = f"The number of superconducting strands must be positive, got {value}"
            bluemira_error(msg)
            raise ValueError(msg)
        self._n_sc_strand = int(np.ceil(value))

    @property
    def n_stab_strand(self):
        """Number of stabilizing strands"""
        return self._n_stab_strand

    @n_stab_strand.setter
    def n_stab_strand(self, value: int):
        if value < 0:
            msg = f"The number of stabilizing strands must be positive, got {value}"
            bluemira_error(msg)
            raise ValueError(msg)
        self._n_stab_strand = int(np.ceil(value))

    @property
    def d_cooling_channel(self):
        return self._d_cooling_channel

    @d_cooling_channel.setter
    def d_cooling_channel(self, value: float):
        if value < 0:
            msg = f"diameter of the cooling channel must be positive, got {value}"
            bluemira_error(msg)
            raise ValueError(msg)

        self._d_cooling_channel = value

    @property
    def void_fraction(self):
        return self._void_fraction

    @void_fraction.setter
    def void_fraction(self, value: float):
        if value < 0 or value > 1:
            msg = f"void_fraction must be between 0 and 1, got {value}"
            bluemira_error(msg)
            raise ValueError(msg)

        self._void_fraction = value

    @property
    def cos_theta(self):
        return self._cos_theta

    @cos_theta.setter
    def cos_theta(self, value: float):
        if value <= 0 or value > 1:
            msg = f"cos theta must be in the interval ]0, 1], got {value}"
            bluemira_error(msg)
            raise ValueError(msg)

        self._cos_theta = value

    def erho(self, **kwargs):
        """
        Computes the cable's equivalent resistivity considering the resistance
        of its strands in parallel.

        Parameters
        ----------
        **kwargs: dict
            Additional parameters for resistance calculations.

        Return
        ------
            float [Ohm m]
        """
        resistances = np.array([
            self.sc_strand.erho(**kwargs) / self.area_sc,
            self.stab_strand.erho(**kwargs) / self.area_stab,
        ])
        res_tot = parall_r(resistances)
        return res_tot * self.area

    def cp_v(self, **kwargs):
        """
        Computes the cable's equivalent specific heat considering the specific heats
        of its strands in series.

        Parameters
        ----------
        **kwargs: dict
            Additional parameters for specific heat calculations.

        Return
        ------
            float [J/K/m]
        """
        weighted_specific_heat = np.array([
            self.sc_strand.cp_v(**kwargs) * self.area_sc,
            self.stab_strand.cp_v(**kwargs) * self.area_stab,
        ])
        return serie_r(weighted_specific_heat) / (self.area_sc + self.area_stab)

    @property
    def area_stab(self):
        """Area of the stabilizer region"""
        return self.stab_strand.area * self.n_stab_strand

    @property
    def area_sc(self):
        """Area of the superconductor region"""
        return self.sc_strand.area * self.n_sc_strand

    @property
    def area_cc(self):
        """Area of the cooling channel"""
        return self.d_cooling_channel**2 / 4 * np.pi

    @property
    def area(self):
        """Area of the cable considering the void fraction"""
        return (
            self.area_sc + self.area_stab
        ) / self.void_fraction / self.cos_theta + self.area_cc

    def E(self, **kwargs):
        """Young's moduli"""
        return 0

    def _heat_balance_model_cable(self, t: float, T: float, B: Callable, I: Callable):
        """
        Calculate the derivative of temperature (dT/dt) for a 0D heat balance problem.

        Parameters
        ----------
            t : float
                The current time in seconds.
            T : float
                The current temperature in Celsius.
            B : Callable
                The magnetic field [T] as time function
            I : Callable
                The current [A] flowing through the conductor as time function
            cable : Cable
                the superconducting cable

        Returns
        -------
            dTdt : float
                The derivative of temperature with respect to time (dT/dt).
        """
        # Calculate the rate of heat generation (Joule dissipation)
        if isinstance(T, np.ndarray):
            T = T[0]

        Q_gen = (I(t) / self.area) ** 2 * self.erho(B=B(t), T=T)

        # Calculate the rate of heat absorption by conductor components
        Q_abs = self.cp_v(T=T)

        # Calculate the derivative of temperature with respect to time (dT/dt)
        dTdt = Q_gen / Q_abs

        return dTdt

    def optimize_n_stab_ths(
        self,
        t0: float,
        tf: float,
        initial_temperature: float,
        target_temperature: float,
        B: Callable,
        I: Callable,
        bounds: np.ndarray = None,
        show: bool = False,
    ):
        """
        Optimize the number of stabilizer strand in the superconducting cable using a
        0-D hot spot criteria.

        Parameters
        ----------
            t0:
                initial time
            tf:
                final time
            initial_temperature:
                temperature [K] at initial time
            target_temperature:
                target temperature [K] at final time
            B :
                The magnetic field [T] as time function
            I :
                The current [A] flowing through the conductor as time function
            bounds:
                lower and upper limits for the number of strand in the cable
            show:
                if True the behavior of temperature as function of time is plotted

        Returns
        -------
            None

        Notes
        -----
            The number of stabilizer strands in the cable is directly modified. An
            error is raised in case the optimization process did not converge.
            Cooling material contribution is neglected when applying the hot spot criteria.
        """

        def _temperature_evolution(
            t0: float,
            tf: float,
            initial_temperature: float,
            B: Callable,
            I: Callable,
        ):
            solution = solve_ivp(
                self._heat_balance_model_cable,
                [t0, tf],
                [initial_temperature],
                args=(B, I),
                dense_output=True,
            )

            if not solution.success:
                raise ValueError("Temperature evolution did not converged")

            return solution

        def final_temperature_difference(
            n_stab: int,
            t0: float,
            tf: float,
            initial_temperature: float,
            target_temperature: float,
            B: Callable,
            I: Callable,
        ):
            self.n_stab_strand = n_stab

            solution = _temperature_evolution(
                t0=t0,
                tf=tf,
                initial_temperature=initial_temperature,
                B=B,
                I=I,
            )
            final_T = float(solution.y[0][-1])
            diff = abs(final_T - target_temperature)
            return diff

        method = None
        if bounds is not None:
            method = "bounded"

        result = minimize_scalar(
            fun=final_temperature_difference,
            args=(t0, tf, initial_temperature, target_temperature, B, I),
            bounds=bounds,
            method=method,
        )

        if not result.success:
            raise ValueError(
                "n_stab optimization did not converge. Check your input parameters or initial bracket."
            )

        solution = _temperature_evolution(t0, tf, initial_temperature, B, I)
        final_temperature = solution.y[0][-1]

        bluemira_print(f"Optimal n_stab: {self.n_stab_strand}")
        bluemira_print(
            f"Final temperature with optimal n_stab: {final_temperature} Kelvin"
        )

        if show:
            # _, ax = plt.subplots()
            # ax.plot(solution.t, solution.y[0], "r*")
            # time_steps = np.linspace(t0, tf, 100)
            # ax.plot(time_steps, solution.sol(time_steps)[0], "b")
            # plt.grid(True)
            # plt.xlabel("Time [s]")
            # plt.ylabel("Temperature [K]")
            # plt.title("Quench temperature evoltuion")
            #
            # # *** Additional info ***
            # additional_info = [f"Hot spot temp. = {target_temperature} [K]",
            #                    f"Initial temp. = {initial_temperature} [K]",
            #                    f"Sc. strand = {self.sc_strand.__class__.__name__}",
            #                    f"n. sc. strand = {self.n_sc_strand}",
            #                    f"Stab. strand = {self.stab_strand.__class__.__name__}",
            #                    f"n. stab. strand = {self.n_stab_strand}"]
            #
            # additional_info = '\n'.join(additional_info)
            # plt.text(50, 80, additional_info)
            # plt.show()

            _, ax = plt.subplots()

            # Plot the main solution
            ax.plot(solution.t, solution.y[0], "r*")
            time_steps = np.linspace(t0, tf, 100)
            ax.plot(time_steps, solution.sol(time_steps)[0], "b")
            plt.grid(True)
            plt.xlabel("Time [s]")
            plt.ylabel("Temperature [K]")
            plt.title("Quench temperature evolution")

            # Create secondary axis on the right (no ticks or labels needed)
            ax2 = ax.twinx()  # This creates a new y-axis that shares the same x-axis
            ax2.set_yticks([])  # Remove y-axis ticks
            ax2.set_ylabel("")  # Remove y-axis label

            # Plot additional info next to the right y-axis
            additional_info = [
                f"Hot spot temp. = {target_temperature} [K]",
                f"Initial temp. = {initial_temperature} [K]",
                f"Sc. strand = {self.sc_strand.__class__.__name__}",
                f"n. sc. strand = {self.n_sc_strand}",
                f"Stab. strand = {self.stab_strand.__class__.__name__}",
                f"n. stab. strand = {self.n_stab_strand}",
            ]
            additional_info = "\n".join(additional_info)

            # Set text position right after the right y-axis
            ax2.text(
                1.05,
                0.5,
                additional_info,
                transform=ax2.transAxes,
                verticalalignment="center",
                fontsize=10,
            )

            plt.show()

        return result

    # OD homogenized structural properties
    @abstractmethod
    def Kx(self, **kwargs):
        """Total equivalent stiffness along x-axis"""
        pass

    @abstractmethod
    def Ky(self, **kwargs):
        """Total equivalent stiffness along y-axis"""
        pass

    def plot(self, xc: float = 0, yc: float = 0, show: bool = False, ax=None):
        """
        Schematic plot of the cable cross-section.

        Parameters
        ----------
        xc:
            x coordinate of the cable center in the considered coordinate system
        yc:
            y coordinate of the cable center in the considered coordinate system
        show:
            if True, the plot is displayed
        ax:
            Matplotlib Axis on which the plot shall be displayed. If None,
            a new figure is created
        """
        if ax is None:
            _, ax = plt.subplots()

        pc = np.array([xc, yc])
        a = self.dx / 2
        b = self.dy / 2

        p0 = np.array([-a, -b])
        p1 = np.array([a, -b])
        p2 = np.array([[a, b]])
        p3 = np.array([-a, b])

        points_ext = np.vstack((p0, p1, p2, p3, p0)) + pc
        points_cc = (
            np.array([
                np.array([np.cos(theta), np.sin(theta)]) * self.d_cooling_channel / 2
                for theta in np.linspace(0, np.radians(360), 19)
            ])
            + pc
        )

        ax.fill(points_ext[:, 0], points_ext[:, 1], "gold")
        ax.fill(points_cc[:, 0], points_cc[:, 1], "r")
        ax.set_aspect("equal")

        if show:
            plt.show()
        return ax


class RectangularCable(ABCCable):
    def __init__(
        self,
        dx: float,
        sc_strand: SuperconductingStrand,
        stab_strand: Strand,
        n_sc_strand: int,
        n_stab_strand: int,
        d_cooling_channel: float,
        void_fraction: float = 0.725,
        cos_theta: float = 0.97,
        name: str = "",
    ):
        """
        Representation of a cable. Only the x-dimension of the cable is given as
        input. The y-dimension is calculated on the basis of the cable design.

        Notes
        -----
        Cooling material not implemented.

        Parameters
        ----------
        dx:
            x-dimension of the cable [m]
        sc_strand:
            strand of the superconductor
        stab_strand:
            strand of the stabilizer
        d_cooling_channel:
            diameter of the cooling channel
        n_sc_strand:
            number of superconducting strands
        n_stab_strand:
            number of stabilizer strands
        void_fraction:
            void fraction defined as material_volume/total_volume
        cos_theta:
            corrective factor that consider the twist of the cable
        name:
            cable string identifier
        """
        super().__init__(
            sc_strand=sc_strand,
            stab_strand=stab_strand,
            n_sc_strand=n_sc_strand,
            n_stab_strand=n_stab_strand,
            d_cooling_channel=d_cooling_channel,
            void_fraction=void_fraction,
            cos_theta=cos_theta,
            name=name,
        )

        # initialize private variables
        self._dx = None

        # assign
        self.dx = dx

    @property
    def dx(self):
        """Cable dimension in the x direction [m]"""
        return self._dx

    @dx.setter
    def dx(self, value: float):
        if value < 0:
            msg = "dx must be positive"
            bluemira_error(msg)
            raise ValueError(msg)
        self._dx = value

    @property
    def dy(self):
        """Cable dimension in the y direction [m]"""
        return self.area / self.dx

    # Todo: decide if this function shall be a setter.
    #       Defined as "normal" function to underline that it modifies dx.
    def set_aspect_ratio(self, value: float) -> None:
        """Modify dx in order to get the given aspect ratio"""
        self.dx = np.sqrt(value * self.area)

    # OD homogenized structural properties
    def Kx(self, **kwargs):
        """Total equivalent stiffness along x-axis"""
        return self.E(**kwargs) * self.dy / self.dx

    def Ky(self, **kwargs):
        """Total equivalent stiffness along y-axis"""
        return self.E(**kwargs) * self.dx / self.dy


class DummyRectangularCableHTS(RectangularCable):
    """
    Dummy rectangular cable with young's moduli set to 120 GPa.

    Parameters
    ----------
    dx:
        x-dimension of the cable [m]
    sc_strand:
        strand of the superconductor
    stab_strand:
        strand of the stabilizer
    d_cooling_channel:
        diameter of the cooling channel
    n_sc_strand:
        number of superconducting strands
    n_stab_strand:
        number of stabilizer strands
    void_fraction:
        void fraction defined as material_volume/total_volume
    cos_theta:
        corrective factor that consider the twist of the cable
    name:
        cable string identifier

    """

    def E(self, **kwargs):
        """Young's module"""
        return 120e9


class DummyRectangularCableLTS(RectangularCable):
    """
    Dummy square cable with young's moduli set to 0.1 GPa

    Parameters
    ----------
    dx:
        x-dimension of the cable [m]
    sc_strand:
        strand of the superconductor
    stab_strand:
        strand of the stabilizer
    d_cooling_channel:
        diameter of the cooling channel
    n_sc_strand:
        number of superconducting strands
    n_stab_strand:
        number of stabilizer strands
    void_fraction:
        void fraction defined as material_volume/total_volume
    cos_theta:
        corrective factor that consider the twist of the cable
    name:
        cable string identifier
    """

    def E(self, **kwargs):
        """Young's module"""
        return 0.1e9


class SquareCable(ABCCable):
    def __init__(
        self,
        sc_strand: SuperconductingStrand,
        stab_strand: Strand,
        n_sc_strand: int,
        n_stab_strand: int,
        d_cooling_channel: float,
        void_fraction: float = 0.725,
        cos_theta: float = 0.97,
        name: str = "",
    ):
        """
        Representation of a square cable.

        Notes
        -----
        No geometrical dimensions are given. They are extrapolated from the cable design parameters.

        Parameters
        ----------
        sc_strand:
            strand of the superconductor
        stab_strand:
            strand of the stabilizer
        d_cooling_channel:
            diameter of the cooling channel
        n_sc_strand:
            number of superconducting strands
        n_stab_strand:
            number of stabilizer strands
        void_fraction:
            void fraction defined as material_volume/total_volume
        cos_theta:
            corrective factor that consider the twist of the cable
        name:
            cable string identifier

        Notes
        -----
        Cooling material not implemented
        """
        super().__init__(
            sc_strand=sc_strand,
            stab_strand=stab_strand,
            n_sc_strand=n_sc_strand,
            n_stab_strand=n_stab_strand,
            d_cooling_channel=d_cooling_channel,
            void_fraction=void_fraction,
            cos_theta=cos_theta,
            name=name,
        )

    @property
    def dx(self):
        """Cable dimension in the x direction [m]"""
        return np.sqrt(self.area)

    @property
    def dy(self):
        """Cable dimension in the y direction [m]"""
        return self.dx

    # OD homogenized structural properties
    def Kx(self, **kwargs):
        """Total equivalent stiffness along x-axis"""
        return self.E(**kwargs) * self.dy / self.dx

    def Ky(self, **kwargs):
        """Total equivalent stiffness along y-axis"""
        return self.E(**kwargs) * self.dx / self.dy


class DummySquareCableHTS(SquareCable):
    """
    Dummy square cable with young's moduli set to 120 GPa

    Parameters
    ----------
    sc_strand:
        strand of the superconductor
    stab_strand:
        strand of the stabilizer
    d_cooling_channel:
        diameter of the cooling channel
    n_sc_strand:
        number of superconducting strands
    n_stab_strand:
        number of stabilizer strands
    void_fraction:
        void fraction defined as material_volume/total_volume
    cos_theta:
        corrective factor that consider the twist of the cable
    name:
        cable string identifier
    """

    def E(self, **kwargs):
        """Young's module"""
        return 120e9


class DummySquareCableLTS(SquareCable):
    """
    Dummy square cable with young's moduli set to 0.1 GPa

    Parameters
    ----------
    sc_strand:
        strand of the superconductor
    stab_strand:
        strand of the stabilizer
    d_cooling_channel:
        diameter of the cooling channel
    n_sc_strand:
        number of superconducting strands
    n_stab_strand:
        number of stabilizer strands
    void_fraction:
        void fraction defined as material_volume/total_volume
    cos_theta:
        corrective factor that consider the twist of the cable
    name:
        cable string identifier
    """

    def E(self, **kwargs):
        """Young's module"""
        return 0.1e9


class RoundCable(ABCCable):
    def __init__(
        self,
        sc_strand: SuperconductingStrand,
        stab_strand: Strand,
        n_sc_strand: int,
        n_stab_strand: int,
        d_cooling_channel: float,
        void_fraction: float = 0.725,
        cos_theta: float = 0.97,
        name: str = "",
    ):
        """
        Representation of a round cable

        Parameters
        ----------
        sc_strand:
            strand of the superconductor
        stab_strand:
            strand of the stabilizer
        d_cooling_channel:
            diameter of the cooling channel
        n_sc_strand:
            number of superconducting strands
        n_stab_strand:
            number of stabilizer strands
        void_fraction:
            void fraction defined as material_volume/total_volume
        cos_theta:
            corrective factor that consider the twist of the cable
        name:
            cable string identifier


        """
        super().__init__(
            sc_strand=sc_strand,
            stab_strand=stab_strand,
            n_sc_strand=n_sc_strand,
            n_stab_strand=n_stab_strand,
            d_cooling_channel=d_cooling_channel,
            void_fraction=void_fraction,
            cos_theta=cos_theta,
            name=name,
        )

    @property
    def dx(self):
        """Cable dimension in the x direction [m] (i.e. cable's diameter)"""
        return np.sqrt(self.area * 4 / np.pi)

    @property
    def dy(self):
        """Cable dimension in the y direction [m] (i.e. cable's diameter)"""
        return self.dx

    # OD homogenized structural properties
    # Todo: check if the rectangular approximation is fine also for this case
    def Kx(self, **kwargs):
        """Total equivalent stiffness along x-axis"""
        return self.E(**kwargs) * self.dy / self.dx

    def Ky(self, **kwargs):
        """Total equivalent stiffness along y-axis"""
        return self.E(**kwargs) * self.dx / self.dy

    def plot(self, xc: float = 0, yc: float = 0, show: bool = False, ax=None):
        """
        Schematic plot of the cable cross-section.

        Parameters
        ----------
        xc:
            x coordinate of the cable center in the considered coordinate system
        yc:
            y coordinate of the cable center in the considered coordinate system
        show:
            if True, the plot is displayed
        ax:
            Matplotlib Axis on which the plot shall be displayed. If None,
            a new figure is created
        """
        if ax is None:
            _, ax = plt.subplots()

        pc = np.array([xc, yc])

        points_ext = (
            np.array([
                np.array([np.cos(theta), np.sin(theta)]) * self.dx / 2
                for theta in np.linspace(0, np.radians(360), 19)
            ])
            + pc
        )

        points_cc = (
            np.array([
                np.array([np.cos(theta), np.sin(theta)]) * self.d_cooling_channel / 2
                for theta in np.linspace(0, np.radians(360), 19)
            ])
            + pc
        )

        ax.fill(points_ext[:, 0], points_ext[:, 1], "gold")
        ax.fill(points_cc[:, 0], points_cc[:, 1], "r")

        if show:
            plt.show()
        return ax


class DummyRoundCableHTS(RoundCable):
    """
    Dummy round cable with young's moduli set to 120 GPa

    Parameters
    ----------
    sc_strand:
        strand of the superconductor
    stab_strand:
        strand of the stabilizer
    d_cooling_channel:
        diameter of the cooling channel
    n_sc_strand:
        number of superconducting strands
    n_stab_strand:
        number of stabilizer strands
    void_fraction:
        void fraction defined as material_volume/total_volume
    cos_theta:
        corrective factor that consider the twist of the cable
    name:
        cable string identifier

    Notes
    -----
    Cooling material not implemented
    """

    def E(self, **kwargs):
        """Young's module"""
        return 120e9


class DummyRoundCableLTS(RoundCable):
    """
    Dummy round cable with young's moduli set to 120 GPa

    Parameters
    ----------
    sc_strand:
        strand of the superconductor
    stab_strand:
        strand of the stabilizer
    d_cooling_channel:
        diameter of the cooling channel
    n_sc_strand:
        number of superconducting strands
    n_stab_strand:
        number of stabilizer strands
    void_fraction:
        void fraction defined as material_volume/total_volume
    cos_theta:
        corrective factor that consider the twist of the cable
    name:
        cable string identifier

    Notes
    -----
    Cooling material not implemented
    """

    def E(self, **kwargs):
        """Young's module"""
        return 0.1e9
