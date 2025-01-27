# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Plasma profile objects, shape functions, and associated tools
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from eqdsk import EQDSKInterface
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from bluemira.base.constants import MU_0
from bluemira.equilibria.constants import BLUEMIRA_DEFAULT_COCOS
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.find import Opoint, find_LCFS_separatrix, in_plasma, in_zone
from bluemira.equilibria.grid import integrate_dx_dz, revolved_volume, volume_integral
from bluemira.equilibria.plotting import ProfilePlotter

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy.typing as npt

    from bluemira.equilibria.find import Xpoint

__all__ = [
    "BetaIpProfile",
    "CustomProfile",
    "DoublePowerFunc",
    "LaoPolynomialFunc",
    "LuxonExpFunc",
    "SinglePowerFunc",
]

# =============================================================================
# Flux function parameterisations
# =============================================================================


def fitfunc(
    func: Callable[[float], float],
    data: npt.NDArray[np.float64],
    order: int | None = None,
) -> npt.NDArray[np.float64]:
    """
    Uses scipy's curve_fit to fit 1-D data to a custom function

    Parameters
    ----------
    func:
        Function parameterisation to use in the fit
    data:
        Data to fit
    order:
        Order of function callable (if user defined)

    Returns
    -------
    :
        Optimised fitting parameters
    """
    x = np.linspace(0, 1, len(data))
    p0 = None if order is None else [1] * order
    popt, _ = curve_fit(func, x, data, p0=p0)
    return popt


@nb.jit(nopython=True)
def singlepowerfunc(x: float, *args) -> float:
    """
    Single power shape function defined e.g. CREATE stuff

    \t:math:`g(x)=(1-x^{n})`
    """  # noqa: DOC201
    n = args[0]
    return 1 - x**n


@nb.jit(nopython=True, cache=True)
def doublepowerfunc(x: float, *args) -> float:
    """
    Double power shape function defined e.g. in
    :doi:`Lao 1985 <10.1088/0029-5515/25/11/007>`

    \t:math:`g(x)=(1-x^{m})^{n}`
    """  # noqa: DOC201
    # sign tweak needed to avoid runtimewarnings in np
    m, n = args
    f = 1 - np.sign(x) * np.abs(x) ** m
    return np.sign(f) * (np.abs(f)) ** n


@nb.jit(cache=True, forceobj=True)
def pshape(
    shape: Callable[[float], float], psinorm: float, psio: float, psix: float
) -> float:
    """
    Integral of jtorshape to calculate pressure

    Notes
    -----
    Factor to convert from normalised psi integral
    """  # noqa: DOC201
    si = quad(shape, psinorm, 1, limit=100)[0]
    si *= psix - psio
    return si


@nb.jit(forceobj=True, looplift=True)  # Cannot cache due to "lifted loops"
def speedy_pressure(
    psi_norm: npt.NDArray[np.float64],
    psio: float,
    psix: float,
    shape: Callable[[float], float],
) -> npt.NDArray[np.float64]:
    """
    Calculate the pressure map on the psi_norm array without any masking because
    the plasma is not clearly bounded.

    Parameters
    ----------
    psi_norm:
        The normalised poloidal magnetic flux array
    psio:
        The psi value at the centre of the plasma (O-point)
    psix:
        The psi value at the edge of the plasma (X-point)
    shape:
        The shape function to use when calculating the pressure at each point

    Returns
    -------
    :
        The pressure on the grid
    """
    nx, nz = psi_norm.shape
    pfunc = np.zeros((nx, nz))
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            if (psi_norm[i, j] >= 0) and (psi_norm[i, j] < 1):
                pfunc[i, j] = pshape(shape, psi_norm[i, j], psio, psix)
    return pfunc


# TODO @je-cook: non-precise type pyobject with the below decorator
# 3582
# @nb.jit(cache=False, forceobj=True)  # Cannot cache due to "lifted loops"
def speedy_pressure_mask(
    ii: npt.NDArray[np.int64],
    jj: npt.NDArray[np.int64],
    psi_norm: npt.NDArray[np.float64],
    psio: float,
    psix: float,
    shape: Callable[[float], float],
) -> npt.NDArray[np.float64]:
    """
    Calculate the pressure map on the psi_norm array without any masking because
    the plasma is not clearly bounded.

    Parameters
    ----------
    ii:
        The i indices of the array to populate (dtype=int)
    jj:
        The j indices of the array to populate (dtype=int)
    psi_norm:
        The normalised poloidal magnetic flux array
    psio:
        The psi value at the centre of the plasma (O-point)
    psix:
        The psi value at the edge of the plasma (X-point)
    shape:
        The shape function to use when calculating the pressure at each point

    Returns
    -------
    :
        The pressure on the grid
    """
    pfunc = np.zeros_like(psi_norm, dtype=np.float64)
    for i, j in zip(ii, jj, strict=False):
        pfunc[i, j] = pshape(shape, psi_norm[i, j], psio, psix)
    return pfunc


# @nb.jit(cache=True)
def laopoly(x: float, *args) -> float:
    """
    Polynomial shape function defined in
    :doi:`Lao 1985 <10.1088/0029-5515/25/11/007>`
    \t:math:`g(x)=\\sum_{n=0}^{n_F} \\alpha_{n}x^{n}-`
    \t:math:`x^{n_F+1}\\sum_{n=0}^{n_F} \\alpha_{n}`
    """  # noqa: DOC201
    res = np.zeros_like(x)
    for i in range(len(args)):
        res += args[i] * x ** int(i)
    res -= sum(args) * x ** len(args)
    return res


@nb.jit(nopython=True, cache=True)
def luxonexp(x: float, *args) -> float:
    """
    Exponential shape function defined in
    :doi:`Luxon 1984 <10.1088/0029-5515/22/6/009>`
    \t:math:`g(x)=\\text{exp}\\big(-\\alpha^2x^2\\big)`
    """  # noqa: DOC201
    alpha = args[0]
    return np.exp(-(x**2) * alpha**2)


class ShapeFunction:
    """
    Shape function object
    """

    @classmethod
    def from_datafit(cls, data: npt.NDArray[np.float64], order: int | None = None):
        """
        Defines function from a dataset, fit using scipy curve_fit
        """  # noqa: DOC201
        if order is None:
            order = cls._order
        # Normalise data here
        data /= max(data)
        coeffs = fitfunc(cls._dfunc, data, order=order)
        cls.data = data
        return cls(coeffs)

    def _func(self, x: float) -> float:
        return self._dfunc(x, *self.coeffs)

    def __call__(self, x: float) -> float:
        """
        Calculate the value of the ShapeFunction for given x.
        """  # noqa: DOC201
        return self._fact * self._func(x)

    def adjust_parameters(self, coeffs: npt.NDArray[np.float64]):
        """
        Adjust the coefficients of the ShapeFunction
        """
        self.coeffs = coeffs

    def plot(self):
        """
        Plot the ShapeFunction
        """
        f, ax = plt.subplots()
        x = np.linspace(0, 1)
        ax.plot(x, self(x), label="Shape function - fitted")
        if hasattr(self, "data"):
            xd = np.linspace(0, 1, len(self.data))
            ax.plot(xd, self.data, label="Shape function - actual")
        ax.legend()
        f.show()

    def __mul__(self, a: float) -> ShapeFunction:
        """
        Multiply the ShapeFunction (adjust factor)
        """  # noqa: DOC201
        self._fact = a
        return self

    def __rmul__(self, a: float) -> ShapeFunction:
        """
        Multiply the ShapeFunction (adjust factor)
        """  # noqa: DOC201
        self._fact = a
        return self


class SinglePowerFunc(ShapeFunction):
    """
    Function object for a single power profile
    """

    _order = 1
    _fact = 1

    def __init__(self, args):
        if len(args) != self._order:
            raise ValueError(f"Function coefficients {len(args)} != {self._order}")
        self.coeffs = args

    @staticmethod
    def _dfunc(x: float, *args) -> float:
        return singlepowerfunc(x, *args)


class DoublePowerFunc(ShapeFunction):
    """
    Function object for a double power profile
    """

    _order = 2
    _fact = 1

    def __init__(self, args):
        if len(args) != self._order:
            raise ValueError(f"Function coefficients {len(args)} != {self._order}")
        self.coeffs = args

    @staticmethod
    def _dfunc(x: float, *args) -> float:
        return doublepowerfunc(x, *args)


class LaoPolynomialFunc(ShapeFunction):
    """
    Function object for a Lao polynomial profile
    """

    _fact = 1
    _order = 3

    def __init__(self, coeffs: npt.ArrayLike):
        if not hasattr(coeffs, "__len__"):
            self.n = 0
        self.n = len(coeffs) - 1
        self.coeffs = coeffs

    @staticmethod
    def _dfunc(x: float, *args) -> float:
        return laopoly(x, *args)


class LuxonExpFunc(ShapeFunction):
    """
    Function object for a Luxon exponential profile
    """

    _fact = 1
    _order = 1

    def __init__(self, coeffs: npt.ArrayLike):
        if not hasattr(coeffs, "__len__"):
            self.n = 1
            self.coeffs = [coeffs]
        else:
            raise ValueError("The Luxon function only has one coefficient.")

    @staticmethod
    def _dfunc(x: float, *args) -> float:
        return luxonexp(x, *args)


# =============================================================================
# Profile classes
# =============================================================================


class OPointCalcOptions(Enum):
    """
    O point estimation fallback options
    """

    RAISE = auto()
    GRID_CENTRE = auto()


class Profile:
    """
    Profile base class

    following some implementation in B. Dudson, FreeGS:
        https://github.com/bendudson/freegs
    """

    def _scalar_denorm(self, prime, norm):
        """
        Convert from integral in psi_norm to integral in psi
        """  # noqa: DOC201
        val = quad(prime, norm, 1)[0]
        return val * (self.psiax - self.psisep)

    @staticmethod
    def _reshape(psinorm):
        """
        Reshaping and array dimensioning utility
        """  # noqa: DOC201
        out = np.zeros_like(psinorm)
        p_vals = np.reshape(psinorm, -1)
        o_vals = np.reshape(out, -1)
        return p_vals, o_vals

    def pressure(self, psinorm: npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """
        Returns
        -------
        :
            Pressure as a function of normalised psi by integrating pprime
        """
        if not isinstance(psinorm, np.ndarray):
            return self._scalar_denorm(self.pprime, psinorm)

        p_vals, o_vals = self._reshape(psinorm)
        for i in range(len(p_vals)):
            o_vals[i] = self._scalar_denorm(self.pprime, p_vals[i])
        return np.reshape(o_vals, psinorm.shape)

    def fRBpol(self, psinorm: npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """
        Returns
        -------
        :
            f as a function of normalised psi
            :math:`FF^{'} = \\dfrac{1}{2}\\dfrac{F^{2}}{d\\psi}`

        Notes
        -----
        Apply a boundary condition:
        :math:`FF^{'}|_{\\substack{\\psi_{N}=1}} = (R_{0}B_{T,0})^{2}`
        """
        fvacuum = self.fvac()
        if not isinstance(psinorm, np.ndarray):
            val = self._scalar_denorm(self.ffprime, psinorm)
            return np.sqrt(2 * val + fvacuum**2)

        p_vals, o_vals = self._reshape(psinorm)

        for i in range(len(p_vals)):
            val = self._scalar_denorm(self.ffprime, p_vals[i])
            o_vals[i] = np.sqrt(2 * val + fvacuum**2)
        return np.reshape(o_vals, psinorm.shape)

    @staticmethod
    def _jtor(
        x: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        psi: npt.NDArray[np.float64],
        o_points: list[Opoint],
        x_points: list[Xpoint],
        lcfs: npt.NDArray[np.float64] | None = None,
        *,
        o_point_fallback: OPointCalcOptions = OPointCalcOptions.RAISE,
    ) -> tuple[float, float, npt.NDArray[np.float64]]:
        """
        Do-not-repeat-yourself utility

        Parameters
        ----------
        x:
            The grid of x coordinates
        z:
            The grid of z coordinates
        psi:
            The psi array
        o_points:
            The list of O-points
        x_points:
            The list of X-points
        lcfs:
            The lcfs (used for fixed boundary handling)
        o_point_fallback:
            If no o points are found use an estimation method

        Returns
        -------
        psix:
            The plasma separatrix psi
        psio:
            The plasma central psi
        mask:
            The numpy array of 0/1 denoting the out/in points of the plasma in
            the grid

        Raises
        ------
        EquilibriaError
            O and X point are needed with LCFS or no o points found
        """
        if lcfs is not None:
            # This is for fixed boundary equilibrium handling
            if not o_points or not x_points:
                raise EquilibriaError(
                    "Need to specify O and X-points when providing an LCFS."
                )

            return x_points[0].psi, o_points[0].psi, in_zone(x, z, lcfs)

        if not o_points:
            match o_point_fallback:
                case OPointCalcOptions.GRID_CENTRE:
                    nx, nz = psi.shape
                    o_points = [Opoint(x[nx // 2], z[nz // 2], psi[nx // 2, nz // 2])]
                case _:
                    _f, ax = plt.subplots()
                    ax.contour(x, z, psi, cmap="viridis")
                    raise EquilibriaError("No O-points found!")

        psio = o_points[0].psi
        if x_points:
            psix = x_points[0].psi
            mask = in_plasma(x, z, psi, o_points, x_points)
        else:
            psix = psi[0, 0]
            mask = None
        return psix, psio, mask

    def fvac(self) -> float:
        """
        Vacuum field function handle
        """  # noqa: DOC201
        try:
            return self._fvac
        except AttributeError:
            raise NotImplementedError("Please specify ._fvac as vacuum R*B.") from None

    def int2d(self, func2d: npt.NDArray[np.float64]) -> float:
        """
        Returns
        -------
        :
            The integral of a 2-D function map (numpy 2-D array) over the
            domain space (X, Z)
        """
        return integrate_dx_dz(func2d, self.dx, self.dz)

    def plot(self, ax=None):
        """
        Plot the Profile object

        Returns
        -------
        :
            The plot axis
        """
        return ProfilePlotter(self, ax=ax)


class BetaIpProfile(Profile):
    """
    Constrain poloidal Beta and plasma current following logic as laid out in
    :doi:`Jeon, 2015 <10.3938/jkps.67.843>` and
    following some implementation in B. Dudson, FreeGS:
    https://github.com/bendudson/freegs

    Parameters
    ----------
    betap:
        Plasma poloidal beta constraint
    I_p:
        Plasma current constraint [A]
    R_0:
        Reactor major radius [m] (used in p' and ff' components)
    B_0:
        Toroidal field at reactor major radius [T]
    shape:
        Shape parameterisation to use

    Notes
    -----
    \t:math:`J_{\\phi} = {\\lambda}\\bigg({\\beta}_{0}\\dfrac{X}{R_{0}}+`
    \t:math:`(1-\\beta_{0})\\dfrac{R_{0}}{X}\\bigg)j_{\\phi_{shape}}(m, n, ..)`

    \t:math:`I_{p}=\\int_{{\\Omega}_{pl}} J_{{\\phi},pl}({\\lambda},{\\beta_{0}})`
    \t:math:`d{\\Omega}`\n

    \t:math:`{\\beta}_{p}=\\dfrac{\\langle p({\\beta_{0}})\\rangle}{\\langle B_{p}^{2}\\rangle_{\\psi_{a}}/2\\mu_{0}}`
    """  # noqa: W505, E501

    # NOTE: For high betap >= 2, this can lead to there being no plasma current
    # on the high field side...
    def __init__(
        self,
        betap: float,
        I_p: float,
        R_0: float,
        B_0: float,
        shape: ShapeFunction | None = None,
    ):
        self.betap = betap
        self.I_p = I_p
        self._fvac = R_0 * B_0
        self.R_0 = R_0
        self._B_0 = B_0  # Store for eqdsk only
        self.scale = 1.0

        if shape is None:
            self.shape = DoublePowerFunc([1, 0.8])
        else:
            self.shape = shape
        if I_p < 0:  # Reverse I_p
            self.shape *= -1

    def jtor(
        self,
        x: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        psi: npt.NDArray[np.float64],
        o_points: list[Opoint],
        x_points: list[Xpoint],
        *,
        o_point_fallback: OPointCalcOptions = OPointCalcOptions.RAISE,
    ) -> npt.NDArray[np.float64]:
        """
        Calculate toroidal plasma current array.

        \t:math:`I_{p} =\\int\\int {\\lambda}\\bigg({\\beta}_{0}\\dfrac{X}{R_{0}}j_{\\phi_{shape}}+`
        \t:math:`(1-\\beta_{0})\\dfrac{R_{0}}{X}j_{\\phi_{shape}}\\bigg)`

        \t:math:`{\\beta}_{p}=\\dfrac{8\\pi}{{\\mu}_{0}{I_{p}^{2}}}\\int\\int pdXdZ`
        \t:math:`= -\\dfrac{8\\pi}{{\\mu}_{0}{I_{p}^{2}}}\\dfrac{\\lambda{\\beta}_{0}}{R_{0}}\\int\\int p_{shape}dXdZ`

        \t:math:`p(\\psi_{N})=-\\dfrac{\\lambda\\beta_{0}}{R_{0}}p_{shape}(\\psi_{N})`

        \t:math:`\\lambda{\\beta_{0}}=-\\dfrac{\\beta_{p}I_{p}^{2}R_{0}\\mu_{0}}{8\\pi \\int\\int p_{shape}}`

        \t:math:`\\lambda=\\dfrac{I_{p}-\\lambda{\\beta_{0}}\\bigg(\\int\\int\\dfrac{X}{R_{0}}f+\\int\\int\\dfrac{R_{0}}{X}f\\bigg)}{\\int\\int\\dfrac{R_{0}}{X}f}`

        Derivation: book 10, p 120
        """  # noqa: W505, E501, DOC201
        self.dx = x[1, 0] - x[0, 0]
        self.dz = z[0, 1] - z[0, 0]
        psix, psio, mask = self._jtor(
            x, z, psi, o_points, x_points, o_point_fallback=o_point_fallback
        )
        psi_norm = (psi - psio) / (psix - psio)
        self.psisep = psix
        self.psiax = psio
        jtorshape = self.shape(psi_norm)

        # Calculate pressure function in plasma including separatrix masking
        if mask is None:
            pfunc = speedy_pressure(psi_norm, psio, psix, self.shape)
        else:
            ii, jj = np.nonzero(mask)
            jtorshape *= mask
            pfunc = speedy_pressure_mask(ii, jj, psi_norm, psio, psix, self.shape)

        if x_points != []:  # NOTE: Necessary unpythonic formulation
            # More accurate beta_p constraint calculation
            # This is the Freidberg approximation
            lcfs, _ = find_LCFS_separatrix(
                x, z, psi, o_points=o_points, x_points=x_points
            )
            v_plasma = revolved_volume(*lcfs.xz)
            Bp = MU_0 * self.I_p / lcfs.length
            p_avg = volume_integral(pfunc, x, self.dx, self.dz) / v_plasma
            beta_p_actual = 2 * MU_0 * p_avg / Bp**2

            lambd_beta0 = -self.betap / beta_p_actual * self.R_0

        else:
            # If there are no X-points, use less accurate beta_p constraint
            lambd_beta0 = (
                -self.betap
                * self.I_p**2
                * self.R_0
                * MU_0
                / (8 * np.pi)
                / self.int2d(pfunc)
            )

        part_a = self.int2d(jtorshape * x / self.R_0)
        part_b = self.int2d(jtorshape * self.R_0 / x)
        lambd = (self.I_p + lambd_beta0 * (part_b - part_a)) / part_b
        beta0 = lambd_beta0 / lambd
        jtor = lambd * (beta0 * x / self.R_0 + (1 - beta0) * self.R_0 / x) * jtorshape
        self.lambd = lambd
        self.beta0 = beta0
        return jtor

    def pprime(self, pn: npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """
        dp/dpsi as a function of normalised psi
        """  # noqa: DOC201
        return self.lambd * self.beta0 / self.R_0 * self.shape(pn)

    def ffprime(self, pn: npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """
        f*df/dpsi as a function of normalised psi
        """  # noqa: DOC201
        return MU_0 * self.lambd * (1 - self.beta0) * self.R_0 * self.shape(pn)


class BetaLiIpProfile(BetaIpProfile):
    """
    Profile is what BLUEPRINT used to do, and Fabrizio told me he had done
    something similar in MIRA, at one point.

    Parameters
    ----------
    betap:
        Plasma poloidal beta constraint
    l_i:
        Normalised internal inductance constraint
    I_p:
        Plasma current constraint [Amps]
    R_0:
        Reactor major radius [m] (used in p' and ff' components)
    B_0:
        Toroidal field [T]
    shape:
        The shape function to use for the flux functions
    li_rel_tol:
        Absolute relative tolerance for the internal inductance constraint
    li_min_iter:
        Iteration at which the profile optimisation should start to be
        carried out. Usually best not to start solving the equilibrium
        with the profile constraint, and fold it in later, when the plasma
        shape is more representative.
    """

    def __init__(
        self,
        betap: float,
        l_i: float,
        I_p: float,
        R_0: float,
        B_0: float,
        shape: ShapeFunction | None = None,
        li_rel_tol: float = 0.015,
        li_min_iter: int = 5,
    ):
        super().__init__(betap, I_p, R_0, B_0, shape=shape)
        self._l_i_target = l_i
        self._l_i_rel_tol = li_rel_tol
        self._l_i_min_iter = li_min_iter


class CustomProfile(Profile):
    """
    User-specified profile functions p'(psi), ff'(psi)
    jtor = R*p' + ff'/(R*MU_0)

    Parameters
    ----------
    pprime_func:
        Pressure prime profile - dp/dpsi(psi_N)
    ffprime_func:
        Force-Force prime profile f*df/dpsi(psi_N)
    R_0:
        Reactor major radius [m]
    B_0:
        Field at major radius [T]
    I_p:
        Plasma current [A]. If None, the plasma current will be calculated
        from p' and ff'.
    """

    def __init__(
        self,
        pprime_func: npt.NDArray[np.float64] | Callable[[float]] | float,
        ffprime_func: npt.NDArray[np.float64] | Callable[[float]] | float,
        R_0: float,
        B_0: float,
        p_func: npt.NDArray[np.float64] | Callable[[float]] | float | None = None,
        f_func: npt.NDArray[np.float64] | Callable[[float]] | float | None = None,
        I_p: float | None = None,
    ):
        self._pprime_in = self.parse_to_callable(pprime_func)
        self._ffprime_in = self.parse_to_callable(ffprime_func)
        self.p_func = self.parse_to_callable(p_func)
        self.f_func = self.parse_to_callable(f_func)
        self._fvac = R_0 * B_0
        self.R_0 = R_0
        self._B_0 = B_0
        self.I_p = I_p
        self.scale = 1.0

        # Fit a shape function to the pprime profile (mostly for plotting)
        x = np.linspace(0, 1, 50)
        self.shape = LaoPolynomialFunc.from_datafit(self.pprime(x))

    @staticmethod
    def parse_to_callable(unknown):
        """
        Make a callable out of an unknown input type

        Raises
        ------
        TypeError
            Cannot make opject callable
        """  # noqa: DOC201
        if callable(unknown):
            return unknown
        if isinstance(unknown, np.ndarray):
            return interp1d(np.linspace(0, 1, len(unknown)), unknown)
        if unknown is None:
            return None
        raise TypeError("Could not make input object a callable function.")

    def pprime(self, pn: npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """
        dp/dpsi as a function of normalised psi
        """  # noqa: DOC201
        return abs(self.scale) * self._pprime_in(pn)

    def ffprime(self, pn: npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """
        f*df/dpsi as a function of normalised psi
        """  # noqa: DOC201
        return abs(self.scale) * self._ffprime_in(pn)

    def jtor(
        self,
        x: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        psi: npt.NDArray[np.float64],
        o_points: list[Opoint],
        x_points: list[Xpoint],
        lcfs: npt.NDArray[np.float64] | None = None,
        *,
        o_point_fallback: OPointCalcOptions = OPointCalcOptions.RAISE,
    ) -> npt.NDArray[np.float64]:
        """
        Calculate toroidal plasma current

        \t:math:`J_{\\phi}=Xp^{'}+\\dfrac{FF^{'}}{\\mu_{0}X}`
        """  # noqa: DOC201
        self.dx = x[1, 0] - x[0, 0]
        self.dz = z[0, 1] - z[0, 0]
        psisep, psiax, mask = self._jtor(
            x, z, psi, o_points, x_points, lcfs=lcfs, o_point_fallback=o_point_fallback
        )
        self.psisep = psisep
        self.psiax = psiax
        psi_norm = np.clip((psi - psiax) / (psisep - psiax), 0, 1)
        jtor = x * self._pprime_in(psi_norm) + self._ffprime_in(psi_norm) / (x * MU_0)
        if mask is not None:
            jtor *= mask
        if self.I_p is not None:
            # This is a simple way to prescribe the plasma current
            I_p = self.int2d(jtor)
            if I_p != 0.0:
                self.scale = self.I_p / I_p
                jtor *= self.scale
        return jtor

    def pressure(self, psinorm: npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """
        Returns
        -------
        :
            Pressure [Pa] at given value(s) of normalised psi
        """
        if self.p_func is not None:
            return abs(self.scale) * self.p_func(psinorm)
        return super().pressure(psinorm)

    def fRBpol(self, psinorm: npt.ArrayLike) -> float | npt.NDArray[np.float64]:
        """
        Returns
        -------
        :
            f=R*Bt at given value(s) of normalised psi
        """
        if self.f_func is not None:
            return abs(self.scale) * self.f_func(psinorm)
        return super().fRBpol(psinorm)

    @classmethod
    def from_eqdsk_file(
        cls,
        filename: Path | str,
        from_cocos: int | None = 11,
        to_cocos: int | None = BLUEMIRA_DEFAULT_COCOS,
        *,
        qpsi_positive: bool | None = None,
        **kwargs,
    ) -> CustomProfile:
        """
        Initialises a CustomProfile object from an eqdsk file
        """  # noqa: DOC201
        e = EQDSKInterface.from_file(
            filename,
            from_cocos=from_cocos,
            to_cocos=to_cocos,
            qpsi_positive=qpsi_positive,
            **kwargs,
        )
        return cls.from_eqdsk(e)

    @classmethod
    def from_eqdsk(cls, eq: EQDSKInterface) -> CustomProfile:
        """
        Initialises a CustomProfile object from an eqdsk object
        """  # noqa: DOC201
        return cls(
            eq.pprime,
            eq.ffprime,
            R_0=eq.xcentre,
            B_0=abs(eq.bcentre),
            p_func=eq.pressure,
            f_func=eq.fpol,
            I_p=abs(eq.cplasma),
        )
