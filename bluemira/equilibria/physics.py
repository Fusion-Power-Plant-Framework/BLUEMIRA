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
A collection of simple equilibrium physics calculations
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, Optional, Union

if TYPE_CHECKING:
    from bluemira.equilibria.equilibrium import Equilibrium
    from bluemira.equilibria.find import Opoint, Xpoint

import numpy as np
from scipy.interpolate import RectBivariateSpline

from bluemira.base.constants import MU_0
from bluemira.equilibria.find import in_plasma
from bluemira.equilibria.grid import revolved_volume, volume_integral


def calc_psi_norm(
    psi: Union[float, np.ndarray], opsi: float, xpsi: float
) -> Union[float, np.ndarray]:
    """
    Calculate normalised magnetic flux.

    Parameters
    ----------
    psi:
        The magnetic flux per radian
    opsi:
        The psi value at the O-point
    xpsi:
        The psi value at the X-point

    Returns
    -------
    The normalised magnetic flux value(s)
    """
    return (opsi - psi) / (opsi - xpsi)


def calc_psi(
    psi_norm: Union[float, np.ndarray], opsi: float, xpsi: float
) -> Union[float, np.ndarray]:
    """
    Calculate the absolute psi values from normalised psi values

    Parameters
    ----------
    psi_norm:
        The normalised psi values
    opsi:
        The psi value at the O-point
    xpsi:
        The psi value at the X-point

    Returns
    -------
    psi:
        The magnetic flux per radian
    """
    return opsi - psi_norm * (opsi - xpsi)


def calc_tau_flattop(psi_sof: float, psi_eof: float, v_burn: float) -> float:
    """
    Calculates the flat-top length

    \t:math:`\\tau_{flat-top}=\\dfrac{\\psi_{SOF}-\\psi_{EOF}}{V_{burn}}`

    Parameters
    ----------
    psi_sof:
        The start of flat-top magnetic flux at the plasma boundary [V.s]
    psi_eof:
        The end of flat-top magnetic flux at the plasma boundary [V.s]
    v_burn:
        The plasma loop voltage during burn [V]

    Returns
    -------
    The duration of the flat-top [s]
    """
    return (psi_sof - psi_eof) / v_burn


def calc_psib(
    psi_bd: float, R_0: float, I_p: float, li: float, c_ejima: float = 0.4
) -> float:
    """
    Calculates the boundary flux at start of flat-top, after the breakdown

    \t:math:`\\psi_b=\\psi(t_{BD})-L_i I_p-\\Delta\\psi_{res}`

    with:
    \t:math:`L_i=\\dfrac{\\mu_0R_0l_i}{2}`

    \t:math:`\\Delta\\psi_{res}=C_{Ejima}\\mu_0R_0I_p`

    Parameters
    ----------
    psi_bd:
        The flux at the breakdown [V.s]
    R_0:
        The machine major radius [m]
    I_p:
        The desired flat-top plasma current [A]
    li:
        The normalised plasma inductance

    Returns
    -------
    The flux at the boundary at start of flat-top [V.s]
    """
    return psi_bd - 0.5 * MU_0 * R_0 * li * I_p - c_ejima * MU_0 * R_0 * I_p


def calc_k0(psi_xx0: float, psi_zz0: float) -> float:
    """
    Calculates the plasma elongation on the plasma axis (rho = 0).

    Parameters
    ----------
    psi_xx0:
        Second derivative of psi in X at the plasma axis (R_0, Z_0)
    psi_zz0:
        Second derivative of psi in Z at the plasma axis (R_0, Z_0)

    Returns
    -------
    Plasma elongation at the plasma axis
    """
    return np.sqrt(psi_xx0 / psi_zz0)


def calc_q0(eq: Equilibrium) -> float:
    """
    Calculates the plasma MHD safety factor on the plasma axis (rho=0).
    Freidberg, Ideal MHD, eq 6.42, p 134

    Parameters
    ----------
    eq:
        Equilibrium for which to calculate the safety factor on axis

    Returns
    -------
    The MHD safety factor on the plasma axis
    """
    opoint = eq.get_OX_points()[0][0]
    psi_xx0 = eq.psi_func(opoint.x, opoint.z, dx=2, grid=False)
    psi_zz0 = eq.psi_func(opoint.x, opoint.z, dy=2, grid=False)
    b_0 = eq.Bt(opoint.x)
    jfunc = RectBivariateSpline(eq.x[:, 0], eq.z[0, :], eq._jtor)
    j_0 = jfunc(opoint.x, opoint.z, grid=False)
    k_0 = calc_k0(psi_xx0, psi_zz0)
    return (b_0 / (MU_0 * opoint.x * j_0)) * (1 + k_0**2) / k_0


def calc_dx_sep(eq: Equilibrium) -> float:
    """
    Calculate the magnitude of the minimum separation between the flux
    surfaces of null points in the equilibrium at the outboard midplane.

    Parameters
    ----------
    eq:
        Equilibrium for which to calculate dx_sep

    Returns
    -------
    Separation distance at the outboard midplane between the active
    null and the next closest flux surface with a null [m]
    """
    o_points, x_points = eq.get_OX_points()
    x, z = eq.get_LCFS().xz
    lfs = np.argmax(x)
    lfp = eq.get_midplane(x[lfs], z[lfs], x_points[0].psi)
    d_x = []
    count = 0  # Necessary because of retrieval of eqdsks with limiters
    for xp in x_points:
        if "Xpoint" in xp.__class__.__name__:
            if count > 0:
                psinorm = calc_psi_norm(xp.psi, o_points[0].psi, x_points[0].psi)
                if psinorm > 1:
                    d_x.append(eq.get_midplane(*lfp, xp.psi)[0])
            count += 1
    return np.min(d_x) - lfp[0]


def calc_volume(eq: Equilibrium) -> float:
    """
    Calculates plasma volume [m^3]
    """
    lcfs = eq.get_LCFS().xz
    return revolved_volume(*lcfs)


def calc_energy(eq: Equilibrium) -> float:
    """
    Calculates the stored poloidal magnetic energy in the plasma [J]

    \t:math:`W=\\dfrac{LI^2}{2}`
    """
    mask = in_plasma(eq.x, eq.z, eq.psi())
    Bp = eq.Bp()
    return volume_integral(Bp**2 * mask, eq.x, eq.dx, eq.dz) / (2 * MU_0)


def calc_Li(eq: Equilibrium) -> float:  # noqa: N802
    """
    Calculates the internal inductance of the plasma [H]

    \t:math:`L_i=\\dfrac{2W}{I_{p}^{2}}`
    """
    p_energy = calc_energy(eq)
    return 2 * p_energy / eq._I_p**2


def calc_li(eq: Equilibrium) -> float:
    """
    Calculates the normalised internal inductance of the plasma

    \t:math:`l_i=\\dfrac{2L_i}{\\mu_{0}R_{0}}`
    """
    li = calc_Li(eq)
    return 2 * li / (MU_0 * eq._R_0)


def calc_li3(eq: Equilibrium) -> float:
    """
    Calculates the normalised internal plasma inductance (ITER approximate
    calculation)

    see https://iopscience.iop.org/article/10.1088/0029-5515/48/12/125002/meta

    \t:math:`li(3)=\\dfrac{2V\\langle B_p^2\\rangle}{(\\mu_0I_p)^2R_0}`

    with:
    \t:math:`\\langle B_p^2\\rangle=\\dfrac{1}{V}\\int B_p^2dV`

    where: Bp is the poloidal magnetic field and V is the plasma volume
    """
    mask = in_plasma(eq.x, eq.z, eq.psi())
    Bp = eq.Bp()
    bpavg = volume_integral(Bp**2 * mask, eq.x, eq.dx, eq.dz)
    return 2 * bpavg / (eq.profiles.R_0 * (MU_0 * eq.profiles.I_p) ** 2)


def calc_li3minargs(
    x: np.ndarray,
    z: np.ndarray,
    psi: np.ndarray,
    Bp: np.ndarray,
    R_0: float,
    I_p: float,
    dx: float,
    dz: float,
    mask: Optional[np.ndarray] = None,
    o_points: Optional[Iterable[Opoint]] = None,
    x_points: Optional[Iterable[Xpoint]] = None,
) -> float:
    """
    Calculate the normalised plasma internal inductance with arguments only.

    Used in the optimisation of the plasma profiles.
    """
    if mask is None:
        mask = in_plasma(x, z, psi, o_points=o_points, x_points=x_points)
    bpavg = volume_integral(Bp**2 * mask, x, dx, dz)
    return 2 * bpavg / (R_0 * (MU_0 * I_p) ** 2)


def calc_p_average(eq: Equilibrium) -> float:
    """
    Calculate the average plasma pressure.

    \t:math:`\\langle p \\rangle = \\dfrac{1}{V_{p}}\\int \\mathbf{p}dxdz`:

    Parameters
    ----------
    eq:
        The Equilibrium object for which to calculate p_average

    Returns
    -------
    The average plasma pressure [Pa]
    """
    p = eq.pressure_map()
    v_plasma = calc_volume(eq)
    return volume_integral(p, eq.x, eq.dx, eq.dz) / v_plasma


def calc_beta_t(eq: Equilibrium) -> float:
    """
    Calculate the ratio of plasma pressure to toroidal magnetic pressure.

    \t:math:`\\beta_t = \\dfrac{2\\mu_0\\langle p \\rangle}{B_t^2}`

    Parameters
    ----------
    eq:
        The Equilibrium object for which to calculate beta_t

    Returns
    -------
    Ratio of plasma to toroidal magnetic pressure
    """
    p_avg = calc_p_average(eq)
    return 2 * MU_0 * p_avg / eq._B_0**2


def calc_beta_p(eq: Equilibrium) -> float:
    """
    Calculate the ratio of plasma pressure to poloidal magnetic pressure

    \t:math:`\\beta_p = \\dfrac{2\\mu_0\\langle p \\rangle}{B_p^2}`

    Parameters
    ----------
    eq:
        The Equilibrium object for which to calculate beta_p

    Returns
    -------
    Ratio of plasma to magnetic pressure
    """
    p = eq.pressure_map()
    mask = eq._get_core_mask()
    Bp = mask * eq.Bp()
    p_int = volume_integral(p, eq.x, eq.dx, eq.dz)
    Bp2_int = volume_integral(Bp**2, eq.x, eq.dx, eq.dz)
    return 2 * MU_0 * p_int / Bp2_int


def calc_beta_p_approx(eq: Equilibrium) -> float:
    """
    Calculate the ratio of plasma pressure to magnetic pressure. This is
    following the definitions of Friedberg, Ideal MHD, pp. 68-69, which is an
    approximation.

    \t:math:`\\beta_p = \\dfrac{2\\mu_0\\langle p \\rangle}{B_p^2}`

    Parameters
    ----------
    eq:
        The Equilibrium object for which to calculate beta_p

    Returns
    -------
    Ratio of plasma to poloidal magnetic pressure
    """
    p_avg = calc_p_average(eq)
    circumference = eq.get_LCFS().length
    Bp = MU_0 * eq._I_p / circumference
    return 2 * MU_0 * p_avg / Bp**2


def calc_summary(eq: Equilibrium) -> Dict[str, float]:
    """
    Calculates interesting values in one go
    """
    R_0, I_p = eq.profiles.R_0, eq.profiles.I_p
    mask = in_plasma(eq.x, eq.z, eq.psi())
    Bp = eq.Bp()
    bpavg = volume_integral(Bp**2 * mask, eq.x, eq.dx, eq.dz)
    energy = bpavg / (2 * MU_0)
    li_true = 2 * energy / I_p**2
    li = 2 * li_true / (MU_0 * R_0)
    li3 = 2 * bpavg / (R_0 * (MU_0 * R_0) ** 2)
    volume = calc_volume(eq)
    beta_p = calc_beta_p(eq)
    return {
        "W": energy,
        "Li": li_true,
        "li": li,
        "li(3)": li3,
        "V": volume,
        "beta_p": beta_p,
    }


def beta(pressure: np.ndarray, field: float) -> float:
    """
    The ratio of plasma pressure to magnetic pressure

    \t:math:`\\beta = \\dfrac{\\langle p \\rangle}{B^2/2\\mu_0}`

    Parameters
    ----------
    pressure:
        Plasma pressure, from which the mean is to be calculated [Pa]
    field:
        Mean total field strength [T]

    Returns
    -------
    Ratio of plasma to magnetic pressure
    """
    return np.mean(pressure) / (field**2 / 2 * MU_0)


def normalise_beta(beta: float, a: float, b_tor: float, I_p: float) -> float:
    """
    Converts beta to normalised beta

    \t:math:`\\beta_{N} = \\beta\\dfrac{aB_{T}}{I_{p}}`

    Parameters
    ----------
    beta:
        Ratio of plasma to magnetic pressure
    a:
        Plasma minor radius [m]
    b_tor:
        Toroidal field [T]
    I_p:
        Plasma current [A]

    Returns
    -------
    Normalised ratio of plasma to magnetic pressure (Troyon factor)
    """
    return beta * a * b_tor / I_p


def beta_N_to_beta(  # noqa: N802
    beta_N: float, a: float, Btor: float, I_p: float  # noqa: N803
) -> float:
    """
    Converts normalised beta to beta

    \t:math:`\\beta = \\beta_{N}\\dfrac{I_{p}}{aB_{T}}`

    Parameters
    ----------
    beta_N:
        Normalised ratio of plasma to magnetic pressure (Troyon factor)
    a:
        Plasma minor radius [m]
    b_tor:
        Toroidal field [T]
    I_p:
        Plasma current [A]

    Returns
    -------
    Ratio of plasma to magnetic pressure
    """
    return beta_N * I_p / (a * Btor)


def calc_infinite_solenoid_flux(r_cs_min: float, r_cs_max: float, B_max: float) -> float:
    """
    Calculate the maximum flux achievable from an infinite solenoid given a peak field.

    Parameters
    ----------
    r_cs_min:
        Inner radius of the infinite solenoid [m]
    r_cs_max:
        Outer radius of the infinite solenoid [m]
    B_max:
        Peak allowable field in the solenoid [T]

    Returns
    -------
    Maximum achievable flux from an infinite solenoid [V.s]
    """
    return B_max * np.pi / 3 * (r_cs_max**2 + r_cs_min**2 + r_cs_max * r_cs_min)
