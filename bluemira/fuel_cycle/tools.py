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
Fuel cycle utility objects, including sink algorithms
"""
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

from bluemira.base.constants import N_AVOGADRO, S_TO_YR, T_LAMBDA, T_MOLAR_MASS, YR_TO_S
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.fuel_cycle.error import FuelCycleError
from bluemira.plasma_physics.reactions import r_T_burn

# =============================================================================
# Miscellaneous utility functions.
# =============================================================================


def find_noisy_locals(
    x: np.ndarray, x_bins: int = 50, mode: str = "min"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find local minima or maxima in a noisy signal.

    Parameters
    ----------
    x:
        The noise data to search
    x_bins:
        The number of bins to search with
    mode:
        The search mode ['min', 'max']

    Returns
    -------
    local_mid_x:
        The arguments of the local minima or maxima
    local_m:
        The local minima or maxima
    """
    if mode == "max":
        peak = np.max
        arg_peak = np.argmax
    elif mode == "min":
        peak = np.min
        arg_peak = np.argmin
    else:
        raise FuelCycleError(f"Unrecognised mode: {mode}.")

    n = len(x)
    bin_size = round(n / x_bins)
    y_bins = [x[i : i + bin_size] for i in range(0, n, bin_size)]

    local_m = np.zeros(len(y_bins))
    local_mid_x = np.zeros(len(y_bins), dtype=int)
    for i, y_bin in enumerate(y_bins):
        local_m[i] = peak(y_bin)
        local_mid_x[i] = arg_peak(y_bin) + i * bin_size
    return local_mid_x, local_m


def discretise_1d(
    x: np.ndarray, y: np.ndarray, n: int, method: str = "linear"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretise x and y for a given number of points.

    Parameters
    ----------
    x:
        The x data
    y:
        The y data
    n:
        The number of discretisation points
    method:
        The interpolation method

    Returns
    -------
    x_1d:
        The discretised x data
    y_1d:
        The discretised y data
    """
    x = np.array(x)
    y = np.array(y)
    x_1d = np.linspace(x[0], x[-1], n)
    y_1d = griddata(x, y, xi=x_1d, method=method)
    return [x_1d, y_1d]


def convert_flux_to_flow(flux: float, area: float) -> float:
    """
    Convert an atomic flux to a flow-rate.

    Parameters
    ----------
    flux:
        The atomic flux [T/m^2/s]
    area:
        The surface area of the flux [m^2]

    Returns
    -------
    The flow-rate [kg/s]
    """
    return flux * area * T_MOLAR_MASS / N_AVOGADRO / 1000


# =============================================================================
# Fitting functions for estimating sink parameters from T retention models.
# =============================================================================


def piecewise_linear_threshold(
    x: np.ndarray, x0: float, y0: float, m1: float, m2: float
) -> np.ndarray:
    """
    Piecewise linear model with initial linear slope, followed by threshold.

    Parameters
    ----------
    x:
        The vector of x values to calculate the function for
    x0:
        The x coordinate of the kink point
    y0:
        The y coordinate of the kink point
    m1:
        The slope of the first curve
    m2:
        The threshold value of the function

    Returns
    -------
    The vector of fitted values
    """
    return np.piecewise(
        x,
        [x < x0],
        [lambda fx: m1 * fx + y0 - m1 * x0, lambda fx: m2],  # noqa: ARG005
    )


def piecewise_sqrt_threshold(
    x: np.ndarray, factor: float, kink: float, threshold: float
) -> np.ndarray:
    """
    Piecewise square-root model, followed by threshold.

    Parameters
    ----------
    x:
        The vector of x values to calculate the function for
    factor:
        The multiplication factor for the sqrt function
    kink:
        The x value where the behaviour changes from sqrt to constant
    threshold:
        The threshold value of the model

    Returns
    -------
    The vector of fitted values
    """
    return np.piecewise(
        x,
        [x < kink],
        [lambda fx: factor * np.sqrt(fx), lambda fx: threshold],  # noqa: ARG005
    )


def fit_sink_data(
    x: np.ndarray, y: np.ndarray, method: str = "sqrt", plot: bool = True
) -> Tuple[float, float]:
    """
    Function used to determine simplified tritium sink model parameters, from
    data values.

    Parameters
    ----------
    x:
        The vector of x values
    y:
        The vector of y values
    method:
        The type of fit to use ['linear', 'sqrt']
    plot:
        Whether or not to plot the fitting result

    Returns
    -------
    slope:
        The slope of the fitted piecewise linear threshold function
    threshold:
        The threshold of the fitted piecewise linear threshold function
    """
    x, y = np.array(x), np.array(y)
    arg = np.nonzero(y > 0.98 * max(y))[0][0]
    kink_point = x[arg]

    if method == "linear":
        fit_func = piecewise_linear_threshold
        bounds = []

    elif method == "sqrt":
        fit_func = piecewise_sqrt_threshold

        bounds = [[-np.inf, kink_point - 1, -np.inf], [np.inf, kink_point + 1, np.inf]]

    else:
        raise FuelCycleError(f"Fitting method '{method}' not recgonised.")

    p_opt = curve_fit(fit_func, x, y, bounds=bounds)

    slope = p_opt[0][0]
    threshold = p_opt[0][-1]

    if plot:
        y_fit = fit_func(x, *p_opt[0])

        true_integral = np.trapz(y)
        fit_integral = np.trapz(y_fit)

        f, ax = plt.subplots()
        ax.set_xlabel("Time [years]")
        ax.set_ylabel("Sequestered tritium [kg]")
        ax.set_title(f"Slope: {slope:.2f}, threshold: {threshold:.2f}")

        ax.plot(x, y, lw=3, color="k", label="Data: $\\int$" + f"{true_integral:.2f}")
        ax.plot(x, y_fit, lw=3, color="r", label="Fit: $\\int$" + f"{fit_integral:.2f}")
        ax.legend()

    return p_opt[0]


# Building Blocks
# =============================================================================
# Elementary flow processing functions ported to numba highly successful in
# reducing runtimes by a factor of ~40
# =============================================================================


@nb.jit(nopython=True, cache=True)
def delay_decay(t: np.ndarray, m_t_flow: np.ndarray, tt_delay: float) -> np.ndarray:
    """
    Time-shift a tritium flow with a delay and account for radioactive decay.

    Parameters
    ----------
    t:
        The time vector
    m_t_flow:
        The mass flow vector
    t_delay:
        The delay duration [s]

    Returns
    -------
    The delayed flow vector
    """
    t_delay = tt_delay * S_TO_YR
    shift = np.argmin(np.abs(t - t_delay))
    flow = np.zeros(shift)
    deldec = np.exp(-T_LAMBDA * t_delay)
    flow = np.append(flow, deldec * m_t_flow)
    # TODO: Slight "loss" of tritium because of this?
    return flow[: len(t)]  # TODO: figure why you had to do this


@nb.jit(nopython=True, cache=True)
def fountain(flow: np.ndarray, t: np.ndarray, min_inventory: float) -> np.ndarray:
    """
    Fountain tritium block. Needs a minimum T inventory to operate.
    This is a binary description. In reality, the TFV systems modelled here
    (such as the cryogenic distillation column) can and do operate below I_min.

    **Inputs:** \\n
      :math:`m_{T_{flow}}` [kg/s]: tritium flow through system [vector] \\n
      :math:`t` [years]: time [vector] \\n
      :math:`I_{min}` [kg]: minimum T inventory for system to operate \\n
    **Outputs:** \\n
      :math:`I` [kg]: built-up T inventory in system [vector]\\n
      :math:`m_{T_{flowout}}` [kg/s]: mass flow out [vector] \\n
    **Calculations** \\n
      :math:`dt = t[i]-t[i-1]` \\n
      :math:`I[i] = I[i-1]e^{-ln(2)dt/t_{1/2}}+m_{T_{flow}}dt` \\n

      if :math:`I > I_{min}`:
        :math:`I[i] = I_{min}` [kg]\\n
        :math:`m_{T_{flowout}} = \\frac{I_{min}-I[i]}{dt}` [kg/s] \\n

      if :math:`I < I_{min}`:
        :math:`I[i] = I[i-1]+m_{T_{flow}}dt` [kg] \\n
        :math:`m_{T_{flowout}} = 0` [kg/s]
    """
    m_out, inventory = np.zeros(len(flow)), np.zeros(len(flow))
    inventory[0] = min_inventory

    for i, _ti in zip(range(1, len(flow)), flow[1:]):
        dt = t[i] - t[i - 1]
        dts = dt * YR_TO_S
        m_in = flow[i] * dts
        inventory[i] = inventory[i - 1] * np.exp(-T_LAMBDA * dt)
        overflow = inventory[i] + m_in

        if overflow > min_inventory:
            m_out[i] = (overflow - min_inventory) / dts
            inventory[i] = min_inventory

        else:
            m_out[i] = 0
            inventory[i] += m_in
    return m_out, inventory


@nb.jit(nopython=True, cache=True)  # Factor ~190 reduction in runtime
def _speed_recycle(
    m_start_up: float, t: np.ndarray, m_in: np.ndarray, m_fuel_injector: np.ndarray
) -> np.ndarray:
    """
    The main recycling loop, JIT compiled.

    Parameters
    ----------
    m_start_up:
        An initial guess for the start-up inventory [kg]
    t:
        The time vector [years]
    m_in:
        The array of tritium flow-rates required for fusion [kg/s]
    m_fuel_injector:
        The array of tritium flow-rates fuelling the plasma [kg/s]

    Returns
    -------
    The tritium in the stores
    """
    m_tritium = np.zeros(len(t))
    m_tritium[0] = m_start_up
    ts = t * YR_TO_S
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        dts = ts[i] - ts[i - 1]
        m_tritium[i] = (
            m_tritium[i - 1] * np.exp(-T_LAMBDA * dt)
            - (m_in[i] - m_fuel_injector[i]) * dts
        )
    return m_tritium


def find_max_load_factor(time_years: np.ndarray, time_fpy: np.ndarray) -> float:
    """
    Finds peak slope in fpy as a function of calendar years
    Divides implicitly by slightly less than a year

    Parameters
    ----------
    time_years:
        The time signal [calendar years]
    time_fpy:
        The time signal [fpy]

    Returns
    -------
    The maximum load factor in the time signal (over a one year period)
    """
    t, rt = discretise_1d(time_years, time_fpy, int(np.ceil(time_years[-1])))
    try:
        a = max([x - x1 for x1, x in zip(rt[:-1], rt[1:])])
    except ValueError:
        # Shortened time overflow error (only happens when debugging)
        a = 1
    if a > 1 or a < 0:
        bluemira_warn(f"Maximum load factor result is non-sensical: {a}.")
        return None
    return a


def legal_limit(
    max_load_factor: float,
    fb: float,
    m_gas: float,
    eta_f: float,
    eta_fuel_pump: float,
    f_dir: float,
    f_exh_split: float,
    f_detrit_split: float,
    f_terscwps: float,
    TBR: float,
    mb: Optional[float] = None,
    p_fus: Optional[float] = None,
):
    """
    Calculates the release rate of T from the model TFV cycle in g/yr.

    :math:`A_{max}\\Bigg[\\Big[\\dot{m_{b}}\\Big((\\frac{1}{f_{b}}-1)+\
    (1-{\\eta}_{f_{pump}})(1-{\\eta}_{f})\\frac{1}{f_{b}{\\eta}_{f}}\\Big)+\
        \\dot{m_{gas}}\\Big](1-f_{DIR})(1-f_{tfv})(1-f_{detrit})+\\dot{m_{b}}\
        \\Lambda f_{TERSCWPS}\\Bigg]\\times365\\times24\\times3600`\n \n
    Where:\n
    :math:`\\dot{m_{b}} = \\frac{P_{fus}[MW]M_{T}[g/mol]}
    {17.58 [MeV]eV[J]N_{A}[1/mol]} [g/s]`
    """
    if p_fus is None and mb is None:
        raise FuelCycleError("You must specify either fusion power or burn rate.")

    if p_fus is not None and mb is not None:
        bluemira_warn(
            "Fusion power and burn rate specified... sticking with fusion power."
        )
        mb = None

    if mb is None:
        mb = r_T_burn(p_fus)

    m_plasma = (
        (mb * ((1 / fb - 1) + (1 - eta_fuel_pump) * (1 - eta_f) / (eta_f * fb)) + m_gas)
        * (1 - f_dir)
        * (1 - f_exh_split)
        * (1 - f_detrit_split)
    )
    m_bb = mb * TBR * (1 - f_terscwps)
    ll = max_load_factor * (m_plasma + m_bb)
    return ll * 365 * 24 * 3600  # g/yr


@nb.jit(nopython=True, cache=True)
def _dec_I_mdot(  # noqa: N802
    inventory: float, eta: float, m_dot: float, t_in: float, t_out: float
) -> float:
    """
    Analytical value of series expansion for an inventory I with a incoming
    flux of tritium (kg/yr).

    \t:math:`I_{end} = Ie^{-{\\lambda}{\\Delta}t}+{\\eta}\\dot{m}\\sum_{t=0}^{{\\Delta}t}e^{-\\lambda(T-t)}`

    \t:math:`I_{end} = Ie^{-{\\lambda}{\\Delta}t}+{\\eta}\\dot{m}\\dfrac{e^{-{\\lambda}T}\\big(e^{{\\lambda}({\\Delta}t+1/2)}-1\\big)}{e^{\\lambda}-1}`
    """  # noqa: W505, E501
    # intuitive hack for 1/2... maths says it should be 1
    dt = t_out - t_in

    out_inventory = inventory * np.exp(-T_LAMBDA * dt) + eta * m_dot * (
        np.exp(-T_LAMBDA * dt) * (np.exp(T_LAMBDA * (dt + 0)) - 1)
    ) / (np.exp(T_LAMBDA) - 1)
    if out_inventory < 0:
        raise ValueError("The out inventory should not be below 0...")
    return out_inventory


@nb.jit(nopython=True, cache=True)
def _timestep_decay(flux: float, dt: float) -> float:
    """
    Analytical value of series expansion for an in-flux of tritium over a time-
    step. Accounts for decay during the timestep only.

    \t:math:`I_{end} = I\\dfrac{e^{-{\\lambda}T}\\big(e^{{\\lambda}({\\Delta}t+1)}-1\\big)}{e^{\\lambda}-1}`

    Parameters
    ----------
    flux:
        The total inventory flowing through on a given time-step [kg]
    dt:
        The time-step [years]

    Returns
    -------
    The value of the total inventory which decayed over the time-step.
    """  # noqa: W505, E501
    return flux * (
        1
        - (np.exp(-T_LAMBDA * dt) * (np.exp(T_LAMBDA * (dt + 0)) - 1))
        / (np.exp(T_LAMBDA) - 1)
    )


@nb.jit(nopython=True, cache=True)
def _find_t15(
    inventory: float,
    eta: float,
    m_flow: float,
    t_in: float,
    t_out: float,
    inventory_limit: float,
) -> float:
    """
    Inter-timestep method solving for dt in the below equality:

    :math:`Ie^{\\lambda{\\Delta}t}+{\\eta}\\dot{m}\\dfrac{e^{-{\\lambda}{\\Delta}t}
    \\big(e^{{\\lambda}({\\Delta}t+1/2)}-1\\big)}{e^{\\lambda}-1}=I_{lim}`\n
    :math:`{\\Delta}t=\\dfrac{ln\\bigg(\\dfrac{Ie^{\\lambda}-I-{\\eta}\\dot{m}}
    {I_{lim}e^{{\\lambda}}-I_{lim}-{\\eta}\\dot{m}e^{\\lambda/2}}\\bigg)}{\\lambda}`
    \n
    returns dt relative to t_in of crossing point
    """
    t = (
        np.log(
            (inventory * np.exp(T_LAMBDA) - inventory - eta * m_flow)
            / (
                inventory_limit * np.exp(T_LAMBDA)
                - inventory_limit
                - eta * m_flow * np.exp(T_LAMBDA / 2)
            )
        )
        / T_LAMBDA
    )
    if t > 0.0:  # don't use max for numbagoodness  # noqa: PLR2004
        dt = t_out - t_in
        if t < dt:
            return t
        return dt
    return 0.0  # Approximate answer sometimes negative... :'(


@nb.jit(nopython=True, cache=True)
def _fountain_linear_sink(
    m_flow: float,
    t_in: float,
    t_out: float,
    inventory: float,
    fs: float,
    max_inventory: float,
    min_inventory: float,
    sum_in: float,
    decayed: float,
) -> Tuple[float, float, float, float]:
    """
    A simple linear fountain tritium retention sink model between a minimum
    and a maximum. Used over a time-step.

    Parameters
    ----------
    m_flow:
        The in-flow of tritium [kg/s]
    t_in:
        The first point in the time-step [years]
    t_out:
        The second point in the time-step [years]
    inventory:
        The inventory of tritium already in the sink [kg]
    fs:
        The tritium release rate of the sink (1-absorbtion rate)
    max_inventory:
        The threshold inventory of the sink at which point it saturates
    min_inventory:
        The minimum inventory required for the system to release tritium
    sum_in:
        Accountancy parameter to calculate the total value lost to a sink
    decayed:
        Accountancy parameter to calculate the total value of decayed T in a sink

    Returns
    -------
    m_out:
        The out-flow of tritium [kg/s]
    inventory:
        The amount of tritium in the sink [kg]
    sum_in:
        Accountancy parameter to calculate the total value lost to a sink
    decayed:
        Accountancy parameter to calculate the total value of decayed T in a sink
    """
    dt = t_out - t_in
    if dt == 0:
        return m_flow, inventory, sum_in, decayed

    m_in = m_flow * YR_TO_S  # kg/yr
    dts = dt * YR_TO_S
    mass_in = m_flow * dts
    sum_in += mass_in

    j_inv0 = inventory

    if inventory <= min_inventory:
        # Case where fountain is not full
        i_mdot = _dec_I_mdot(inventory, 1, m_in, t_in, t_out)
        if i_mdot < min_inventory:
            # Case where M_in still doesn't fill up
            m_out = 0.0
            inventory = i_mdot

        elif i_mdot >= min_inventory:
            # Case where M_in crosses up into to uncanny valley
            # (below which eta=1)

            t15 = _find_t15(inventory, 1, m_in, t_in, t_out, min_inventory)
            i_mdot2 = _dec_I_mdot(min_inventory, 1 - fs, m_in, t_in + t15, t_out)
            if i_mdot2 <= min_inventory:
                # Case where infinite unstable oscillations occur in model
                # Treat reasonably here (you got unlucky) ==> stall

                inventory = min_inventory
                topup = min_inventory * (1 - np.exp(-T_LAMBDA * (t_out - t_in - t15)))
                m_out_temp = mass_in - m_in * t15 - topup
                m_out_temp = max(m_out_temp, 0)
                m_out = m_out_temp / dts  # spread evenly over timestep
            elif i_mdot2 >= max_inventory:
                # Case (unlikely) where massive overshoot occurs
                # TODO: Handle properly
                inventory = max_inventory
                t175 = _find_t15(
                    min_inventory, 1 - fs, m_in, t_in + t15, t_out, max_inventory
                )
                topup = max_inventory * (
                    1 - np.exp(-T_LAMBDA * (t_out - t_in - t175 - t15))
                )
                m_out_temp = mass_in - topup - m_in * t15 - (1 - fs) * m_in * t175
                m_out_temp = max(m_out_temp, 0)
                m_out = m_out_temp / dts
            else:
                # Case where successfully crosses up
                dt2 = t_out - t_in - t15
                inventory = i_mdot2
                m_out = (mass_in - m_in * t15 - (1 - fs) * m_in * dt2) / dts

    elif inventory <= max_inventory:
        # Uncanny valley, no man's land
        i_mdot = _dec_I_mdot(inventory, 1 - fs, m_in, t_in, t_out)
        if i_mdot < min_inventory:
            # Case where it crosses from uncanny valley downwards
            t15 = _find_t15(inventory, 1 - fs, m_in, t_in, t_out, min_inventory)
            i_mdot2 = _dec_I_mdot(min_inventory, 1, m_in, t_in + t15, t_out)
            if i_mdot2 < min_inventory:
                # Case where successfully crosses down
                dt2 = t_out - t_in - t15
                inventory = i_mdot2

                m_out = (mass_in - (1 - fs) * m_in * t15 - m_in * dt2) / dts
            elif i_mdot2 >= min_inventory:
                # Case where infinite unstable oscillations occur in model
                # Treat reasonably here (you got unlucky) ==> stall

                inventory = min_inventory
                topup = min_inventory * (1 - np.exp(-T_LAMBDA * (t_out - t_in - t15)))
                m_out_temp = mass_in - m_in * t15 - topup
                if m_out_temp < 0:
                    m_out_temp = 0
                m_out = m_out_temp / dts  # spread evenly over timestep

        elif i_mdot >= max_inventory:
            t15 = _find_t15(inventory, 1 - fs, m_in, t_in, t_out, max_inventory)

            dt2 = t_out - t_in - t15
            # Case where fountain and bathub are overflowing
            topup = max_inventory * (1 - np.exp(-T_LAMBDA * dt2))
            if topup <= mass_in:
                # Case where I stays constant because of sufficient refill

                m_out = (mass_in - topup) / dts
                inventory = max_inventory
            else:
                # Case where refill insufficient and I depletes
                i_mdot = _dec_I_mdot(inventory, 1 - fs, m_in, t_in, t_out)
                m_out = (mass_in - (1 - fs) * m_in * dt2) / dts

                inventory = i_mdot
        else:
            # Case where we stay in uncanny valley
            inventory = i_mdot
            m_out = (mass_in - (1 - fs) * m_in * dt) / dts
    else:
        # inventory > max_inventory
        raise ValueError("Undefined behaviour for inventory > max_inventory.")

    decayed += j_inv0 - inventory

    if m_out > m_flow:
        print(m_flow, m_out)
        raise ValueError(
            "Out flow greater than in flow. Check that your timesteps are small enough."
        )
    if m_out < 0:
        raise ValueError("Negative out flow in fountain_linear_sink.")
    if inventory < 0:
        raise ValueError("Negative inventory in fountain_linear_sink.")
    return m_out, inventory, sum_in, decayed


@nb.jit(nopython=True, cache=True)
def _linear_thresh_sink(
    m_flow: float,
    t_in: float,
    t_out: float,
    inventory: float,
    fs: float,
    max_inventory: float,
    sum_in: float,
    decayed: float,
) -> Tuple[float, float, float, float]:
    """
    A simple linear tritium retention sink model. Used over a time-step.

    Parameters
    ----------
    m_flow:
        The in-flow of tritium [kg/s]
    t_in:
        The first point in the time-step [years]
    t_out:
        The second point in the time-step [years]
    inventory:
        The inventory of tritium already in the sink [kg]
    fs:
        The tritium release rate of the sink (1-absorbtion rate)
    max_inventory:
        The threshold inventory of the sink at which point it saturates
    sum_in:
        Accountancy parameter to calculate the total value lost to a sink
    decayed:
        Accountancy parameter to calculate the total value of decayed T in a sink

    Returns
    -------
    m_out:
        The out-flow of tritium [kg/s]
    inventory:
        The amount of tritium in the sink [kg]
    sum_in:
        Accountancy parameter to calculate the total value lost to a sink
    decayed:
        Accountancy parameter to calculate the total value of decayed T in a sink
    """
    years = 365 * 24 * 3600
    dt = t_out - t_in
    if dt == 0:
        return m_flow, inventory, sum_in, decayed

    m_in = m_flow * years  # kg/yr
    dts = dt * years
    mass_in = m_flow * dts
    sum_in += mass_in
    j_inv0 = inventory

    i_mdot = _dec_I_mdot(inventory, 1 - fs, m_in, t_in, t_out)
    if i_mdot >= max_inventory:
        t15 = _find_t15(inventory, 1 - fs, m_in, t_in, t_out, max_inventory)
        dt2 = t_out - t_in - t15
        # Case where fountain and bathub are overflowing
        topup = max_inventory * (1 - np.exp(-T_LAMBDA * dt2))
        if topup <= mass_in:
            # Case where I stays constant because of sufficient refill
            m_out = (mass_in - topup) / dts

            inventory = max_inventory
        else:
            # Case where refill insufficient and I depletes
            i_mdot = _dec_I_mdot(inventory, 1 - fs, m_in, t_in, t_out)
            m_out = (mass_in - (1 - fs) * m_in * dt2) / dts
            inventory = i_mdot
    else:
        inventory = i_mdot
        m_out = fs * m_flow

    decayed += j_inv0 - inventory
    return m_out, inventory, sum_in, decayed


@nb.jit(nopython=True, cache=True)
def _sqrt_thresh_sink(
    m_flow: float,
    t_in: float,
    t_out: float,
    inventory: float,
    factor: float,
    max_inventory: float,
    sum_in: float,
    decayed: float,
    _testing: bool,
) -> Tuple[float, float, float, float]:
    """
    A simple sqrt tritium retention sink model. Used over a time-step.

    Parameters
    ----------
    m_flow:
        The in-flow of tritium [kg/s]
    t_in:
        The first point in the time-step [years]
    t_out:
        The second point in the time-step [years]
    inventory:
        The inventory of tritium already in the sink [kg]
    factor:
        The multiplication factor of the sqrt function
    max_inventory:
        The threshold inventory of the sink at which point it saturates
    sum_in:
        Accountancy parameter to calculate the total value lost to a sink
    decayed:
        Accountancy parameter to calculate the total value of decayed T in a sink

    Returns
    -------
    m_out:
        The out-flow of tritium [kg/s]
    inventory:
        The amount of tritium in the sink [kg]
    sum_in:
        Accountancy parameter to calculate the total value lost to a sink
    decayed:
        Accountancy parameter to calculate the total value of decayed T in a sink

    Notes
    -----
    \t:math:`I_{sequestered} = factor \\times \\sqrt{ t_{fpy}}`

    The time in the equation is sub-planted for the inventory, to make the
    retention model independent of time.

    The values for the threshold and factor must be obtained from detailed T
    retention modelling.

    Here, we're tacking the growth of the inventory to a function, but decay is
    not accounted for in this function. We have to add decay in the sink and
    ensure this is handled when calculation the absorbtion and out-flow.
    """
    years = 365 * 24 * 3600
    dt = t_out - t_in
    if dt == 0:
        # Nothing can happen if time is zero
        return m_flow, inventory, sum_in, decayed

    dts = dt * years
    mass_in = m_flow * dts
    sum_in += mass_in

    decay = inventory * (1 - np.exp(-T_LAMBDA * dt))

    if mass_in == 0:
        # Inventory decays, nothing else happens
        new_inventory = inventory - decay
        # If the in mass is 0, so must be the out-flow
        return 0.0, new_inventory, sum_in, decayed

    if inventory >= max_inventory:
        # Sqrt bathtub is over-flowing
        inventory -= decay
        # Determine the equivalent time for a given inventory level
        x = (inventory / factor) ** 2
        new_inventory = factor * np.sqrt(x + dt)
        absorbed = new_inventory - inventory
        absorbed_decay = _timestep_decay(absorbed, dt)
        absorbed += absorbed_decay
        if absorbed > decay:
            # Case where the absorbtion is greater than the decay loss
            new_inventory = max_inventory

            # Only absorb the decayed amount and top-up the sink to its limit
            fraction = decay / mass_in
            m_out = (1 - fraction) * m_flow
        else:
            # Case where there is decay which is not compensated by absorbtion
            new_inventory += absorbed
            new_inventory -= decay
            fraction = absorbed / mass_in
            m_out = (1 - fraction) * m_flow

    else:
        # Sqrt bathtub is not yet full..
        # Determine the equivalent time for a given inventory level
        x = (inventory / factor) ** 2

        # This is equivalent to determining the gradient, but stabler
        new_inventory = factor * np.sqrt(x + dt)
        absorbed = new_inventory - inventory

        absorbed_decay = _timestep_decay(absorbed, dt)

        # Sum all the absolute loss terms and modify the out-flow
        delta_inv = absorbed + absorbed_decay
        fraction = delta_inv / mass_in
        m_out = (1 - fraction) * m_flow
        if not _testing:
            new_inventory -= decay

    return m_out, new_inventory, sum_in, decayed


@nb.jit(nopython=True, cache=True)  # Factor ~70 reduction in runtime
def linear_bathtub(
    flow: np.ndarray, t: np.ndarray, eta: float, bci: int, max_inventory: float
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Bathtub sink model.

    Parameters
    ----------
    flow:
        The vector of flow-rates [kg/s]
    t:
        The time vector [years]
    eta:
        The bathtub tritium release fraction
    bci:
        The blanket change index. Used if a component is replaced to reset the
        inventory to 0.
    max_inventory:
        The threshold inventory for the bathtub.

    Returns
    -------
    m_out:
        The out-flow of tritium [kg/s]
    inventory:
        The amount of tritium in the sink [kg]
    sum_in:
        Accountancy parameter to calculate the total value lost to a sink
    decayed:
        Accountancy parameter to calculate the total value of decayed T in a sink
    """
    decayed, sum_in = 0, 0
    if bci is None:
        bci = -1  # Numba typing fix
    m_out, inventory = np.zeros(len(flow)), np.zeros(len(flow))
    for i, mflow in enumerate(flow[:-1]):
        m_out[i], inventory[i], sum_in, decayed = _linear_thresh_sink(
            mflow, t[i], t[i + 1], inventory[i - 1], eta, max_inventory, sum_in, decayed
        )
        if i == bci or i < 1:
            # Dump stored inventory on component change.
            inventory[i] = 0
    return m_out, inventory, sum_in, decayed


@nb.jit(nopython=True, cache=True)
def sqrt_bathtub(
    flow: np.ndarray,
    t: np.ndarray,
    factor: float,
    bci: int,
    max_inventory: float,
    _testing: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Bathtub sink model with a sqrt inventory retention law.

    Parameters
    ----------
    flow:
        The vector of flow-rates [kg/s]
    bci:
        The blanket change index. Used if a component is replaced to reset the
        inventory to 0.
    t:
        The time vector [years]
    factor:
        The sqrt model multiplication factor
    max_inventory:
        The threshold inventory for the bathtub.
    _testing:
        Used for testing purposes only (switches off decay).

    Returns
    -------
    m_out:
        The out-flow of tritium [kg/s]
    inventory:
        The amount of tritium in the sink [kg]
    sum_in:
        Accountancy parameter to calculate the total value lost to a sink
    decayed:
        Accountancy parameter to calculate the total value of decayed T in a sink
    """
    decayed, sum_in = 0, 0
    if bci is None:
        bci = -1  # Numba typing fix
    m_out, inventory = np.zeros(len(flow)), np.zeros(len(flow))

    for i, mflow in enumerate(flow[:-1]):
        m_out[i], inventory[i], sum_in, decayed = _sqrt_thresh_sink(
            mflow,
            t[i],
            t[i + 1],
            inventory[i - 1],
            factor,
            max_inventory,
            sum_in,
            decayed,
            _testing,
        )
        if i == bci or i < 1:
            # Dump stored inventory on component change.
            inventory[i] = 0
    return m_out, inventory, sum_in, decayed


@nb.jit(nopython=True, cache=True)
def fountain_bathtub(
    flow: np.ndarray,
    t: np.ndarray,
    fs: float,
    max_inventory: float,
    min_inventory: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    A fountain and bathtub sink simultaneously.

    Parameters
    ----------
    flow:
        Tritium flow through system [kg/s]
    t:
        Time [years]
    fs:
        efficiency of bathtub
    min_inventory:
        Minimum T inventory for system to operate [kg]
    max_inventory:
        Maximum T inventory for system to operate [kg]

    Returns
    -------
    m_out:
        The out-flow of tritium [kg/s]
    inventory:
        The amount of tritium in the sink [kg]
    sum_in:
        Accountancy parameter to calculate the total value lost to a sink
    decayed:
        Accountancy parameter to calculate the total value of decayed T in a sink

    \t:math:`dt = t[i]-t[i-1]` \n
    \t:math:`I[i] = I[i-1]e^{-ln(2)dt/t_{1/2}}` \n
    \tif :math:`I < I_{min}`:
    \t\t:math:`I[i] += m_{T_{flow}}dt`\n
    \t\t:math:`m_{T_{flowout}} = 0` \n
    \tif :math:`I >= I_{max}`:
    \t\t:math:`I[i] = I_{max}` \n
    \t\t:math:`m_{T_{flowout}} = m_{T_{flow}}` \n
    \tif :math:`I < I_{max}`:
    \t\t:math:`m_{T_{flowout}} = {\\eta}m_{T_{flow}}`\n
    \t\t:math:`I += (1-{\\eta})m_{T_{flow}}dt`
    """
    decayed, sum_in = 0, 0
    m_out, inventory = np.zeros(len(flow)), np.zeros(len(flow))
    for i, mflow in enumerate(flow[:-1]):
        m_out[i], inventory[i], sum_in, decayed = _fountain_linear_sink(
            mflow,
            t[i],
            t[i + 1],
            inventory[i - 1],
            fs,
            max_inventory,
            min_inventory,
            sum_in,
            decayed,
        )
        if i < 1:
            inventory[i], m_out[i] = min_inventory, 0

    return m_out, inventory, sum_in, decayed
