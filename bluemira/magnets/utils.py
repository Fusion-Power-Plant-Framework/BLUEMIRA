from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from bluemira.magnets.cable import Cable
from bluemira.magnets.conductor import Conductor


def delayed_exp_func(x0: float, tau: float, t_delay: float = 0):
    """
    Delayed Exponential function

    x = x0 * exp(-(t-t_delay)/tau)

    Parameters
    ----------
    x0: float
        initial value
    tau: float
        characteristic time constant
    t_delay: float
        delay time

    Returns
    -------
    A Callable - exponential function

    """

    def fun(t):
        x = x0
        if t > t_delay:
            x = x0 * np.exp(-(t - t_delay) / tau)
        return x

    return fun


def _heat_balance_model_cable(t, T, B: Callable, I: Callable, cable: Cable):
    """
    Calculate the derivative of temperature (dT/dt) for the heat balance problem.

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

    Q_gen = (I(t) / cable.area) ** 2 * cable.res(B=B(t), T=T)

    # Calculate the rate of heat absorption by conductor components
    Q_abs = cable.cp_v(T=T)

    # Calculate the derivative of temperature with respect to time (dT/dt)
    dTdt = Q_gen / Q_abs

    return dTdt


def _temperature_evolution(
        t0: float,
        tf: float,
        initial_temperature: float,
        B: Callable,
        I: Callable,
        cable: Cable,
):
    solution = solve_ivp(
        _heat_balance_model_cable,
        np.array([t0, tf]),
        [initial_temperature],
        args=(B, I, cable),
        dense_output=True,
    )

    if not solution.success:
        raise ValueError("Temperature evolution did not converged")

    return solution


def _sigma_r_jacket(conductor: Conductor, pressure: float, T: float, B: float):
    saf_jacket = (conductor.cable.dx + 2 * conductor.dx_jacket) / (
            2 * conductor.dx_jacket
    )
    X_jacket = conductor.Xx(T=T, B=B)
    return pressure * X_jacket * saf_jacket


def optimize_jacket_conductor(
        conductor: Conductor,
        pressure: float,
        T: float,
        B: float,
        allowable_sigma: float,
        bounds: np.array = None,
):
    def sigma_difference(
            dx_jacket: float,
            pressure: float,
            T: float,
            B: float,
            conductor: Conductor,
            allowable_sigma: float,
    ):
        conductor.dx_jacket = dx_jacket
        sigma_r = _sigma_r_jacket(conductor, pressure, T, B)
        diff = abs(sigma_r - allowable_sigma)
        return diff

    method = None
    if bounds is not None:
        method = "bounded"

    result = minimize_scalar(
        fun=sigma_difference,
        args=(pressure, T, B, conductor, allowable_sigma),
        bounds=bounds,
        method=method,
        options={"xatol": 1e-4},
    )

    if not result.success:
        raise ValueError("dx_jacket optimization did not converge.")
    conductor.dx_jacket = result.x
    print(f"Optimal dx_jacket: {conductor.dx_jacket}")
    print(f"Averaged sigma_r: {_sigma_r_jacket(conductor, pressure, T, B) / 1e6} MPa")

    return result


def optimize_n_stab_cable(
        cable: Cable,
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
    Optimize the number of stabilizer strand in a superconducting cable using a 0-D hot spot criteria

    Parameters
    ----------
        cable: Cable
            the superconducting cable
        t0: float
            initial time
        tf: float
            final time
        initial_temperature: float
            temperature [K] at initial time
        target_temperature: float
            target temperature [K] at final time
        B : Callable
            The magnetic field [T] as time function
        I : Callable
            The current [A] flowing through the conductor as time function
        bounds: np.ndarray
            lower and upper limits for the number of strand in the cable
        show: bool
            if True the behavior of temperature as function of time is plotted

    Returns
    -------
        None

    Notes
    -----
        The number of stabilizer strands in the cable is directly modified. An error is raised in case the optimization
        process did not converge.
    """

    def final_temperature_difference(
            n_stab: int,
            t0: float,
            tf: float,
            initial_temperature: float,
            target_temperature: float,
            B: Callable,
            I: Callable,
            cable: Cable,
    ):
        cable.n_stab_strand = n_stab

        solution = _temperature_evolution(
            t0=t0, tf=tf, initial_temperature=initial_temperature, B=B, I=I, cable=cable
        )
        final_T = float(solution.y[0][-1])
        diff = abs(final_T - target_temperature)
        return diff

    method = None
    if bounds is not None:
        method = "bounded"

    result = minimize_scalar(
        fun=final_temperature_difference,
        args=(t0, tf, initial_temperature, target_temperature, B, I, cable),
        bounds=bounds,
        method=method,
    )

    if not result.success:
        raise ValueError(
            "n_stab optimization did not converge. Check your input parameters or initial bracket."
        )

    solution = _temperature_evolution(t0, tf, initial_temperature, B, I, cable)
    final_temperature = solution.y[0][-1]

    print(f"Optimal n_stab: {cable.n_stab_strand}")
    print(f"Final temperature with optimal n_stab: {final_temperature} Kelvin")

    if show:
        _, ax = plt.subplots()
        ax.plot(solution.t, solution.y[0], "r")
        time_steps = np.linspace(t0, tf, 100)
        ax.plot(time_steps, solution.sol(time_steps)[0], "b")
        plt.show()

    return result
