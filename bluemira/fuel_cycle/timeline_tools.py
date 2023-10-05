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
Distribution and timeline utilities
"""
import abc
from typing import Iterable, Tuple

import numpy as np
from scipy.optimize import brentq

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.fuel_cycle.error import FuelCycleError

__all__ = [
    "GompertzLearningStrategy",
    "UserSpecifiedLearningStrategy",
    "UniformLearningStrategy",
    "LogNormalAvailabilityStrategy",
    "TruncNormAvailabilityStrategy",
    "ExponentialAvailabilityStrategy",
]


def f_gompertz(t: float, a: float, b: float, c: float) -> float:
    """
    Gompertz sigmoid function parameterisation.

    \t:math:`a\\text{exp}(-b\\text{exp}(-ct))`
    """
    return a * np.exp(-b * np.exp(-c * t))


def f_logistic(t: float, value: float, k: float, x_0: float) -> float:
    """
    Logistic function parameterisation.
    """
    return value / (1 + np.exp(-k * (t - x_0)))


def histify(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform values into arrays usable to make histograms.
    """
    x, y = np.array(x), np.array(y)
    return x.repeat(2)[1:-1], y.repeat(2)


def generate_lognorm_distribution(n: int, integral: float, sigma: float) -> np.ndarray:
    """
    Generate a log-norm distribution for a given standard deviation of the
    underlying normal distribution. The mean value of the normal distribution
    is optimised approximately.

    Parameters
    ----------
    n:
        The size of the distribution
    integral:
        The integral value of the distribution
    sigma:
        The standard deviation of the underlying normal distribution

    Returns
    -------
    The distribution of size n and of the correct integral value
    """
    rng = np.random.default_rng()

    def f_integral(x):
        return np.sum(rng.lognormal(x, sigma, n)) - integral

    mu = brentq(f_integral, -1e3, 1e3, maxiter=200)
    distribution = rng.lognormal(mu, sigma, n)
    # Correct distribution integral
    error = np.sum(distribution) - integral
    distribution -= error / n
    return distribution


def generate_truncnorm_distribution(n: int, integral: float, sigma: float) -> np.ndarray:
    """
    Generate a truncated normal distribution for a given standard deviation.

    Parameters
    ----------
    n:
        The size of the distribution
    integral:
        The integral value of the distribution
    sigma:
        The standard deviation of the underlying normal distribution

    Returns
    -------
    The distribution of size n and of the correct integral value
    """
    rng = np.random.default_rng()
    distribution = rng.normal(0, sigma, n)
    # Truncate distribution by 0-folding
    distribution = np.abs(distribution)
    # Correct distribution integral
    distribution /= np.sum(distribution)
    distribution *= integral
    return distribution


def generate_exponential_distribution(
    n: int, integral: float, lambdda: float
) -> np.ndarray:
    """
    Generate an exponential distribution for a given rate parameter.

    Parameters
    ----------
    n:
        The size of the distribution
    integral:
        The integral value of the distribution
    lambdda:
        The rate parameter of the distribution

    Returns
    -------
    The distribution of size n and of the correct integral value
    """
    rng = np.random.default_rng()
    distribution = rng.exponential(lambdda, n)
    # Correct distribution integral
    distribution /= np.sum(distribution)
    distribution *= integral
    return distribution


class LearningStrategy(abc.ABC):
    """
    Abstract base class for learning strategies distributing the total operational
    availability over different operational phases.
    """

    @abc.abstractmethod
    def generate_phase_availabilities(
        self, lifetime_op_availability: float, op_durations: Iterable[float]
    ) -> Iterable[float]:
        """
        Generate operational availabilities for the specified phase durations.

        Parameters
        ----------
        lifetime_op_availability:
            Operational availability averaged over the lifetime
        op_durations:
            Durations of the operational phases [fpy]

        Returns
        -------
        Operational availabilities at each operational phase
        """


class UniformLearningStrategy(LearningStrategy):
    """
    Uniform learning strategy
    """

    @staticmethod
    def generate_phase_availabilities(
        lifetime_op_availability: float, op_durations: Iterable[float]
    ) -> Iterable[float]:
        """
        Generate operational availabilities for the specified phase durations.

        Parameters
        ----------
        lifetime_op_availability:
            Operational availability averaged over the lifetime
        op_durations:
            Durations of the operational phases [fpy]

        Returns
        -------
        Operational availabilities at each operational phase
        """
        return lifetime_op_availability * np.ones(len(op_durations))


class UserSpecifiedLearningStrategy(LearningStrategy):
    """
    User-specified learning strategy to hard-code the operational availabilities at
    each operational phase.
    """

    def __init__(self, operational_availabilities: Iterable[float]):
        """
        Parameters
        ----------
        operational_availabilities:
            Operational availabilities to prescribe
        """
        self.operational_availabilities = operational_availabilities

    def generate_phase_availabilities(
        self, lifetime_op_availability: float, op_durations: Iterable[float]
    ) -> Iterable[float]:
        """
        Generate operational availabilities for the specified phase durations.

        Parameters
        ----------
        lifetime_op_availability:
            Lifetime operational availability
        op_durations:
            Durations of the operational phases [fpy]

        Returns
        -------
        Operational availabilities at each operational phase
        """
        if len(op_durations) != len(self.operational_availabilities):
            raise FuelCycleError(
                "The number of phases is not equal to the number of user-specified"
                " operational availabilities."
            )

        total_fpy = np.sum(op_durations)
        fraction = (total_fpy / lifetime_op_availability) / (
            op_durations / self.operational_availabilities
        )
        if fraction != 1.0:  # noqa: PLR2004
            bluemira_warn(
                "User-specified operational availabilities do not match the specified"
                f" lifetime operational : {fraction:.2f} != 1.0. Normalising to adjust"
                " to meet the specified lifetime operational availability."
            )

        return fraction * self.operational_availabilities


class GompertzLearningStrategy(LearningStrategy):
    """
    Gompertz learning strategy.
    """

    def __init__(
        self, learn_rate: float, min_op_availability: float, max_op_availability: float
    ):
        """
        Parameters
        ----------
        learn_rate:
            Gompertz distribution learning rate
        min_op_availability:
            Minimum operational availability within any given operational phase
        max_op_availability:
            Maximum operational availability within any given operational phase
        """
        self.learn_rate = learn_rate
        self.min_op_a = min_op_availability
        self.max_op_a = max_op_availability
        super().__init__()

    def _f_op_availabilities(self, t, x, arg_dates):
        a_ops = self.min_op_a + f_gompertz(
            t, self.max_op_a - self.min_op_a, x, self.learn_rate
        )

        return np.array(
            [np.mean(a_ops[arg_dates[i] : d]) for i, d in enumerate(arg_dates[1:])]
        )

    def generate_phase_availabilities(
        self, lifetime_op_availability: float, op_durations: Iterable[float]
    ) -> Iterable[float]:
        """
        Generate operational availabilities for the specified phase durations.

        Parameters
        ----------
        lifetime_op_availability:
            Operational availability averaged over the lifetime
        op_durations:
            Durations of the operational phases [fpy]

        Returns
        -------
        Operational availabilities at each operational phase
        """
        if not self.min_op_a < lifetime_op_availability < self.max_op_a:
            raise FuelCycleError(
                "Input lifetime operational availability must be within the specified"
                " bounds on the phase operational availability."
            )

        op_durations = np.append(0, op_durations)
        total_fpy = np.sum(op_durations)
        cum_fpy = np.cumsum(op_durations)

        t = np.linspace(0, total_fpy, 100)
        arg_dates = np.array([np.argmin(abs(t - i)) for i in cum_fpy])

        def f_opt(x):
            """
            Optimisation objective for chunky fit to Gompertz

            \t:math:`a_{min}+(a_{max}-a_{min})e^{\\dfrac{-\\text{ln}(2)}{e^{-ct_{infl}}}}`
            """
            a_ops_i = self._f_op_availabilities(t, x, arg_dates)
            # NOTE: Fancy analytical integral objective of Gompertz function
            # was a resounding failure. Do not touch this again.
            # The brute force is strong in this one.
            return total_fpy / lifetime_op_availability - sum(op_durations[1:] / a_ops_i)

        x_opt = brentq(f_opt, 0, 10e10)
        return self._f_op_availabilities(t, x_opt, arg_dates)


class OperationalAvailabilityStrategy(abc.ABC):
    """
    Abstract base class for operational availability strategies to generate
    distributions of unplanned outages.
    """

    @abc.abstractmethod
    def generate_distribution(self, n: int, integral: float) -> np.ndarray:
        """
        Generate a distribution with a specified number of entries and integral.

        Parameters
        ----------
        n:
            Number of entries in the distribution
        integral:
            Integral of the distribution

        Returns
        -------
        The distribution of size n and of the correct integral value
        """


class LogNormalAvailabilityStrategy(OperationalAvailabilityStrategy):
    """
    Log-normal distribution strategy
    """

    def __init__(self, sigma: float):
        """
        Parameters
        ----------
        sigma:
            Standard deviation of the underlying normal distribution
        """
        self.sigma = sigma
        super().__init__()

    def generate_distribution(self, n: int, integral: float) -> np.ndarray:
        """
        Generate a log-normal distribution with a specified number of entries and
        integral.

        Parameters
        ----------
        n:
            Number of entries in the distribution
        integral:
            Integral of the distribution

        Returns
        -------
        The distribution of size n and of the correct integral value
        """
        return generate_lognorm_distribution(n, integral, self.sigma)


class TruncNormAvailabilityStrategy(OperationalAvailabilityStrategy):
    """
    Truncated normal distribution strategy
    """

    def __init__(self, sigma: float):
        """
        Parameters
        ----------
        sigma:
            Standard deviation of the underlying normal distribution
        """
        self.sigma = sigma
        super().__init__()

    def generate_distribution(self, n: int, integral: float) -> np.ndarray:
        """
        Generate a truncated normal distribution with a specified number of entries and
        integral.

        Parameters
        ----------
        n:
            Number of entries in the distribution
        integral:
            Integral of the distribution

        Returns
        -------
        The distribution of size n and of the correct integral value
        """
        return generate_truncnorm_distribution(n, integral, self.sigma)


class ExponentialAvailabilityStrategy(OperationalAvailabilityStrategy):
    """
    Exponential distribution strategy
    """

    def __init__(self, lambdda: float):
        """
        Parameters
        ----------
        lambdda:
            Rate of the distribution
        """
        self.lambdda = lambdda
        super().__init__()

    def generate_distribution(self, n: int, integral: float) -> np.ndarray:
        """
        Generate an exponential distribution with a specified number of entries and
        integral.

        Parameters
        ----------
        n
            Number of entries in the distribution
        integral:
            Integral of the distribution

        Returns
        -------
        The distribution of size n and of the correct integral value
        """
        return generate_exponential_distribution(n, integral, self.lambdda)
