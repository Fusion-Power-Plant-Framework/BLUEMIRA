# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Utils for magnets"""

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


def serie_r(arr: Union[List, np.array]):
    """
    Compute the serie (as for resistance)

    Parameters
    ----------
    arr:
        list or numpy array containing the elements on which the serie shall
        be calculated

    Returns
    -------
    Result: float
    """
    out = np.sum(arr)
    return out


def parall_r(arr: Union[List, np.array]):
    """
    Compute the parallel (as for resistance)

    Parameters
    ----------
    arr:
        list or numpy array containing the elements on which the parallel
        shall be calculated

    Returns
    -------
    Result: float
    """
    out = 0
    for i in range(len(arr)):
        out += 1 / arr[i]
    out = out ** -1
    return out


def serie_k(arr: Union[List, np.array]):
    """
    Compute the serie (as for spring)

    Parameters
    ----------
    arr:
        list or numpy array containing the elements on which the serie
        shall be calculated

    Returns
    -------
    Result: float
    """
    out = 0
    for i in range(len(arr)):
        out += 1 / arr[i]
    out = out ** -1
    return out


def parall_k(arr: Union[List, np.array]):
    """
    Compute the parallel (as for spring)

    Parameters
    ----------
    arr:
        list or numpy array containing the elements on which the parallel
        shall be calculated

    Returns
    -------
    Result: float
    """
    out = np.sum(arr)
    return out


class StructuralModelHooke(ABC):
    """Abstract base class for a """

    @abstractmethod
    def Kx(self, *args, **kwargs) -> float:
        """Total equivalent stiffness along x-axis"""
        pass

    @abstractmethod
    def Ky(self, *args, **kwargs) -> float:
        """Total equivalent stiffness along y-axis"""
        pass


class StructuralComponent(ABC):
    """Abstract base class for a structural component"""

    @property
    @abstractmethod
    def structural_model(self, *args, **kwargs):
        """Structural model used evaluate stress/strain into the component"""
        pass


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
