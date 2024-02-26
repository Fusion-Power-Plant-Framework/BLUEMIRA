# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Base class"""

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


class StructuralComponent(ABC):
    """Abstract base class for a structural component"""

    @abstractmethod
    def Kx(self, *args, **kwargs) -> float:
        """Total equivalent stiffness along x-axis"""
        pass

    @abstractmethod
    def Ky(self, *args, **kwargs) -> float:
        """Total equivalent stiffness along y-axis"""
        pass

    @abstractmethod
    def Xx(self, **kwargs):
        pass

    @abstractmethod
    def Yy(self, **kwargs):
        pass

    @abstractmethod
    def plot(self, ax=None, show: bool = False, homogenized: bool = False):
        """Plotting function for the case. It shall return a matplotlib figure axis"""
        return ax
