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
Equilibrium optimisation objective functions.

Objective functions must be of the form:

.. code-block:: python

    class Objective(ObjectiveFunction):

        def f_objective(self, vector: npt.NDArray) -> npt.NDArray:
            return objective_calc(vector)

        def df_objective(self, vector: npt.NDArray) -> npt.NDArray:
            return gradient_calc(vector)

The objective function is minimised, so lower values are "better".

Note that the gradient of the objective function is of the form:

:math:`\\nabla f = \\bigg[\\dfrac{\\partial f}{\\partial x_0}, \\dfrac{\\partial f}{\\partial x_1}, ...\\bigg]`
"""  # noqa: W505, E501

import abc
from typing import Tuple

import numpy as np
import numpy.typing as npt

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.error import EquilibriaError


class ObjectiveFunction(abc.ABC):
    """
    Base class for ObjectiveFunctions

    Notes
    -----
    Optionally the function 'df_objective' can be implemented on any child
    classes to calculate the gradient of the objective function.
    The function should take an `npt.NDArray` as its only argument and
    return only an `npt.NDArray`.
    If the `df_objective` function is not provided and the optimisation algorithm
    is gradient based the approximate derivate is calculated.
    """

    @abc.abstractmethod
    def f_objective(self, vector: npt.NDArray) -> float:
        """Objective function for an optimisation."""


class RegularisedLsqObjective(ObjectiveFunction):
    """
    Least-squares objective with Tikhonov regularisation term.

    Parameters
    ----------
    scale:
        Scaling factor for the vector
    a_mat:
        The 2-D a_mat control matrix A (n, m)
    b_vec:
        The 1-D b vector of target values (n)
    gamma:
        The Tikhonov regularisation parameter.
    """

    def __init__(
        self,
        scale: float,
        a_mat: npt.NDArray,
        b_vec: npt.NDArray,
        gamma: float,
    ) -> None:
        self.scale = scale
        self.a_mat = a_mat
        self.b_vec = b_vec
        self.gamma = gamma

    def f_objective(self, x: npt.NDArray) -> float:
        """Objective function for an optimisation."""
        x = x * self.scale
        fom, _ = regularised_lsq_fom(x, self.a_mat, self.b_vec, self.gamma)
        if fom <= 0:
            raise EquilibriaError(
                "Optimiser least-squares objective function less than zero or nan."
            )
        return fom

    def df_objective(self, x: npt.NDArray) -> npt.NDArray:
        """Gradient of the objective function for an optimisation."""
        x = x * self.scale
        jac = 2 * self.a_mat.T @ self.a_mat @ x / float(len(self.b_vec))
        jac -= 2 * self.a_mat.T @ self.b_vec / float(len(self.b_vec))
        jac += 2 * self.gamma * self.gamma * x
        return self.scale * jac


class CoilCurrentsObjective(ObjectiveFunction):
    """Objective function for the minimisation of the sum of coil currents squared."""

    @staticmethod
    def f_objective(vector: npt.NDArray) -> float:
        """Objective function for an optimisation."""
        return np.sum(vector**2)

    @staticmethod
    def df_objective(vector: npt.NDArray) -> npt.NDArray:
        """Gradient of the objective function for an optimisation."""
        return 2 * vector


class MaximiseFluxObjective(ObjectiveFunction):
    """
    Objective function to maximise flux

    Parameters
    ----------
    c_psi_mat:
        Response matrix of the coil psi contributions to the point at which the flux
        should be maximised
    scale:
        Scaling factor for the vector
    """

    def __init__(self, c_psi_mat: npt.NDArray, scale: float):
        self.c_psi_mat = c_psi_mat
        self.scale = scale

    def f_objective(self, vector: npt.NDArray) -> float:
        """Objective function for an optimisation."""
        return -self.scale * self.c_psi_mat @ vector

    def df_objective(self, vector: npt.NDArray) -> npt.NDArray:  # noqa: ARG002
        """Gradient of the objective function for an optimisation."""
        return -self.scale * self.c_psi_mat


# =============================================================================
# Figures of merit
# =============================================================================


def tikhonov(a_mat: np.ndarray, b_vec: np.ndarray, gamma: float) -> np.ndarray:
    """
    Tikhonov regularisation of Ax-b problem.

    \t:math:`\\textrm{minimise} || Ax - b ||^2 + ||{\\gamma} \\cdot x ||^2`\n
    \t:math:`x = (A^T A + {\\gamma}^2 I)^{-1}A^T b`

    Parameters
    ----------
    a_mat:
        The 2-D A matrix of responses
    b_vec:
        The 1-D b vector of values
    gamma: float
        The Tikhonov regularisation parameter

    Returns
    -------
    x:
        The result vector
    """
    try:
        return np.dot(
            np.linalg.inv(np.dot(a_mat.T, a_mat) + gamma**2 * np.eye(a_mat.shape[1])),
            np.dot(a_mat.T, b_vec),
        )
    except np.linalg.LinAlgError:
        bluemira_warn("Tikhonov singular matrix..!")
        return np.dot(
            np.linalg.pinv(np.dot(a_mat.T, a_mat) + gamma**2 * np.eye(a_mat.shape[1])),
            np.dot(a_mat.T, b_vec),
        )


def regularised_lsq_fom(
    x: np.ndarray, a_mat: np.ndarray, b_vec: np.ndarray, gamma: float
) -> Tuple[float, np.ndarray]:
    """
    Figure of merit for the least squares problem Ax = b, with
    Tikhonov regularisation term. Normalised for the number of
    targets.

    ||(Ax - b)||²/ len(b)] + ||Γx||²

    Parameters
    ----------
    x :
        The 1-D x state vector (m)
    a_mat:
        The 2-D a_mat control matrix A (n, m)
    b_vec:
        The 1-D b vector of target values (n)
    gamma:
        The Tikhonov regularisation parameter.

    Returns
    -------
    fom:
        Figure of merit, explicitly given by
        ||(Ax - b)||²/ len(b)] + ||Γx||²
    residual:
        Residual vector (Ax - b)
    """
    residual = np.dot(a_mat, x) - b_vec
    number_of_targets = float(len(residual))
    fom = residual.T @ residual / number_of_targets + gamma * gamma * x.T @ x

    if fom <= 0:
        raise EquilibriaError("Least-squares objective function less than zero or nan.")
    return fom, residual
