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

    def f_objective(x, grad, args):
        if grad.size > 0:
            grad[:] = my_gradient_calc(x)
        return my_objective_calc(x)

The objective function is minimised, so lower values are "better".

Note that the gradient of the objective function is of the form:

:math:`\\nabla f = \\bigg[\\dfrac{\\partial f}{\\partial x_0}, \\dfrac{\\partial f}{\\partial x_1}, ...\\bigg]`
"""  # noqa (W505)
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from bluemira.base.look_and_feel import bluemira_print_flush
from bluemira.equilibria.error import EquilibriaError
from bluemira.utilities.optimiser import approx_derivative

warnings.warn(
    f"The module '{__name__}' is deprecated and will be removed in v2.0.0.\n"
    "See "
    "https://bluemira.readthedocs.io/en/latest/optimisation/optimisation.html "
    "for documentation of the new optimisation module.",
    DeprecationWarning,
    stacklevel=2,
)


def ad_objective(
    vector: np.ndarray,
    grad: np.ndarray,
    objective: Callable[[np.ndarray], np.ndarray],
    objective_args: Dict[str, Any],
    ad_args: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Objective function that calculates gradient information via
    automatic differentiation of the figure of merit returned from a
    provided objective.

    If the provided objective already provides gradient information,
    it will be overwritten by the approximated gradient.

    Parameters
    ----------
    vector:
        State vector of the array of coil currents.
    grad:
        Local gradient of objective function used by LD NLOPT algorithms.
        Updated in-place.
    objective:
        Objective function for which a numerical approximation for
        derivative information will be calculated.
    objective_args:
        Arguments to pass to objective function during call.
    ad_args:
        Optional keyword arguments to pass to derivative approximation
        function.

    Returns
    -------
    Value of objective function (figure of merit).
    """
    fom = objective(vector, grad, **objective_args)
    if grad.size > 0:
        grad[:] = approx_derivative(
            objective, vector, args=objective_args, f0=fom, **ad_args
        )
    bluemira_print_flush(f"EQUILIBRIA Coilset iteration figure of merit = {fom:.2e}")
    return fom


def regularised_lsq_objective(
    vector: np.ndarray,
    grad: np.ndarray,
    scale: float,
    a_mat: np.ndarray,
    b_vec: np.ndarray,
    gamma: float,
) -> float:
    """
    Objective function for nlopt optimisation (minimisation),
    consisting of a least-squares objective with Tikhonov
    regularisation term, which updates the gradient in-place.

    Parameters
    ----------
    vector:
        State vector of the array of coil currents (m)
    grad:
        Local gradient of objective function used by LD NLOPT algorithms
        Updated in-place (n, m).
    scale: float
        Scaling factor for the vector
    a_mat:
        The 2-D a_mat control matrix A (n, m)
    b_vec:
        The 1-D b vector of target values (n)
    gamma:
        The Tikhonov regularisation parameter.

    Returns
    -------
    Value of objective function (figure of merit).
    """
    vector = vector * scale
    fom, err = regularised_lsq_fom(vector, a_mat, b_vec, gamma)
    if grad.size > 0:
        jac = 2 * a_mat.T @ a_mat @ vector / float(len(b_vec))
        jac -= 2 * a_mat.T @ b_vec / float(len(b_vec))
        jac += 2 * gamma * gamma * vector
        grad[:] = scale * jac
    if fom <= 0:
        raise EquilibriaError(
            "Optimiser least-squares objective function less than zero or nan."
        )
    return fom


def minimise_coil_currents(vector: np.ndarray, grad: np.ndarray) -> float:
    """
    Objective function for the minimisation of the sum of coil currents squared

    Parameters
    ----------
    vector:
        State vector of the array of coil currents.
    grad:
        Local gradient of objective function used by LD NLOPT algorithms.
        Updated in-place.

    Returns
    -------
    Sum of the currents squared.
    """
    sum_sq_currents = np.sum(vector**2)

    if grad.size > 0:
        grad[:] = 2 * vector

    return sum_sq_currents


def maximise_flux(
    vector: np.ndarray, grad: np.ndarray, c_psi_mat: np.ndarray, scale: float
) -> float:
    """
    Objective function to maximise flux

    Parameters
    ----------
    vector:
        State vector of the array of coil currents.
    grad:
        Local gradient of objective function used by LD NLOPT algorithms.
        Updated in-place.
    c_psi_mat:
        Response matrix of the coil psi contributions to the point at which the flux
        should be maximised
    scale:
        Scaling factor for the vector

    Returns
    -------
    Psi value at the point
    """
    psi = -scale * c_psi_mat @ vector
    if grad.size > 0:
        grad[:] = -scale * c_psi_mat

    return psi


# =============================================================================
# Figures of merit
# =============================================================================


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
