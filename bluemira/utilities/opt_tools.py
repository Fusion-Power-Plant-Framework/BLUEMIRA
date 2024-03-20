# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Optimisation utilities
"""

import warnings

import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.utilities.tools import floatify
from bluemira.utilities.error import InternalOptError

warnings.warn(
    f"The module '{__name__}' is deprecated and will be removed in v2.0.0.\n"
    "See "
    "https://bluemira.readthedocs.io/en/latest/optimisation/optimisation.html "
    "for documentation of the new optimisation module.",
    DeprecationWarning,
    stacklevel=2,
)


# =============================================================================
# Analytical objective functions
# =============================================================================


def tikhonov(A, b, gamma):
    """
    Tikhonov regularisation of Ax-b problem.

    \t:math:`\\textrm{minimise} || Ax - b ||^2 + ||{\\gamma} \\cdot x ||^2`\n
    \t:math:`x = (A^T A + {\\gamma}^2 I)^{-1}A^T b`

    Parameters
    ----------
    A: np.array(n, m)
        The 2-D A matrix of responses
    b: np.array(n)
        The 1-D b vector of values
    gamma: float
        The Tikhonov regularisation parameter

    Returns
    -------
    x: np.array(m)
        The result vector
    """
    try:
        return np.dot(
            np.linalg.inv(np.dot(A.T, A) + gamma**2 * np.eye(A.shape[1])),
            np.dot(A.T, b),
        )
    except np.linalg.LinAlgError:
        bluemira_warn("Tikhonov singular matrix..!")
        return np.dot(
            np.linalg.pinv(np.dot(A.T, A) + gamma**2 * np.eye(A.shape[1])),
            np.dot(A.T, b),
        )


def regularised_lsq_fom(x, A, b, gamma):
    """
    Figure of merit for the least squares problem Ax = b, with
    Tikhonov regularisation term. Normalised for the number of
    targets.

    ||(Ax - b)||²/ len(b)] + ||Γx||²

    Parameters
    ----------
    x : np.array(m)
        The 1-D x state vector.
    A: np.array(n, m)
        The 2-D A control matrix
    b: np.array(n)
        The 1-D b vector of target values
    gamma: float
        The Tikhonov regularisation parameter.

    Returns
    -------
    fom: float
        Figure of merit, explicitly given by
        ||(Ax - b)||²/ len(b)] + ||Γx||²
    residual: np.array(n)
        Residual vector (Ax - b)
    """
    residual = np.dot(A, x) - b
    number_of_targets = floatify(len(residual))
    fom = residual.T @ residual / number_of_targets + gamma * gamma * x.T @ x

    if fom <= 0:
        raise bluemira_warn("Least-squares objective function less than zero or nan.")
    return fom, residual


def least_squares(A, b):
    """
    Least squares optimisation.

    \t:math:`\\textrm{minimise} || Ax - b ||^{2}_{2}`\n
    \t:math:`x = (A^T A)^{-1}A^T b`

    Parameters
    ----------
    A: np.array(n, m)
        The 2-D A matrix of responses
    b: np.array(n)
        The 1-D b vector of values

    Returns
    -------
    x: np.array(m)
        The result vector
    """
    return np.linalg.solve(A, b)


# =============================================================================
# Scipy interface
# =============================================================================


def process_scipy_result(res):
    """
    Handle a scipy.minimize OptimizeResult object. Process error codes, if any.

    Parameters
    ----------
    res: OptimizeResult

    Returns
    -------
    x: np.array
        The optimal set of parameters (result of the optimisation)

    Raises
    ------
    InternalOptError if an error code returned without a usable result.
    """
    if res.success:
        return res.x

    if not hasattr(res, "status"):
        bluemira_warn("Scipy optimisation was not succesful. Failed without status.")
        raise InternalOptError(f"{res.message}\n{res.__str__()}")

    elif res.status == 8:
        # This can happen when scipy is not convinced that it has found a minimum.
        bluemira_warn(
            f"{res.message}\nOptimiser (scipy) found a positive directional"
            f" derivative,\nreturning suboptimal result. \n\n{res.__str__()}"
        )
        return res.x

    elif res.status == 9:
        bluemira_warn(
            f"{res.message}\nOptimiser (scipy) exceeded number of iterations, returning"
            f" suboptimal result. \n\n{res.__str__()}"
        )
        return res.x

    else:
        raise InternalOptError(f"{res.message}\n{res.__str__()}")
