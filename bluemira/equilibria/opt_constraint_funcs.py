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
Equilibrium optimisation constraint functions.
for use in NLOpt constrained
optimisation problems.

Constraint functions must be of the form:

.. code-block:: python

    def f_constraint(constraint, x, grad, args):
        constraint[:] = my_constraint_calc(x)
        if grad.size > 0:
            grad[:] = my_gradient_calc(x)
        return constraint

The constraint function convention is such that c <= 0 is sought. I.e. all constraint
values must be negative.

Note that the gradient (Jacobian) of the constraint function is of the form:

.. math::

    \\nabla \\mathbf{c} = \\begin{bmatrix}
            \\dfrac{\\partial c_{0}}{\\partial x_0} & \\dfrac{\\partial c_{0}}{\\partial x_1} & ... \n
            \\dfrac{\\partial c_{1}}{\\partial x_0} & \\dfrac{\\partial c_{1}}{\\partial x_1} & ... \n
            ... & ... & ... \n
            \\end{bmatrix}

The grad and constraint matrices must be assigned in place.

If grad is not updated, the constraint can still be used for derivative-free
optimisation algorithms, but will need to be updated or approximated for use
in derivative based algorithms, such as those utilising gradient descent.
"""  # noqa: W505

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from bluemira.equilibria.equilibrium import Equilibrium

import numpy as np

from bluemira.equilibria import harmonics

warnings.warn(
    f"The module '{__name__}' is deprecated and will be removed in v2.0.0.\n"
    "See "
    "https://bluemira.readthedocs.io/en/latest/optimisation/optimisation.html "
    "for documentation of the new optimisation module.",
    DeprecationWarning,
    stacklevel=2,
)


def objective_constraint(
    constraint: np.ndarray,
    vector: np.ndarray,
    grad: np.ndarray,
    objective_function: Callable[[np.ndarray], np.ndarray],
    maximum_fom: float = 1.0,
) -> np.ndarray:
    """
    Constraint function to constrain the maximum value of an NLOpt objective
    function provided

    Parameters
    ----------
    constraint:
        Constraint vector (modified in place)
    vector:
        Variable vector with which to evaluate the objective function
    grad:
        Constraint Jacobian
    objective_function:
        Objective function to use in constraint
    maximum_fom:
        Value to constrain the objective function by during optimisation

    Returns
    -------
    Updated constraint vector
    """
    constraint[:] = objective_function(vector, grad) - maximum_fom
    return constraint


def Ax_b_constraint(  # noqa: N802
    constraint: np.ndarray,
    vector: np.ndarray,
    grad: np.ndarray,
    a_mat: np.ndarray,
    b_vec: np.ndarray,
    value: float,
    scale: float,
) -> np.ndarray:
    """
    Constraint function of the form:
        A.x - b < value

    Parameters
    ----------
    constraint:
        Constraint array (modified in place)
    vector:
        Variable vector
    grad:
        Constraint Jacobian (modified in place)
    a_mat:
        Response matrix
    b_vec:
        Target value vector
    value:
        Target constraint value
    scale:
        Current scale with which to calculate the constraints

    Returns
    -------
    Updated constraint vector
    """
    constraint[:] = a_mat @ (scale * vector) - b_vec - value
    if grad.size > 0:
        grad[:] = scale * a_mat
    return constraint


def L2_norm_constraint(  # noqa: N802
    constraint: np.ndarray,
    vector: np.ndarray,
    grad: np.ndarray,
    a_mat: np.ndarray,
    b_vec: np.ndarray,
    value: float,
    scale: float,
) -> np.ndarray:
    """
    Constrain the L2 norm of an Ax-b system of equations.
    ||(Ax - b)||² < value

    Parameters
    ----------
    constraint:
        Constraint array (modified in place)
    vector:
        Variable vector
    grad:
        Constraint Jacobian (modified in place)
    A_mat:
        Response matrix
    b_vec:
        Target value vector
    scale:
        Current scale with which to calculate the constraints

    Returns
    -------
    Updated constraint vector
    """
    vector = scale * vector
    residual = a_mat @ vector - b_vec
    constraint[:] = residual.T @ residual - value

    if grad.size > 0:
        grad[:] = 2 * scale * (a_mat.T @ a_mat @ vector - a_mat.T @ b_vec)

    return constraint


def current_midplane_constraint(
    constraint: np.ndarray,
    vector: np.ndarray,
    grad: np.ndarray,
    eq: Equilibrium,
    radius: float,
    scale: float,
    inboard: bool = True,
) -> np.ndarray:
    """
    Constraint function to constrain the inboard or outboard midplane
    of the plasma during optimisation.

    Parameters
    ----------
    constraint:
        Constraint array (modified in place)
    vector:
        Current vector
    grad:
        Constraint Jacobian (modified in place)
    eq:
        Equilibrium to use to fetch last closed flux surface from.
    radius:
        Toroidal radius at which to constrain the plasma midplane.
    scale:
        Current scale with which to calculate the constraints
    inboard:
        Boolean controlling whether to constrain the inboard (if True) or
        outboard (if False) side of the plasma midplane.

    Returns
    -------
    Updated constraint vector
    """
    eq.coilset.get_control_coils().current = vector * scale
    lcfs = eq.get_LCFS()
    if inboard:
        constraint[:] = radius - min(lcfs.x)
    else:
        constraint[:] = max(lcfs.x) - radius
    return constraint


def coil_force_constraints(
    constraint: np.ndarray,
    vector: np.ndarray,
    grad: np.ndarray,
    a_mat: np.ndarray,
    b_vec: np.ndarray,
    n_PF: int,
    n_CS: int,
    PF_Fz_max: float,
    CS_Fz_sum_max: float,
    CS_Fz_sep_max: float,
    scale: float,
) -> np.ndarray:
    """
    Current optimisation force constraints on coils

    Parameters
    ----------
    constraint:
        Constraint array (modified in place)
    vector:
        Current vector
    grad:
        Constraint Jacobian (modified in place)
    a_mat:
        Response matrix block for Fx and Fz
    b_vec:
        Background value vector block for Fx and Fz
    n_PF:
        Number of PF coils
    n_CS:
        Number of CS coils
    PF_Fz_max:
        Maximum vertical force on each PF coil [N]
    CS_Fz_sum_max:
        Maximum total vertical force on the CS stack [N]
    CS_Fz_sep_max:
        Maximum vertical separation force in the CS stack [N]
    scale:
        Current scale with which to calculate the constraints

    Returns
    -------
    Updated constraint vector
    """
    n_coils = len(vector)
    currents = scale * vector

    # get coil force and jacobian
    F = np.zeros((n_coils, 2))
    PF_Fz_max /= scale
    CS_Fz_sep_max /= scale
    CS_Fz_sum_max /= scale

    for i in range(2):  # coil force
        # NOTE: * Hadamard matrix product
        F[:, i] = currents * (a_mat[:, :, i] @ currents + b_vec[:, i])

    F /= scale  # Scale down to MN

    # Absolute vertical force constraint on PF coils
    constraint[:n_PF] = F[:n_PF, 1] ** 2 - PF_Fz_max**2

    if n_CS != 0:
        # vertical forces on CS coils
        cs_fz = F[n_PF:, 1]
        # vertical force on CS stack
        cs_z_sum = np.sum(cs_fz)
        # Absolute sum of vertical force constraint on entire CS stack
        constraint[n_PF] = cs_z_sum**2 - CS_Fz_sum_max**2
        for i in range(n_CS - 1):  # evaluate each gap in CS stack
            # CS separation constraints
            f_sep = np.sum(cs_fz[: i + 1]) - np.sum(cs_fz[i + 1 :])
            constraint[n_PF + 1 + i] = f_sep - CS_Fz_sep_max

    # calculate constraint jacobian
    if grad.size > 0:
        dF = np.zeros((n_coils, n_coils, 2))  # noqa: N806
        im = currents.reshape(-1, 1) @ np.ones((1, n_coils))  # current matrix
        for i in range(2):
            dF[:, :, i] = im * a_mat[:, :, i]
            diag = (
                a_mat[:, :, i] @ currents
                + currents * np.diag(a_mat[:, :, i])
                + b_vec[:, i]
            )
            np.fill_diagonal(dF[:, :, i], diag)

        # Absolute vertical force constraint on PF coils
        grad[:n_PF] = 2 * dF[:n_PF, :, 1]

        if n_CS != 0:
            # Absolute sum of vertical force constraint on entire CS stack
            grad[n_PF] = 2 * np.sum(dF[n_PF:, :, 1], axis=0)

            for i in range(n_CS - 1):  # evaluate each gap in CS stack
                # CS separation constraint Jacobians
                f_up = np.sum(dF[n_PF : n_PF + i + 1, :, 1], axis=0)
                f_down = np.sum(dF[n_PF + i + 1 :, :, 1], axis=0)
                grad[n_PF + 1 + i] = f_up - f_down
    return constraint


def field_constraints(
    constraint: np.ndarray,
    vector: np.ndarray,
    grad: np.ndarray,
    ax_mat: np.ndarray,
    az_mat: np.ndarray,
    bxp_vec: np.ndarray,
    bzp_vec: np.ndarray,
    B_max: np.ndarray,
    scale: float,
) -> np.ndarray:
    """
    Current optimisation poloidal field constraints at prescribed locations

    Parameters
    ----------
    constraint:
        Constraint array (modified in place)
    vector:
        Current vector
    grad:
        Constraint Jacobian (modified in place)
    ax_mat:
        Response matrix for Bx (active coil contributions)
    az_mat:
        Response matrix for Bz (active coil contributions)
    bxp_vec:
        Background vector for Bx (passive coil contributions)
    bzp_vec:
        Background vector for Bz (passive coil contributions)
    B_max:
        Maximum fields inside the coils
    scale:
        Current scale with which to calculate the constraints

    Returns
    -------
    Updated constraint vector
    """
    currents = scale * vector
    Bx_a = ax_mat @ currents
    Bz_a = az_mat @ currents

    B = np.hypot(Bx_a + bxp_vec, Bz_a + bzp_vec)
    if grad.size > 0:
        grad[:] = (
            Bx_a * (Bx_a @ currents + bxp_vec) + Bz_a * (Bz_a @ currents + bzp_vec)
        ) / (B * scale**2)

    constraint[:] = B - B_max
    return constraint


def spherical_harmonics_constraint(
    constraint: np.ndarray,
    vector: np.ndarray,
    grad: np.ndarray,
    ref_harmonics: np.ndarray,
    scale: float,
    eq: Equilibrium,
    r_t: float,
    max_degree: int,
) -> np.ndarray:
    """
    Constraint function to constrain spherical harmonics starting from initial
    coil currents and associated core plasma.

    Parameters
    ----------
    constraint:
        Constraint array (modified in place)
    vector:
        Current vector
    grad:
        Constraint Jacobian (modified in place)
    ref_harmonics:
        Initial harmonic amplitudes obtained from desired core plasma
    scale:
        Current scale with which to calculate the constraints
    eq:
        Equilibrium used to for coilset.
    r_t: float
        Typical length scale of the problem (e.g. radius at outer midplane)
    max_degree:
        Maximum degree of spherical harmonics desired to constrain
    """
    currents = scale * vector

    vector_harmonics_matrix = harmonics.coil_harmonic_amplitude_matrix(
        eq.coilset, max_degree, r_t
    )

    # SH coefficients from function of the current distribution outside of the sphere
    # containing the plasma, i.e., LCFS (r_lcfs)
    # N.B., cannot use coil located within r_lcfs as part of this method.
    vector_harmonics = vector_harmonics_matrix @ currents
    constraint[:] = vector_harmonics - ref_harmonics

    # calculate constraint jacobian
    if grad.size > 0:
        grad[:] = scale * vector_harmonics_matrix

    return constraint
