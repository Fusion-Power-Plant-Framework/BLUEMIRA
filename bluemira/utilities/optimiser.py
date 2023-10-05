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
Static API to optimisation library
"""
import warnings
from pprint import pformat
from typing import Union

import numpy as np
from scipy.optimize._numdiff import approx_derivative as _approx_derivative  # noqa

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.codes._nlopt_api import NLOPTOptimiser
from bluemira.utilities.tools import is_num

__all__ = ["approx_derivative", "Optimiser"]

warnings.warn(
    f"The module '{__name__}' is deprecated and will be removed in version 2.0.0.\n"
    "See "
    "https://bluemira.readthedocs.io/en/latest/optimisation/optimisation.html "
    "for documentation of the new optimisation module.",
    DeprecationWarning,
    stacklevel=2,
)


def approx_derivative(
    func, x0, method="3-point", rel_step=None, f0=None, bounds=None, args=()
):
    """
    Approximate the gradient of a function about a point.

    Parameters
    ----------
    func: callable
        Function for which to calculate the gradient
    x0: np.ndarray
        Point about which to calculate the gradient
    method: str
        Finite difference method to use
    rel_step: Optional[float, np.ndarray]
        Relative step size to use
    f0: Optional[float, np.ndarray]
        Result of func(x0). If None, this is recomputed
    bounds: Optional[Iterable]
        Lower and upper bounds on individual variables
    args: tuple
        Additional arguments to func
    """
    if bounds is None:
        bounds = (-np.inf, np.inf)
    return _approx_derivative(
        func, x0, method=method, rel_step=rel_step, f0=f0, bounds=bounds, args=args
    )


class Optimiser(NLOPTOptimiser):
    """
    Optimiser interface class.


    Objective functions must be of the form:

    .. code-block:: python

        def f_objective(x, grad, args):
            if grad.size > 0:
                grad[:] = my_gradient_calc(x)
            return my_objective_calc(x)

    The objective function is minimised, so lower values are "better".

    Note that the gradient of the objective function is of the form:

    :math:`\\nabla f = \\bigg[\\dfrac{\\partial f}{\\partial x_0}, \\dfrac{\\partial f}{\\partial x_1}, ...\\bigg]`


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
    """  # noqa: W505

    def optimise(self, x0=None, check_constraints: bool = True):
        """
        Run the optimisation problem.

        Parameters
        ----------
        x0: Optional[np.ndarray]
            Starting solution vector

        Returns
        -------
        x_star: np.ndarray
            Optimal solution vector
        """
        if x0 is None:
            x0 = 0.5 * np.ones(self.n_variables)

        x_star = super().optimise(x0)
        if self.constraints and check_constraints:
            self.check_constraints(x_star)
        return x_star

    def approx_derivative(self, function, x, f0=None):
        """
        Utility function to numerical approximate the derivative of a function.

        Parameters
        ----------
        function: callable
            Function to get a numerical derivative for
        x: np.ndarray
            Point about which to calculate the derivative
        f0: Optional[float]
            Objective function value at x (speed optimisation)

        Notes
        -----
        Use with caution... numerical approximations of gradients can often lead to
        headaches.
        """
        return approx_derivative(
            function, x, bounds=[self.lower_bounds, self.upper_bounds], f0=f0
        )

    def set_objective_function(self, f_objective):
        """
        Set the objective function (minimisation).

        Parameters
        ----------
        f_objective: callable
            Objective function to minimise
        """
        if f_objective is None:
            return

        super().set_objective_function(f_objective)

    def add_eq_constraints(
        self, f_constraint: callable, tolerance: Union[float, np.ndarray]
    ):
        """
        Add a vector-valued equality constraint.

        Parameters
        ----------
        f_constraint: callable
            Constraint function
        tolerance: Union[float, np.ndarray]
            Tolerance with which to enforce the constraint
        """
        if f_constraint is None:
            return
        if is_num(tolerance) and not isinstance(tolerance, np.ndarray):
            tolerance = np.array([tolerance])
        super().add_eq_constraints(f_constraint, tolerance)

    def add_ineq_constraints(
        self, f_constraint: callable, tolerance: Union[float, np.ndarray]
    ):
        """
        Add a vector-valued inequality constraint.

        Parameters
        ----------
        f_constraint: callable
            Constraint function
        tolerance: Union[float, np.ndarray]
            Tolerance array with which to enforce the constraint
        """
        if f_constraint is None:
            return
        if is_num(tolerance) and not isinstance(tolerance, np.ndarray):
            tolerance = np.array([tolerance])
        super().add_ineq_constraints(f_constraint, tolerance)

    def check_constraints(self, x: np.ndarray, warn: bool = True):
        """
        Check that the constraints have been met.

        Parameters
        ----------
        x: np.ndarray
            Solution vector to check the constraints for

        Returns
        -------
        check: bool
            Whether or not the constraints have been satisfied within the tolerances
        """
        c_values = []
        tolerances = []
        for constraint, tolerance in zip(self.constraints, self.constraint_tols):
            c_values.extend(constraint(np.zeros(len(tolerance)), x, np.empty(0)))
            tolerances.extend(tolerance)

        c_values = np.array(c_values)
        tolerances = np.array(tolerances)

        if not np.all(c_values < tolerances):
            if warn:
                indices = np.where(c_values > tolerances)[0]
                message = "\n".join(
                    [
                        f"constraint number {i}: {pformat(c_values[i])} !< "
                        f"{pformat(tolerances[i])}"
                        for i in indices
                    ]
                )
                bluemira_warn(
                    f"Some constraints have not been adequately satisfied.\n{message}"
                )
            return False
        return True
