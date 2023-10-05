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
Equilibrium optimisation constraint classes
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union

if TYPE_CHECKING:
    from bluemira.equilibria.coils import CoilSet
    from bluemira.equilibria.equilibrium import Equilibrium

import matplotlib.patches as patch
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from bluemira.equilibria.harmonics import coil_harmonic_amplitude_matrix
from bluemira.equilibria.optimisation.constraint_funcs import (
    AxBConstraint,
    ConstraintFunction,
    FieldConstraintFunction,
    L2NormConstraint,
)
from bluemira.equilibria.optimisation.constraint_funcs import (
    CoilForceConstraint as CoilForceConstraintFunction,
)
from bluemira.equilibria.plotting import ConstraintPlotter
from bluemira.geometry.coordinates import interpolate_points
from bluemira.utilities.tools import abs_rel_difference, is_num


def _get_dummy_equilibrium(equilibrium: Equilibrium):
    """
    Get a dummy equilibrium for current optimisation where the background response is
    solely due to the plasma and passive coils.

    Notes
    -----
    When we do dI (current gradient) optimisation, the background vector includes the
    contributions from the whole coilset (including active coils).

    When we do I (current vector) optimisation, the background vector only includes
    contributions from the passive coils (plasma).
    """
    # TODO: Add passive coil contributions here
    dummy = equilibrium.plasma
    dummy.coilset = deepcopy(equilibrium.coilset)
    return dummy


class UpdateableConstraint(ABC):
    """
    Abstract base mixin class for an equilibrium optimisation constraint that is
    updateable.
    """

    @abstractmethod
    def prepare(self, equilibrium: Equilibrium, I_not_dI=False, fixed_coils=False):
        """
        Prepare the constraint for use in an equilibrium optimisation problem.
        """

    @abstractmethod
    def control_response(self, coilset: CoilSet):
        """
        Calculate control response of a CoilSet to the constraint.
        """

    @abstractmethod
    def evaluate(self, equilibrium: Equilibrium):
        """
        Calculate the value of the constraint in an Equilibrium.
        """

    @abstractmethod
    def f_constraint(self) -> ConstraintFunction:
        """The numerical non-linear part of the constraint."""


class FieldConstraints(UpdateableConstraint):
    """
    Inequality constraints for the poloidal field at certain locations.

    Parameters
    ----------
    x:
        Radial coordinate(s) at which to constrain the poloidal field
    z:
        Vertical coordinate(s) at which to constrain the poloidal field
    B_max:
        Maximum poloidal field value(s) at location(s)
    tolerance:
        Tolerance with which the constraint(s) will be met
    constraint_type:
        Type of constraint
    """

    def __init__(
        self,
        x: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        B_max: Union[float, np.ndarray],
        tolerance: Union[float, np.ndarray] = 1.0e-6,
        constraint_type: str = "inequality",
    ):
        if is_num(x):
            x = np.array([x])
        if is_num(z):
            z = np.array([z])

        if is_num(B_max):
            B_max = B_max * np.ones(len(x))
        if len(B_max) != len(x):
            raise ValueError(
                "Maximum field vector length not equal to the number of points."
            )

        if is_num(tolerance):
            tolerance = tolerance * np.ones(len(x))
        if len(tolerance) != len(x):
            raise ValueError("Tolerance vector length not equal to the number of coils.")

        self.x = x
        self.z = z
        self._args = {
            "ax_mat": None,
            "az_mat": None,
            "bxp_vec": None,
            "bzp_vec": None,
            "B_max": B_max,
            "scale": 1.0,
        }
        self.tolerance = tolerance
        self.f_constraint_type = constraint_type

    def prepare(
        self, equilibrium: Equilibrium, I_not_dI: bool = False, fixed_coils: bool = False
    ):
        """
        Prepare the constraint for use in an equilibrium optimisation problem.
        """
        if I_not_dI:
            equilibrium = _get_dummy_equilibrium(equilibrium)

        # Re-build control response matrix
        if not fixed_coils or (fixed_coils and self._args["ax_mat"] is None):
            ax_mat, az_mat = self.control_response(equilibrium.coilset)
            self._args["ax_mat"] = ax_mat
            self._args["az_mat"] = az_mat

        bxp_vec, bzp_vec = self.evaluate(equilibrium)
        self._args["bxp_vec"] = bxp_vec
        self._args["bzp_vec"] = bzp_vec

    def control_response(self, coilset: CoilSet) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate control response of a CoilSet to the constraint.
        """
        return (
            coilset.Bx_response(self.x, self.z, control=True),
            coilset.Bz_response(self.x, self.z, control=True),
        )

    def evaluate(self, equilibrium: Equilibrium) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the value of the constraint in an Equilibrium.
        """
        Bx = np.atleast_1d(equilibrium.Bx(self.x, self.z))
        Bz = np.atleast_1d(equilibrium.Bz(self.x, self.z))
        return Bx, Bz

    def f_constraint(self) -> FieldConstraintFunction:
        """Calculate the constraint function"""
        f_constraint = FieldConstraintFunction(**self._args)
        f_constraint.constraint_type = self.f_constraint_type
        return f_constraint

    def __len__(self) -> int:
        """
        Length of field constraints.
        """
        return len(self.x)


class CoilFieldConstraints(FieldConstraints):
    """
    Inequality constraints on the poloidal field at the middle of the inside edge
    of the coils, where the field is usually highest.

    Parameters
    ----------
    coilset:
        Coilset for which to constrain the fields in the coils
    B_max:
        Maximum field allowed in the coils
    tolerance:
        Tolerance with which the inequality constraints will be met

    Notes
    -----
    This is a fast approximation constraint, and does not solve for the peak field
    at all points in the coils. Use with caution.
    TODO: Presently only handles CoilSets with Coils (SymmetricCircuits not yet
    supported)
    TODO: Presently only accounts for poloidal field contributions from PF coils and
    plasma (TF from TF coils not accounted for if PF coils are inside the TF coils.)
    """

    def __init__(
        self,
        coilset: CoilSet,
        B_max: Union[float, np.ndarray],
        tolerance: Union[float, np.ndarray] = 1.0e-6,
    ):
        n_coils = coilset.n_coils()
        if is_num(B_max):
            B_max = B_max * np.ones(n_coils)
        if len(B_max) != n_coils:
            raise ValueError(
                "Maximum field vector length not equal to the number of coils."
            )

        if is_num(tolerance):
            tolerance = tolerance * np.ones(n_coils)
        if len(tolerance) != n_coils:
            raise ValueError("Tolerance vector length not equal to the number of coils.")

        x, z = self._get_constraint_points(coilset)

        super().__init__(x, z, B_max, tolerance=tolerance, constraint_type="inequality")

    @staticmethod
    def _get_constraint_points(coilset):
        return coilset.x - coilset.dx, coilset.z

    def prepare(
        self, equilibrium: Equilibrium, I_not_dI: bool = False, fixed_coils: bool = False
    ):
        """
        Prepare the constraint for use in an equilibrium optimisation problem.
        """
        if I_not_dI:
            equilibrium = _get_dummy_equilibrium(equilibrium)

        # Re-build control response matrix
        if not fixed_coils or (fixed_coils and self._args["ax_mat"] is None):
            # Update the target points for the constraints (the coils may be moving)
            self.x, self.z = self._get_constraint_points(equilibrium.coilset)
            ax_mat, az_mat = self.control_response(equilibrium.coilset)
            self._args["ax_mat"] = ax_mat
            self._args["az_mat"] = az_mat

        bxp_vec, bzp_vec = self.evaluate(equilibrium)
        self._args["bxp_vec"] = bxp_vec
        self._args["bzp_vec"] = bzp_vec


class CoilForceConstraints(UpdateableConstraint):
    """
    Inequality constraints on the vertical forces in the PF and CS coils.

    Parameters
    ----------
    coilset:
        Coilset for which to constrain the fields in the coils
    PF_Fz_max:
        Maximum absolute vertical force in a PF coil [MN]
    CS_Fz_sum_max:
        Maximum absolute vertical force sum in the CS stack [MN]
    CS_Fz_sep_max:
        Maximum separation vertical force between two CS modules [MN]
    tolerance:
        Tolerance with which the inequality constraints will be met

    Notes
    -----
    TODO: Presently only handles CoilSets with Coils (SymmetricCircuits not yet
    supported)
    """

    def __init__(
        self,
        coilset: CoilSet,
        PF_Fz_max: float,
        CS_Fz_sum_max: float,
        CS_Fz_sep_max: float,
        tolerance: Union[float, np.ndarray] = 1.0e-6,
    ):
        n_PF = coilset.n_coils("PF")
        n_CS = coilset.n_coils("CS")
        n_f_constraints = n_PF + n_CS

        if is_num(tolerance):
            tolerance = tolerance * np.ones(n_f_constraints)
        elif len(tolerance) != n_f_constraints:
            raise ValueError(f"Tolerance vector not of length {n_f_constraints}")

        self._args = {
            "a_mat": None,
            "b_vec": None,
            "scale": 1.0,
            "PF_Fz_max": PF_Fz_max,
            "CS_Fz_sum_max": CS_Fz_sum_max,
            "CS_Fz_sep_max": CS_Fz_sep_max,
            "n_PF": n_PF,
            "n_CS": n_CS,
        }
        self.tolerance = tolerance

    def prepare(
        self, equilibrium: Equilibrium, I_not_dI: bool = False, fixed_coils: bool = False
    ):
        """
        Prepare the constraint for use in an equilibrium optimisation problem.
        """
        if I_not_dI:
            equilibrium = _get_dummy_equilibrium(equilibrium)

        # Re-build control response matrix
        if not fixed_coils or (fixed_coils and self._args["a_mat"] is None):
            self._args["a_mat"] = self.control_response(equilibrium.coilset)

        self._args["b_vec"] = self.evaluate(equilibrium)

    @staticmethod
    def control_response(coilset: CoilSet) -> np.ndarray:
        """
        Calculate control response of a CoilSet to the constraint.
        """
        return coilset.control_F(coilset)

    @staticmethod
    def evaluate(equilibrium: Equilibrium) -> np.ndarray:
        """
        Calculate the value of the constraint in an Equilibrium.
        """
        fp = np.zeros((equilibrium.coilset.n_coils(), 2))
        current = equilibrium.coilset.current
        non_zero = np.nonzero(current)[0]
        if non_zero.size:
            fp[non_zero] = (
                equilibrium.coilset.F(equilibrium)[non_zero] / current[non_zero][:, None]
            )
        return fp

    def f_constraint(self) -> CoilForceConstraintFunction:
        """Calculate the constraint function"""
        return CoilForceConstraintFunction(**self._args)


class MagneticConstraint(UpdateableConstraint):
    """
    Abstract base class for a magnetic optimisation constraint.

    Can be used as a standalone constraint for use in an optimisation problem. In which
    case the constraint is of the form: ||(Ax - b)||² < target_value

    Can be used in a MagneticConstraintSet
    """

    def __init__(
        self,
        target_value: float = 0.0,
        weights: Union[float, np.ndarray] = 1.0,
        tolerance: Union[float, np.ndarray] = 1e-6,
        f_constraint: Type[ConstraintFunction] = L2NormConstraint,
        constraint_type: str = "inequality",
    ):
        self.target_value = target_value * np.ones(len(self))
        if is_num(tolerance):
            if f_constraint == L2NormConstraint:
                tolerance = tolerance * np.ones(1)
            else:
                tolerance = tolerance * np.ones(len(self))
        self.weights = weights
        self._f_constraint = f_constraint
        self._args = {"a_mat": None, "b_vec": None, "value": 0.0, "scale": 1.0}
        self.tolerance = tolerance
        self.constraint_type = constraint_type

    def prepare(
        self, equilibrium: Equilibrium, I_not_dI: bool = False, fixed_coils: bool = False
    ):
        """
        Prepare the constraint for use in an equilibrium optimisation problem.
        """
        if I_not_dI:
            equilibrium = _get_dummy_equilibrium(equilibrium)

        # Re-build control response matrix
        if not fixed_coils or (fixed_coils and self._args["a_mat"] is None):
            self._args["a_mat"] = self.control_response(equilibrium.coilset)

        self.update_target(equilibrium)
        self._args["b_vec"] = self.target_value - self.evaluate(equilibrium)

    def update_target(self, equilibrium: Equilibrium):
        """
        Update the target value of the magnetic constraint.
        """

    @abstractmethod
    def plot(self, ax):
        """
        Plot the constraint onto an Axes.
        """

    def __len__(self) -> int:
        """
        The mathematical size of the constraint.

        Notes
        -----
        Length of the array if an array is specified, otherwise 1 for a float.
        """
        return len(self.x) if hasattr(self.x, "__len__") else 1

    def f_constraint(self) -> ConstraintFunction:
        """Return the non-linear, numerical, part of the constraint."""
        f_constraint = self._f_constraint(**self._args)
        f_constraint.constraint_type = self.constraint_type
        return f_constraint


class SphericalHarmonicsConstraint(MagneticConstraint):
    """
    Constraint function to constrain spherical harmonics starting from initial
    coil currents and associated core plasma.

    Parameters
    ----------
    r_t:
        Typical length scale of the problem (e.g. radius at outer midplane)
    ref_coil_psi_amplitudes:
        Initial harmonic amplitudes obtained from desired core plasma
    max_degree:
        Maximum degree of spherical harmonics desired to constrain.
    """

    def __init__(
        self,
        r_t: float,
        ref_coil_psi_amplitudes: npt.NDArray,
        max_degree: int,
        target_value: float = 0.0,
        weights: Union[float, npt.NDArray] = 1.0,
        tolerance: Union[float, npt.NDArray] = 1e-6,
        f_constraint: Type[ConstraintFunction] = AxBConstraint,
        constraint_type: str = "inequality",
    ):
        super().__init__(
            target_value,
            weights,
            tolerance=tolerance,
            f_constraint=f_constraint,
            constraint_type=constraint_type,
        )
        self.max_degree = max_degree
        self.r_t = r_t
        self.ref_coil_psi_amplitudes = ref_coil_psi_amplitudes

    def control_response(self, coilset: CoilSet) -> npt.NDArray:
        """
        Calculate control response of a CoilSet to the constraint.
        """
        return coil_harmonic_amplitude_matrix(coilset, self.max_degree, self.r_t)

    def evaluate(self, _: Equilibrium) -> npt.NDArray:
        """
        Calculate the value of the constraint in an Equilibrium.
        """
        return self.ref_coil_psi_amplitudes

    def __len__(self) -> int:
        """
        The mathematical size of the constraint.
        """
        return len(self.ref_coil_psi_amplitudes)

    def plot(self, ax=None):
        """
        Plot the constraint onto an Axes.
        """
        if ax is None:
            _, ax = plt.subplots()

        ax.add_patch(patch.Circle((0, 0), self.r_t, ec="orange", fill=True, fc="orange"))


class AbsoluteMagneticConstraint(MagneticConstraint):
    """
    Abstract base class for absolute magnetic constraints, where the target
    value is prescribed in absolute terms.
    """

    def __init__(
        self,
        x: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        target_value: float,
        weights: Union[float, np.ndarray] = 1.0,
        tolerance: Union[float, np.ndarray] = 1e-6,
        f_constraint: Type[ConstraintFunction] = AxBConstraint,
        constraint_type: str = "equality",
    ):
        self.x = x
        self.z = z
        super().__init__(
            target_value,
            weights,
            tolerance=tolerance,
            f_constraint=f_constraint,
            constraint_type=constraint_type,
        )


class RelativeMagneticConstraint(MagneticConstraint):
    """
    Abstract base class for relative magnetic constraints, where the target
    value is prescribed with respect to a reference point.
    """

    def __init__(
        self,
        x: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        ref_x: float,
        ref_z: float,
        constraint_value: float = 0.0,
        weights: Union[float, np.ndarray] = 1.0,
        tolerance: Union[float, np.ndarray] = 1e-6,
        f_constraint: Type[ConstraintFunction] = L2NormConstraint,
        constraint_type: str = "inequality",
    ):
        self.x = x
        self.z = z
        self.ref_x = ref_x
        self.ref_z = ref_z
        super().__init__(
            0.0,
            weights,
            tolerance=tolerance,
            f_constraint=f_constraint,
            constraint_type=constraint_type,
        )
        self._args["value"] = constraint_value

    @abstractmethod
    def update_target(self, equilibrium: Equilibrium):
        """
        Update the target value of the magnetic constraint.
        """


class FieldNullConstraint(AbsoluteMagneticConstraint):
    """
    Magnetic field null constraint. In practice sets the Bx and Bz field components
    to be 0 at the specified location.
    """

    def __init__(
        self,
        x: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        weights: Union[float, np.ndarray] = 1.0,
        tolerance: float = 1e-6,
    ):
        super().__init__(
            x,
            z,
            target_value=0.0,
            weights=weights,
            tolerance=tolerance,
            constraint_type="inequality",
            f_constraint=L2NormConstraint,
        )

    def control_response(self, coilset: CoilSet) -> np.ndarray:
        """
        Calculate control response of a CoilSet to the constraint.
        """
        return np.vstack(
            [
                coilset.Bx_response(self.x, self.z, control=True),
                coilset.Bz_response(self.x, self.z, control=True),
            ]
        )

    def evaluate(self, eq: Equilibrium) -> np.ndarray:
        """
        Calculate the value of the constraint in an Equilibrium.
        """
        return np.array([eq.Bx(self.x, self.z), eq.Bz(self.x, self.z)])

    def plot(self, ax):
        """
        Plot the constraint onto an Axes.
        """
        kwargs = {
            "marker": "X",
            "color": "b",
            "markersize": 10,
            "zorder": 45,
            "linestyle": "None",
        }
        ax.plot(self.x, self.z, **kwargs)

    def __len__(self) -> int:
        """
        The mathematical size of the constraint.
        """
        return 2


class PsiConstraint(AbsoluteMagneticConstraint):
    """
    Absolute psi value constraint.
    """

    def __init__(
        self,
        x: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        target_value: float,
        weights: Union[float, np.ndarray] = 1.0,
        tolerance: Union[float, np.ndarray] = 1e-6,
    ):
        super().__init__(
            x,
            z,
            target_value,
            weights=weights,
            tolerance=tolerance,
            f_constraint=AxBConstraint,
            constraint_type="equality",
        )

    def control_response(self, coilset: CoilSet) -> np.ndarray:
        """
        Calculate control response of a CoilSet to the constraint.
        """
        return coilset.psi_response(self.x, self.z, control=True)

    def evaluate(self, eq: Equilibrium) -> np.ndarray:
        """
        Calculate the value of the constraint in an Equilibrium.
        """
        return eq.psi(self.x, self.z)

    def plot(self, ax):
        """
        Plot the constraint onto an Axes.
        """
        kwargs = {"marker": "s", "markersize": 8, "color": "b", "linestyle": "None"}
        ax.plot(self.x, self.z, **kwargs)


class IsofluxConstraint(RelativeMagneticConstraint):
    """
    Isoflux constraint for a set of points relative to a reference point.
    """

    def __init__(
        self,
        x: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        ref_x: float,
        ref_z: float,
        constraint_value: float = 0.0,
        weights: Union[float, np.ndarray] = 1.0,
        tolerance: float = 1e-6,
    ):
        super().__init__(
            x,
            z,
            ref_x,
            ref_z,
            constraint_value,
            weights=weights,
            f_constraint=L2NormConstraint,
            tolerance=tolerance,
            constraint_type="inequality",
        )

    def control_response(self, coilset: CoilSet) -> np.ndarray:
        """
        Calculate control response of a CoilSet to the constraint.
        """
        return coilset.psi_response(self.x, self.z, control=True) - coilset.psi_response(
            self.ref_x, self.ref_z, control=True
        )

    def evaluate(self, eq: Equilibrium) -> np.ndarray:
        """
        Calculate the value of the constraint in an Equilibrium.
        """
        return eq.psi(self.x, self.z)

    def update_target(self, eq: Equilibrium):
        """
        We need to update the target value, as it is a relative constraint.
        """
        self.target_value = float(eq.psi(self.ref_x, self.ref_z))

    def plot(self, ax):
        """
        Plot the constraint onto an Axes.
        """
        kwargs = {
            "marker": "o",
            "markeredgewidth": 3,
            "markeredgecolor": "b",
            "markersize": 10,
            "linestyle": "None",
            "markerfacecolor": "None",
            "zorder": 45,
        }
        ax.plot(self.x, self.z, **kwargs)
        kwargs["markeredgewidth"] = 5
        ax.plot(self.ref_x, self.ref_z, **kwargs)


class PsiBoundaryConstraint(AbsoluteMagneticConstraint):
    """
    Absolute psi value constraint on the plasma boundary. Gets updated when
    the plasma boundary flux value is changed.
    """

    def __init__(
        self,
        x: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        target_value: float,
        weights: Union[float, np.ndarray] = 1.0,
        tolerance: Union[float, np.ndarray] = 1e-6,
    ):
        super().__init__(
            x,
            z,
            target_value,
            weights,
            tolerance,
            f_constraint=L2NormConstraint,
            constraint_type="inequality",
        )

    def control_response(self, coilset: CoilSet) -> np.ndarray:
        """
        Calculate control response of a CoilSet to the constraint.
        """
        return coilset.psi_response(self.x, self.z, control=True)

    def evaluate(self, eq: Equilibrium) -> np.ndarray:
        """
        Calculate the value of the constraint in an Equilibrium.
        """
        return eq.psi(self.x, self.z)

    def plot(self, ax):
        """
        Plot the constraint onto an Axes.
        """
        kwargs = {"marker": "o", "markersize": 8, "color": "b", "linestyle": "None"}
        ax.plot(self.x, self.z, **kwargs)


class MagneticConstraintSet(ABC):
    """
    A set of magnetic constraints to be applied to an equilibrium. The optimisation
    problem is of the form:

        [A][x] = [b]

    where:

        [b] = [target] - [background]

    The target vector is the vector of desired values. The background vector
    is the vector of values due to uncontrolled current terms (plasma and passive
    coils).


    Use of class:

        - Inherit from this class
        - Add a __init__(args) method
        - Populate constraints with super().__init__(List[MagneticConstraint])
    """

    __slots__ = ["constraints", "eq", "coilset", "A", "w", "target", "background"]

    def __init__(self, constraints: List[MagneticConstraint]):
        self.constraints = constraints
        self.eq = None
        self.A = None
        self.target = None
        self.background = None

    def __call__(
        self, equilibrium: Equilibrium, I_not_dI: bool = False, fixed_coils: bool = False
    ):
        """
        Update the MagneticConstraintSet
        """
        if I_not_dI:
            equilibrium = _get_dummy_equilibrium(equilibrium)

        self.eq = equilibrium
        self.coilset = equilibrium.coilset

        # Update relative magnetic constraints without updating A matrix
        for constraint in self.constraints:
            if isinstance(constraint, RelativeMagneticConstraint):
                constraint.update_target(equilibrium)

        # Re-build control response matrix
        if not fixed_coils or (fixed_coils and self.A is None):
            self.build_control_matrix()
            self.build_target()

        self.build_background()
        self.build_weight_matrix()

    def __len__(self) -> int:
        """
        The mathematical size of the constraint set.
        """
        return sum([len(c) for c in self.constraints])

    def get_weighted_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get [A] and [b] scaled by weight matrix.
        Weight matrix assumed to be diagonal.
        """
        weights = self.w
        weighted_a = weights[:, np.newaxis] * self.A
        weighted_b = weights * self.b
        return weights, weighted_a, weighted_b

    def build_weight_matrix(self):
        """
        Build the weight matrix used in optimisation.
        Assumed to be diagonal.
        """
        self.w = np.zeros(len(self))

        i = 0
        for constraint in self.constraints:
            n = len(constraint)
            self.w[i : i + n] = constraint.weights
            i += n

    def build_control_matrix(self):
        """
        Build the control response matrix used in optimisation.
        """
        self.A = np.zeros((len(self), len(self.coilset.control)))

        i = 0
        for constraint in self.constraints:
            n = len(constraint)
            self.A[i : i + n, :] = constraint.control_response(self.coilset)
            i += n

    def build_target(self):
        """
        Build the target value vector.
        """
        self.target = np.zeros(len(self))

        i = 0
        for constraint in self.constraints:
            n = len(constraint)
            self.target[i : i + n] = constraint.target_value * np.ones(n)
            i += n

    def build_background(self):
        """
        Build the background value vector.
        """
        self.background = np.zeros(len(self))

        i = 0
        for constraint in self.constraints:
            n = len(constraint)
            self.background[i : i + n] = constraint.evaluate(self.eq)
            i += n

    @property
    def b(self) -> np.ndarray:
        """
        The b vector of target - background values.
        """
        return self.target - self.background

    def update_psi_boundary(self, psi_bndry: float):
        """
        Update the target value for all PsiBoundaryConstraints.

        Parameters
        ----------
        psi_bndry:
            The target psi boundary value [V.s/rad]
        """
        for constraint in self.constraints:
            if isinstance(constraint, PsiBoundaryConstraint):
                constraint.target_value = psi_bndry
        self.build_target()

    def plot(self, ax=None):
        """
        Plots constraints
        """
        return ConstraintPlotter(self, ax=ax)


class AutoConstraints(MagneticConstraintSet):
    """
    Utility class for crude reconstruction of magnetic constraints from a
    specified LCFS set of coordinates.

    Parameters
    ----------
    x:
        The x coordinates of the LCFS
    z:
        The z coordinates of the LCFS
    psi_boundary:
        The psi boundary value to use as a constraint. If None, an
        isoflux constraint is used.
    n_points:
        The number of interpolated points to use
    """

    def __init__(
        self,
        x: np.ndarray,
        z: np.ndarray,
        psi_boundary: Optional[float] = None,
        n_points: int = 40,
    ):
        x = np.array(x)
        z = np.array(z)
        z_max = max(z)
        z_min = min(z)
        x_z_max = x[np.argmax(z)]
        x_z_min = x[np.argmin(z)]

        # Determine if we are dealing with SN or DN
        single_null = abs_rel_difference(abs(z_min), z_max) > 0.05  # noqa: PLR2004

        if single_null:
            # Determine if it is an upper or lower SN
            lower = abs(z_min) > z_max

            if lower:
                constraints = [FieldNullConstraint(x_z_min, z_min)]
            else:
                constraints = [FieldNullConstraint(x_z_max, z_max)]

        else:
            constraints = [
                FieldNullConstraint(x_z_min, z_min),
                FieldNullConstraint(x_z_max, z_max),
            ]

        # Interpolate some points on the LCFS
        x_boundary, _, z_boundary = interpolate_points(x, np.zeros_like[x], z, n_points)

        # Apply an appropriate constraint on the LCFS
        if psi_boundary is None:
            arg_inner = np.argmin(x_boundary**2 + z_boundary**2)
            ref_x = x_boundary[arg_inner]
            ref_z = z_boundary[arg_inner]

            constraints.append(IsofluxConstraint(x_boundary, z_boundary, ref_x, ref_z))

        else:
            constraints.append(
                PsiBoundaryConstraint(x_boundary, z_boundary, psi_boundary)
            )
        super().__init__(constraints)
