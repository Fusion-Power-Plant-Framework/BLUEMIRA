# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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
Constrained and unconstrained optimisation tools for coilset design.
"""

from typing import List

import numpy as np

import bluemira.equilibria.opt_objectives as objectives
from bluemira.base.look_and_feel import bluemira_print_flush
from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.positioner import RegionMapper
from bluemira.utilities.opt_problems import (
    OptimisationConstraint,
    OptimisationObjective,
    OptimisationProblem,
)
from bluemira.utilities.opt_tools import regularised_lsq_fom, tikhonov
from bluemira.utilities.optimiser import Optimiser

__all__ = [
    "UnconstrainedCurrentCOP",
    "BoundedCurrentCOP",
    "CoilsetPositionCOP",
    "NestedCoilsetPositionCOP",
]


class CoilsetOP(OptimisationProblem):
    def __init__(
        self,
        coilset: CoilSet,
        optimiser: Optimiser = None,
        objective: OptimisationObjective = None,
        constraints: List[OptimisationConstraint] = [],
    ):
        super().__init__(coilset, optimiser, objective, constraints)
        self.scale = 1e6
        self.initial_state, self.substates = self.read_coilset_state(self.coilset)
        self.x0, self.z0, self.I0 = np.array_split(self.initial_state, self.substates)

    @property
    def coilset(self):
        return self._parameterisation

    @coilset.setter
    def coilset(self, value: CoilSet):
        self._parameterisation = value

    def read_coilset_state(self, coilset):
        """
        Reads the input coilset and generates the state vector as an array to represent
        it.

        Parameters
        ----------
        coilset: Coilset
            Coilset to be read into the state vector.

        Returns
        -------
        coilset_state: np.array
            State vector containing substate (position and current)
            information for each coil.
        substates: int
            Number of substates (blocks) in the state vector.
        """
        substates = 3
        x, z = coilset.get_positions()
        currents = coilset.get_control_currents() / self.scale

        coilset_state = np.concatenate((x, z, currents))
        return coilset_state, substates

    def set_coilset_state(self, coilset_state):
        """
        Set the optimiser coilset from a provided state vector.

        Parameters
        ----------
        coilset_state: np.array
            State vector representing degrees of freedom of the coilset,
            to be used to update the coilset.
        """
        x, z, currents = np.array_split(coilset_state, 3)

        # coilset.set_positions not currently working for
        # SymmetricCircuits, it appears...
        # positions = list(zip(x, z))
        # self.coilset.set_positions(positions)
        for i, coil in enumerate(self.coilset.coils.values()):
            coil.x = x[i]
            coil.z = z[i]
        self.coilset.set_control_currents(currents * self.scale)

    def get_state_bounds(self, x_bounds, z_bounds, current_bounds):
        """
        Set bounds on the state vector from provided bounds on the substates.

        Parameters
        ----------
        opt: nlopt.opt
            Optimiser on which to set the bounds.
        x_bounds: tuple
            Tuple containing lower and upper bounds on the radial coil positions.
        z_bounds: tuple
            Tuple containing lower and upper bounds on the vertical coil positions.
        current_bounds: tuple
            Tuple containing bounds on the coil currents.

        Returns
        -------
        opt: nlopt.opt
            Optimiser updated in-place with bounds set.
        """
        lower_bounds = np.concatenate((x_bounds[0], z_bounds[0], current_bounds[0]))
        upper_bounds = np.concatenate((x_bounds[1], z_bounds[1], current_bounds[1]))
        bounds = np.array([lower_bounds, upper_bounds])
        return bounds

    def get_current_bounds(self, max_currents):
        """
        Gets the current vector bounds. Must be called prior to optimise.

        Parameters
        ----------
        max_currents: float or np.ndarray
            Maximum magnitude of currents in each coil [A] permitted during optimisation.
            If max_current is supplied as a float, the float will be set as the
            maximum allowed current magnitude for all coils.
            If the coils have current density limits that are more restrictive than these
            coil currents, the smaller current limit of the two will be used for each
            coil.

        Returns
        -------
        current_bounds: (np.narray, np.narray)
            Tuple of arrays containing lower and upper bounds for currents
            permitted in each control coil.
        """
        n_control_currents = len(self.coilset.get_control_currents())
        scaled_input_current_limits = np.inf * np.ones(n_control_currents)

        if max_currents is not None:
            input_current_limits = np.asarray(max_currents)
            input_size = np.size(np.asarray(input_current_limits))
            if input_size == 1 or input_size == n_control_currents:
                scaled_input_current_limits = input_current_limits / self.scale
            else:
                raise EquilibriaError(
                    "Length of max_currents array provided to optimiser is not"
                    "equal to the number of control currents present."
                )

        # Get the current limits from coil current densities
        coilset_current_limits = self.coilset.get_max_currents(0.0)
        if len(coilset_current_limits) != n_control_currents:
            raise EquilibriaError(
                "Length of array containing coilset current limits"
                "is not equal to the number of control currents in optimiser."
            )

        # Limit the control current magnitude by the smaller of the two limits
        control_current_limits = np.minimum(
            scaled_input_current_limits, coilset_current_limits
        )
        current_bounds = (-control_current_limits, control_current_limits)

        return current_bounds

    def __call__(self, eq=None, targets=None, psi_bndry=None):
        """
        Parameters
        ----------
        Dummy input parameters for consistency with deprecated interface
        in Iterators.
        """
        return self.optimise()


class UnconstrainedCurrentCOP(CoilsetOP):
    """
    Unconstrained norm-2 optimisation with Tikhonov regularisation

    Intended to replace Norm2Tikhonov as a Coilset optimiser.
    """

    def __init__(self, coilset, eq, targets, gamma=1e-12):
        super().__init__(coilset)

        self.eq = eq
        self.targets = targets
        self.gamma = gamma

    def optimise(self):
        """
        Optimise the prescribed problem.

        Notes
        -----
        The weight vector is used to scale the response matrix and
        constraint vector. The weights are assumed to be uncorrelated, such that the
        weight matrix W_ij used to define (for example) the least-squares objective
        function (Ax - b)ᵀ W (Ax - b), is diagonal, such that
        weights[i] = w[i] = sqrt(W[i,i]).
        """
        # Scale the control matrix and constraint vector by weights.
        self.targets(self.eq, I_not_dI=False)
        _, a_mat, b_vec = self.targets.get_weighted_arrays()

        # Optimise currents using analytic expression for optimum.
        current_adjustment = tikhonov(a_mat, b_vec, self.gamma)

        self.coilset.adjust_currents(current_adjustment)
        return self.coilset


class BoundedCurrentCOP(CoilsetOP):
    """
    NLOpt based optimiser for coil currents subject to maximum current bounds.

    Parameters
    ----------
    coilset: CoilSet
        Coilset used to get coil current limits and number of coils.
    max_currents float or np.array(len(coilset._ccoils)) (default = None)
        Maximum allowed current for each independent coil current in coilset [A].
        If specified as a float, the float will set the maximum allowed current
        for all coils.
    gamma: float (default = 1e-8)
        Tikhonov regularisation parameter in units of [A⁻¹].
    opt_args: dict
        Dictionary containing arguments to pass to NLOpt optimiser.
        Defaults to using LD_SLSQP.
    opt_constraints: iterable (default = [])
        Iterable of OptimisationConstraint objects containing optimisation
        constraints held during optimisation.
    """

    def __init__(
        self,
        coilset,
        eq,
        targets,
        gamma=1e-8,
        max_currents=None,
        optimiser=Optimiser(
            algorithm_name="SLSQP",
            opt_conditions={
                "xtol_rel": 1e-4,
                "xtol_abs": 1e-4,
                "ftol_rel": 1e-4,
                "ftol_abs": 1e-4,
                "max_eval": 100,
            },
            opt_parameters={"initial_step": 0.03},
        ),
        opt_constraints=[],
    ):
        # noqa :N803

        # Set objective function for this OptimisationProblem,
        # and initialise
        objective = OptimisationObjective(
            objectives.regularised_lsq_objective, {"gamma": gamma}
        )
        super().__init__(coilset, optimiser, objective, opt_constraints)

        # Set up optimiser
        bounds = self.get_current_bounds(max_currents)
        dimension = len(bounds[0])
        self.set_up_optimiser(dimension, bounds)

        # Save additional parameters used to generate remaining
        # objective/constraint arguments at runtime
        self.eq = eq
        self.targets = targets

    def optimise(self):
        """
        Optimiser handle. Used in __call__

        Returns np.ndarray of optimised currents
        in each coil [A].
        """
        # Get initial currents.
        initial_currents = self.coilset.get_control_currents() / self.scale
        initial_currents = np.clip(
            initial_currents, self.opt.lower_bounds, self.opt.upper_bounds
        )

        # Set up data needed in FoM evaluation.
        # Scale the control matrix and constraint vector by weights.
        self.targets(self.eq, I_not_dI=True)
        _, a_mat, b_vec = self.targets.get_weighted_arrays()

        self._objective._args["scale"] = self.scale
        self._objective._args["a_mat"] = a_mat
        self._objective._args["b_vec"] = b_vec

        # Optimise
        currents = self.opt.optimise(initial_currents)

        coilset_state = np.concatenate((self.x0, self.z0, currents))
        self.set_coilset_state(coilset_state)
        return self.coilset


class CoilsetPositionCOP(CoilsetOP):
    """
    NLOpt based optimiser for coilsets (currents and positions)
    subject to maximum current and position bounds.
    Coil currents and positions are optimised simultaneously.

    Parameters
    ----------
    coilset: CoilSet
        Coilset used to get coil current limits and number of coils.
    pfregions: dict(coil_name:Loop, coil_name:Loop, ...)
        Dictionary of loops that specify convex hull regions inside which
        each PF control coil position is to be optimised.
        The loop objects must be 2d in x,z in units of [m].
    max_currents: float or np.array(len(coilset._ccoils)) (default = None)
        Maximum allowed current for each independent coil current in coilset [A].
        If specified as a float, the float will set the maximum allowed current
        for all coils.
    gamma: float (default = 1e-8)
        Tikhonov regularisation parameter in units of [A⁻¹].
    opt_args: dict
        Dictionary containing arguments to pass to NLOpt optimiser.
        Defaults to using LN_SBPLX, terminating when the figure of
        merit < stop_val = 1.0, or max_eval =100.
    opt_constraints: iterable (default = [])
        Iterable of OptimisationConstraint objects containing optimisation
        constraints held during optimisation.

    Notes
    -----
    Setting stopval and maxeval is the most reliable way to stop optimisation
    at the desired figure of merit and number of iterations respectively.
    Some NLOpt optimisers display unexpected behaviour when setting xtol and
    ftol, and may not terminate as expected when those criteria are reached.
    """

    def __init__(
        self,
        coilset,
        pfregions,
        optimiser=Optimiser(
            algorithm_name="SBPLX",
            opt_conditions={
                "stop_val": 1.0,
                "max_eval": 100,
            },
            opt_parameters={},
        ),
        max_currents=None,
        gamma=1e-8,
        opt_constraints=[],
    ):
        # noqa :N803
        objective = OptimisationObjective(self.f_min_objective)
        super().__init__(coilset, optimiser, objective, opt_constraints)

        # Create region map
        self.region_mapper = RegionMapper(pfregions)

        # Store inputs
        self.max_currents = max_currents
        self.gamma = gamma

        # Set up optimiser
        bounds = self.get_mapped_state_bounds(self.region_mapper, self.max_currents)
        dimension = len(bounds[0])
        self.set_up_optimiser(dimension, bounds)

    def get_mapped_state_bounds(self, region_mapper, max_currents):
        """
        Get coilset state bounds after position mapping.
        """
        # Get mapped position bounds from RegionMapper
        _, lower_lmap_bounds, upper_lmap_bounds = region_mapper.get_Lmap(self.coilset)
        current_bounds = self.get_current_bounds(max_currents)

        lower_bounds = np.concatenate((lower_lmap_bounds, current_bounds[0]))
        upper_bounds = np.concatenate((upper_lmap_bounds, current_bounds[1]))
        bounds = (lower_bounds, upper_bounds)
        return bounds

    def optimise(self):
        """
        Optimiser handle. Used in __call__

        Returns np.ndarray of optimised currents in each coil [A].
        """
        # Get initial state and apply region mapping to coil positions.
        initial_state, _ = self.read_coilset_state(self.coilset)
        _, _, initial_currents = np.array_split(initial_state, self.substates)
        initial_mapped_positions, _, _ = self.region_mapper.get_Lmap(self.coilset)
        initial_mapped_state = np.concatenate(
            (initial_mapped_positions, initial_currents)
        )
        self.iter = 0

        # Optimise
        state = self.opt.optimise(initial_mapped_state)

        # Call objective function final time on optimised state
        # to set coilset.
        # Necessary as optimised state may not always be the final
        # one evaluated by optimiser.
        self.get_state_figure_of_merit(state)
        return self.coilset

    def f_min_objective(self, vector, grad):
        """
        Objective function for nlopt optimisation (minimisation),
        consisting of a least-squares objective with Tikhonov
        regularisation term, which updates the gradient in-place.

        Parameters
        ----------
        vector: np.array(n_C)
            State vector of the array of coil currents.
        grad: np.array
            Local gradient of objective function used by LD NLOPT algorithms.
            Updated in-place.

        Returns
        -------
        fom: Value of objective function (figure of merit).
        """
        self.iter += 1
        fom = self.get_state_figure_of_merit(vector)
        if grad.size > 0:
            grad[:] = self.opt.approx_derivative(
                self.get_state_figure_of_merit,
                vector,
                f0=fom,
            )
        bluemira_print_flush(
            f"EQUILIBRIA Coilset iter {self.iter}: figure of merit = {fom:.2e}"
        )
        return fom

    def get_state_figure_of_merit(self, vector):
        """
        Calculates figure of merit from objective function,
        consisting of a least-squares objective with Tikhonov
        regularisation term, which updates the gradient in-place.

        Parameters
        ----------
        vector: np.array(n_C)
            State vector of the array of coil currents.

        Returns
        -------
        fom: Value of objective function (figure of merit).
        """
        mapped_x, mapped_z, currents = np.array_split(vector, self.substates)
        mapped_positions = np.concatenate((mapped_x, mapped_z))
        self.region_mapper.set_Lmap(mapped_positions)
        x_vals, z_vals = self.region_mapper.get_xz_arrays()
        coilset_state = np.concatenate((x_vals, z_vals, currents))

        self.set_coilset_state(coilset_state)

        # Update target
        self.eq._remap_greens()

        self.targets(self.eq, I_not_dI=True, fixed_coils=False)
        self.A = self.targets.A
        self.b = self.targets.b
        self.w = self.targets.w
        self.A = self.w[:, np.newaxis] * self.A
        self.b *= self.w

        # Calculate objective function
        fom, err = regularised_lsq_fom(currents * self.scale, self.A, self.b, self.gamma)
        return fom


class NestedCoilsetPositionCOP(CoilsetOP):
    """
    NLOpt based optimiser for coilsets (currents and positions)
    subject to maximum current and position bounds. Performs a
    nested optimisation for coil currents within each position
    optimisation function call.

    Parameters
    ----------
    sub_opt: EquilibriumOptimiser
        Optimiser to use for the optimisation of coil currents at each trial
        set of coil positions. sub_opt.coilset must exist, and will be
        modified during the optimisation.
    pfregions: dict(coil_name:Loop, coil_name:Loop, ...)
        Dictionary of loops that specify convex hull regions inside which
        each PF control coil position is to be optimised.
        The loop objects must be 2d in x,z in units of [m].
    opt_args: dict
        Dictionary containing arguments to pass to NLOpt optimiser
        used in position optimisation.
        Defaults to using LN_SBPLX, terminating when the figure of
        merit < stop_val = 1.0, or max_eval = 100.
    opt_constraints: iterable (default = [])
        Iterable of OptimisationConstraint objects containing optimisation
        constraints held during optimisation.

    Notes
    -----
        Setting stopval and maxeval is the most reliable way to stop optimisation
        at the desired figure of merit and number of iterations respectively.
        Some NLOpt optimisers display unexpected behaviour when setting xtol and
        ftol, and may not terminate as expected when those criteria are reached.
    """

    def __init__(
        self,
        sub_opt,
        pfregions,
        optimiser=Optimiser(
            algorithm_name="SBPLX",
            opt_conditions={
                "stop_val": 1.0,
                "max_eval": 100,
            },
            opt_parameters={},
        ),
        opt_constraints=[],
    ):
        # noqa :N803
        opt_objective = OptimisationObjective(self.f_min_objective)
        super().__init__(sub_opt.coilset, optimiser, opt_objective)

        self.region_mapper = RegionMapper(pfregions)

        # Set up optimiser
        _, lower_bounds, upper_bounds = self.region_mapper.get_Lmap(self.coilset)
        bounds = (lower_bounds, upper_bounds)
        dimension = len(bounds[0])
        self.set_up_optimiser(dimension, bounds)
        self.sub_opt = sub_opt

    def optimise(self):
        """
        Optimiser handle. Used in __call__

        Returns optimised coilset object.
        """
        # Get initial currents, and trim to within current bounds.
        initial_state, substates = self.read_coilset_state(self.coilset)
        _, _, self.currents = np.array_split(initial_state, substates)
        intial_mapped_positions, _, _ = self.region_mapper.get_Lmap(self.coilset)

        # Optimise
        self.iter = 0
        positions = self.opt.optimise(intial_mapped_positions)

        # Call objective function final time on optimised state
        # to set coilset.
        # Necessary as optimised state may not always be the final
        # one evaluated by optimiser.
        self.get_state_figure_of_merit(positions)
        return self.coilset

    def f_min_objective(self, vector, grad):
        """
        Objective function for nlopt optimisation (minimisation),
        fetched from the current optimiser provided at each
        trial set of coil positions.

        Parameters
        ----------
        vector: np.array(n_C)
            State vector of the array of coil currents.
        grad: np.array
            Local gradient of objective function used by LD NLOPT algorithms.
            Updated in-place.

        Returns
        -------
        fom: Value of objective function (figure of merit).
        """
        self.iter += 1
        fom = self.get_state_figure_of_merit(vector)
        if grad.size > 0:
            grad[:] = self.opt.approx_derivative(
                self.get_state_figure_of_merit,
                vector,
                f0=fom,
            )
        bluemira_print_flush(
            f"EQUILIBRIA Coilset iter {self.iter}: figure of merit = {fom:.2e}"
        )
        return fom

    def get_state_figure_of_merit(self, vector):
        """
        Calculates figure of merit, returned from the current
        optimiser at each trial coil position.

        Parameters
        ----------
        vector: np.array(n_C)
            State vector of the array of coil currents.

        Returns
        -------
        fom: Value of objective function (figure of merit).
        """
        self.region_mapper.set_Lmap(vector)
        x_vals, z_vals = self.region_mapper.get_xz_arrays()
        positions = np.concatenate((x_vals, z_vals))
        coilset_state = np.concatenate((positions, self.currents))
        self.set_coilset_state(coilset_state)

        # Update target
        self.eq._remap_greens()
        self.targets(self.eq, I_not_dI=True, fixed_coils=False)

        # Calculate objective function
        self.sub_opt(self.eq, self.targets)
        fom = self.sub_opt.opt.optimum_value
        return fom
