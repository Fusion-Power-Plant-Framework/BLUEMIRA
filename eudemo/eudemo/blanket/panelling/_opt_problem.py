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
"""Definition of panelling optimisation problem for EUDEMO."""

from typing import List, Tuple

import numpy as np

from bluemira.optimisation import OptimisationProblem
from bluemira.optimisation.typing import ConstraintT
from eudemo.blanket.panelling._paneller import Paneller


class PanellingOptProblem(OptimisationProblem):
    """
    Optimisation problem to minimise the cumulative length of first wall panels.

    The optimisation parameters are the normalised lengths along the
    panelling boundary at which the panels and boundary touch.

    The objective is to minimise the cumulative length of the panels,
    subject to a maximum tail-to-tail angle between panels and a minimum
    panel length. Note that the parameters for these constraints are
    properties of the ``paneller``.

    Parameters
    ----------
    paneller:
        The :class:`.Paneller` to optimise the parameters of. Note that
        these parameters are the normalised length along the paneller's
        boundary where the panels and boundary meet.
    """

    def __init__(self, paneller: Paneller):
        self.paneller = paneller

    def objective(self, x: np.ndarray) -> float:
        """Objective function to minimise the total panel length."""
        return self.paneller.length(x)

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """The normalized bounds of the parameterisation."""
        return np.zeros_like(self.paneller.x0), np.ones_like(self.paneller.x0)

    def ineq_constraints(self) -> List[ConstraintT]:
        """
        The inequality constraints for the optimiser.

        We are constraining the tail-to-tail angle between panels and
        the minimum panel length.
        """
        return [
            {
                "f_constraint": self.constrain_min_length_and_angles,
                "tolerance": np.full(self.n_constraints, 1e-8),
            }
        ]

    @property
    def n_opts(self) -> int:
        """
        The number of optimisation parameters.

        The optimisation parameters are how far along the boundary's
        length each panel tangents the boundary (normalized to between
        0 and 1). We exclude the panel's start and end points, which are
        fixed.
        """
        # exclude start and end points; hence 'N - 2'
        return self.paneller.n_panels - 2

    @property
    def n_constraints(self) -> int:
        """
        The number of optimisation constraints.

        We constrain:

            - the minimum length of each panel
            - the angle between each panel
              (no. of angles = no. of panels - 1)

        Note that we exclude the start and end touch points which are
        fixed.
        """
        return 2 * self.paneller.n_panels - 1

    def constrain_min_length_and_angles(self, x: np.ndarray) -> np.ndarray:
        """Constraint function function for the optimiser."""
        n_panels = self.paneller.n_panels
        constraint = np.empty(self.n_constraints, dtype=float)
        constraint[:n_panels] = self._constrain_min_length(x)
        constraint[n_panels:] = self._constrain_max_angle(x)
        return constraint

    def _constrain_min_length(self, x: np.ndarray) -> np.ndarray:
        lengths = self.paneller.panel_lengths(x)
        return self.paneller.dx_min - lengths

    def _constrain_max_angle(self, x: np.ndarray) -> np.ndarray:
        return self.paneller.angles(x) - self.paneller.max_angle

    def constraints_violated_by(self, x: np.ndarray, atol: float) -> bool:
        """Return True if any constraints are violated by more that ``atol``."""
        constraint = self.constrain_min_length_and_angles(x)
        return np.any(constraint > atol)
