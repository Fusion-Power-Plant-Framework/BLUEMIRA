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
"""Designer for wall panelling."""

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Tuple, Union

import numpy as np

from bluemira.base.designer import Designer
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.geometry.wire import BluemiraWire
from bluemira.utilities.error import ExternalOptError
from eudemo.blanket.panelling._opt_problem import PanellingOptProblem
from eudemo.blanket.panelling._paneller import Paneller


@dataclass
class PanellingDesignerParams(ParameterFrame):
    """Parameters for :class:`.PanellingDesigner`."""

    fw_a_max: Parameter[float]
    """The maximum angle of rotation between adjacent panels [degrees]."""
    fw_dL_min: Parameter[float]  # noqa: N815
    """The minimum length for an individual panel [m]."""


class PanellingDesigner(Designer[np.ndarray]):
    r"""
    Design the shape for panels of the first wall.

    The panel design's objective is to minimise the cumulative panel
    length (minimising the amount of material required), whilst
    constraining the maximum angle of rotation between adjacent panels,
    and the minimum length of a panel.

    A best first guess will be made for the number of panels and their
    positions and then an optimiser will be run to minimise the
    cumulative length of the panels, whilst satisfying the minimum length
    and maximum angle constraints.

    Sometimes the initial guess will not enable the optimiser to find a
    solution that satisfies the constraints. In these cases an extra
    panel is added and the optimiser run again, until the constraints
    are satisfied, or the number of panels added exceeds the
    ``n_panel_increment_attempts`` build config parameter.

    Parameters
    ----------
    params:
        The parameters for the panelling design problem. See
        :class:`.PanellingDesignerParams` for the required parameters.
    wall_boundary:
        The boundary of the first wall to build the panels around. Note
        that this designer constructs panels around the *outside* of the
        given wall boundary; i.e., the panels enclose the wall.
    build_config:
        Configuration options for the designer:

        * algorithm: str
            The optimisation algorithm to use (default: ``'SLSQP'``\).
        * opt_conditions: Dict[str, Union[float, int]]
            The stopping conditions for the optimiser
            (default: ``{"max_eval": 400, "ftol_rel": 1e-4}``\).
        * n_panel_increment_attempts: int
            The number of times to try incrementing the number of panels
            in order to satisfy the given constraints (default: 3).
        * boundary_discretisation: int
            The number of points to discretise the input boundary wire
            into. A higher number here increases the fidelity in the
            interpolation used in the :class:`.Paneller`, and can prevent
            overlaps in the panels and boundary.
    """

    param_cls = PanellingDesignerParams
    params: PanellingDesignerParams
    _defaults: ClassVar = {
        "algorithm": "SLSQP",
        "opt_conditions": {"max_eval": 500, "ftol_rel": 1e-8},
        "n_panel_increment_attempts": 3,
        "boundary_discretisation": 200,
    }

    def __init__(
        self,
        params: Union[Dict, PanellingDesignerParams, ParameterFrame],
        wall_boundary: BluemiraWire,
        build_config: Optional[Dict] = None,
    ):
        super().__init__(params, build_config)
        self.wall_boundary = wall_boundary
        self._n_boundary_discr = int(
            self._get_config_or_default("boundary_discretisation")
        )
        self.opt_algorithm = str(self._get_config_or_default("algorithm"))
        self.opt_conditions = dict(self._get_config_or_default("opt_conditions"))

    def run(self) -> np.ndarray:
        """
        Run the design problem, performing the optimisation.

        Returns
        -------
        The coordinates of the panel end points (or joints). Has
        shape (2, N).
        """
        boundary = self.wall_boundary.discretize(
            ndiscr=self._n_boundary_discr, byedges=True
        ).xz
        opt_problem = self._set_up_opt_problem(boundary)
        initial_solution = opt_problem.paneller.joints(opt_problem.paneller.x0)
        max_retries = int(self._get_config_or_default("n_panel_increment_attempts"))
        x_opt, opt_problem, num_retries = self._run_opt_problem_with_retries(
            boundary, opt_problem, max_retries
        )

        # Make sure we warn about broken tolerances this time.
        if num_retries == max_retries and (
            x_opt is None or not opt_problem.check_constraints(x_opt, warn=True)
        ):
            bluemira_warn(
                "Could not solve panelling optimisation problem: no feasible "
                "solution found. Try reducing the minimum length and/or increasing "
                "the maximum allowed angle."
            )
        if x_opt is None or opt_problem.constraints_violated_by(x_opt, 1):
            return initial_solution

        return opt_problem.paneller.joints(x_opt)

    def mock(self) -> np.ndarray:
        """
        Mock the design problem, returning the initial guess for panel placement.

        This guarantees that panels will always fully contain the given
        boundary, but does not guarantee the maximum angle and minimum
        length constraints are honoured.
        """
        boundary = self.wall_boundary.discretize(
            ndiscr=self._n_boundary_discr, byedges=True
        ).xz
        paneller = Paneller(
            boundary, self.params.fw_a_max.value, self.params.fw_dL_min.value
        )
        return paneller.joints(paneller.x0)

    def _run_opt_problem_with_retries(
        self,
        boundary: np.ndarray,
        opt_problem: PanellingOptProblem,
        max_retries: int,
    ) -> Tuple[Union[np.ndarray, None], PanellingOptProblem, int]:
        """
        Run the minimise panel length optimisation problem.

        If the optimisation fails, retry with an extra panel
        ``max_retries`` times.

        Return ``None`` as the first return value if the final
        optimisation crashes with a 'more than iter SQP iterations'
        error (which it often does given an infeasible problem).
        """
        try:
            x_opt = opt_problem.optimise(
                opt_problem.paneller.x0,
                check_constraints=False,
                algorithm=self.opt_algorithm,
                opt_conditions=self.opt_conditions,
            ).x
        except ExternalOptError:
            # Avoid 'more than iter SQP iterations' errors stopping the
            # design.
            # Ignoring the error here is OK, as the optimiser prints a
            # warning and we either try again with more panels, or
            # return our initial guess as a fall back.
            x_opt = None
        iter_num = 0
        while (
            x_opt is None or not opt_problem.check_constraints(x_opt, warn=False)
        ) and iter_num < max_retries:
            # We couldn't satisfy the constraints on our last attempt,
            # so try increasing the number of panels.
            # Note we're actually increasing the number of panels by 1
            # by adding 3 below, as there are two more panels than
            # optimisation parameters.
            n_panels = opt_problem.n_opts + 3
            opt_problem = self._set_up_opt_problem(boundary, n_panels)
            try:
                x_opt = opt_problem.optimise(
                    opt_problem.paneller.x0,
                    check_constraints=False,
                    algorithm=self.opt_algorithm,
                    opt_conditions=self.opt_conditions,
                ).x
            except ExternalOptError:
                x_opt = None
            iter_num += 1
        return x_opt, opt_problem, iter_num

    def _set_up_opt_problem(
        self, boundary: np.ndarray, fix_num_panels: Optional[int] = None
    ) -> PanellingOptProblem:
        """Set up an instance of the minimise panel length optimisation problem."""
        paneller = Paneller(
            boundary,
            self.params.fw_a_max.value,
            self.params.fw_dL_min.value,
            fix_num_panels=fix_num_panels,
        )
        return PanellingOptProblem(paneller)

    def _get_config_or_default(self, config_key: str) -> Any:
        return self.build_config.get(config_key, self._defaults[config_key])
