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
from dataclasses import asdict, dataclass
from typing import Dict, Optional

from bluemira.base.constants import EPS
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.optimisation.error import OptimisationConditionsError


@dataclass
class NLOptConditions:
    """Hold and validate optimiser stopping conditions."""

    ftol_abs: Optional[float] = None
    ftol_rel: Optional[float] = None
    xtol_abs: Optional[float] = None
    xtol_rel: Optional[float] = None
    max_eval: Optional[int] = None
    max_time: Optional[float] = None
    stop_val: Optional[float] = None

    def __post_init__(self):
        """Validate initialised values."""
        self._validate()

    def to_dict(self) -> Dict[str, float]:
        """Return the data in dictionary form."""
        return asdict(self)

    def _validate(self) -> None:
        for condition in [
            self.ftol_abs,
            self.ftol_rel,
            self.xtol_abs,
            self.xtol_rel,
        ]:
            if condition and condition < EPS:
                bluemira_warn(
                    "optimisation: Setting stopping condition to less than machine "
                    "precision. This condition may never be met."
                )
        if self._no_stopping_condition_set():
            raise OptimisationConditionsError(
                "Must specify at least one stopping condition for the optimiser."
            )
        if self.max_eval is not None and isinstance(self.max_eval, float):
            bluemira_warn("optimisation: max_eval must be an integer, forcing type.")
            self.max_eval = int(self.max_eval)

    def _no_stopping_condition_set(self) -> bool:
        return all(
            condition is None
            for condition in [
                self.ftol_abs,
                self.ftol_rel,
                self.xtol_abs,
                self.xtol_rel,
                self.max_eval,
                self.max_time,
                self.stop_val,
            ]
        )
