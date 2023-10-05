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
Interface for designer classes.
"""

import abc
from typing import Callable, Dict, Generic, Optional, Type, TypeVar, Union

from bluemira.base.parameter_frame import ParameterFrame, make_parameter_frame
from bluemira.base.reactor_config import ConfigParams
from bluemira.base.tools import _timing

_DesignerReturnT = TypeVar("_DesignerReturnT")


class Designer(abc.ABC, Generic[_DesignerReturnT]):
    """
    Base class for 'Designers' that solver design problems as part of
    building a reactor component.

    Parameters
    ----------
    params:
        The parameters required by the designer.
    build_config:
        The build configuration options for the designer.
    verbose:
        control how much logging the designer will output

    Notes
    -----
    If there are no parameters associated with a concrete builder, set
    `param_cls` to `None` and pass `None` into this class's constructor.
    """

    KEY_RUN_MODE = "run_mode"

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, ConfigParams, None],
        build_config: Optional[Dict] = None,
        *,
        verbose=True,
    ):
        self.params = make_parameter_frame(params, self.param_cls)
        self.build_config = build_config if build_config is not None else {}
        self._verbose = verbose

    def execute(self) -> _DesignerReturnT:
        """
        Execute the designer with the run mode specified by the build config.

        By default the run mode is 'run'.
        """
        return _timing(
            self._get_run_func(self.run_mode),
            "Executed in",
            f"Executing {type(self).__name__}",
            debug_info_str=not self._verbose,
        )()

    @abc.abstractmethod
    def run(self) -> _DesignerReturnT:
        """Run the design problem."""

    def mock(self) -> _DesignerReturnT:
        """
        Return a mock of a design.

        Optionally implemented.
        """
        raise NotImplementedError

    def read(self) -> _DesignerReturnT:
        """
        Read a design from a file.

        The file path should be specified in the build config.

        Optionally implemented.
        """
        raise NotImplementedError

    @abc.abstractproperty
    def param_cls(self) -> Type[ParameterFrame]:
        """The ParameterFrame class defining this designer's parameters."""

    @property
    def run_mode(self) -> str:
        """Get the run mode of this designer."""
        return self.build_config.get(self.KEY_RUN_MODE, "run")

    def _get_run_func(self, mode: str) -> Callable:
        """Retrieve the function corresponding to the given run mode."""
        try:
            return getattr(self, mode)
        except AttributeError:
            raise ValueError(
                f"{type(self).__name__} has no run mode '{mode}'."
            ) from None


def run_designer(
    designer_cls: Type[Designer[_DesignerReturnT]],
    params: Union[ParameterFrame, Dict],
    build_config: Dict,
    **kwargs,
) -> _DesignerReturnT:
    """Make and run a designer, returning the result."""
    designer = designer_cls(params, build_config, **kwargs)
    return designer.execute()
