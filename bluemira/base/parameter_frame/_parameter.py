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
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, Tuple, Type, TypeVar, TypedDict, Union

import numpy as np
import pint
from typeguard import config, typechecked

from bluemira.base.constants import raw_uc, units_compatible


def type_fail(exc, memo):  # noqa: ARG001
    """
    Raise TypeError on wrong type

    Notes
    -----
    typeguard by default raises a TypeCheckError
    may want to have a custom checker in future

    """
    raise TypeError(f"{exc._path[0]} {exc.args[0]}")


config.typecheck_fail_callback = type_fail

ParameterValueType = TypeVar("ParameterValueType")


class ParamDictT(TypedDict):
    """Typed dictionary for a Parameter."""

    name: str
    value: ParameterValueType
    unit: str
    source: str
    description: str
    long_name: str


@dataclass
class ParameterValue(Generic[ParameterValueType]):
    """Holds parameter value information."""

    value: ParameterValueType
    source: str


class Parameter(Generic[ParameterValueType]):
    """
    Represents a parameter with physical units.

    Parameters
    ----------
    name
        The name of the parameter.
    value
        The parameter's value.
    unit
        The parameter's unit.
    source
        The origin of the parameter's value.
    description
        A description of the parameter.
    long_name
        A longer name for the parameter.
    """

    @typechecked
    def __init__(
        self,
        name: str,
        value: ParameterValueType,
        unit: str = "",
        source: str = "",
        description: str = "",
        long_name: str = "",
        _value_types: Optional[Tuple[Type, ...]] = None,
    ):
        value = self._type_check(name, value, _value_types)
        self._name = name
        self._value = value
        self._unit = pint.Unit(unit)
        self._source = source
        self._description = description
        self._long_name = long_name

        self._history: List[ParameterValue[ParameterValueType]] = []
        self._add_history_record()

    @staticmethod
    def _type_check(
        name: str, value: ParameterValueType, value_types: Optional[Tuple[Type, ...]]
    ) -> ParameterValueType:
        if value_types and value is not None:
            if float in value_types and isinstance(value, int):
                value = float(value)
            elif (
                int in value_types
                and isinstance(value, float)
                and np.isclose(value, int(value), rtol=0)
            ):
                value = int(value)
            elif not isinstance(value, value_types):
                raise TypeError(
                    f'{name}: type of "value" must be one of {value_types}; '
                    f"got {type(value)} (value: {value}) instead"
                )
        return value

    def __repr__(self) -> str:
        """String repr of class instance."""
        return f"<{type(self).__name__}({self.name}={self.value} {self.unit})>"

    def __eq__(self, __o: object) -> bool:
        """
        Check if this parameter is equal to another.

        Parameters are equal if their names and values (with matching
        units) are equal.
        """
        if not isinstance(__o, Parameter):
            return NotImplemented
        try:
            o_value_with_correct_unit = raw_uc(__o.value, __o.unit, self.unit)
        except pint.DimensionalityError:
            # incompatible units
            return False
        return (self.name == __o.name) and (self.value == o_value_with_correct_unit)

    def __hash__(self):
        return hash((self._name, self._description, self._long_name))

    def history(self) -> List[ParameterValue[ParameterValueType]]:
        """Return the history of this parameter's value."""
        return copy.deepcopy(self._history)

    @typechecked
    def set_value(self, new_value: ParameterValueType, source: str = ""):
        """Set the parameter's value and update the source."""
        self._value = new_value
        self._source = source
        self._add_history_record()

    def to_dict(self) -> Dict:
        """Serialize the parameter to a dictionary."""
        out = {
            "name": self.name,
            "value": self.value,
            "unit": "dimensionless" if not self.unit else self.unit,
        }
        for field in ["source", "description", "long_name"]:
            if value := getattr(self, field):
                out[field] = value
        return out

    @property
    def name(self) -> str:
        """Return the name of the parameter."""
        return self._name

    @property
    def value(self) -> ParameterValueType:
        """Return the current value of the parameter."""
        return self._value

    @value.setter
    def value(self, new_value: ParameterValueType):
        self.set_value(new_value, source="")

    def value_as(self, unit: Union[str, pint.Unit]) -> Union[ParameterValueType, None]:
        """
        Return the current value in a given unit

        Notes
        -----
        If the current value of the parameter is None the function checks for
        a valid unit conversion
        """
        try:
            return raw_uc(self.value, self.unit, unit)
        except pint.errors.PintError as pe:
            raise ValueError("Unit conversion failed") from pe
        except TypeError as te:
            if self.value is None:
                if units_compatible(self.unit, unit):
                    return None
                raise ValueError("Unit conversion failed") from te
            raise

    @property
    def unit(self) -> str:
        """Return the physical unit of the parameter."""
        return f"{self._unit:~P}"

    @property
    def source(self) -> str:
        """Return the source that last set the value of this parameter."""
        return self._source

    @property
    def long_name(self) -> str:
        """Return a long name for this parameter."""
        return self._long_name

    @property
    def description(self) -> str:
        """Return a description for the parameter."""
        return self._description

    def _add_history_record(self):
        history_entry = ParameterValue(self.value, self.source)
        self._history.append(history_entry)
