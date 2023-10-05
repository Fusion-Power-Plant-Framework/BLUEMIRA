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
Optimisation variable class.
"""
from __future__ import annotations

import json
from dataclasses import MISSING, Field, field
from pathlib import Path
from typing import Dict, Generator, Optional, TextIO, TypedDict, Union

import numpy as np
from tabulate import tabulate

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.utilities.error import OptVariablesError
from bluemira.utilities.tools import json_writer


class OptVarVarDictValueT(TypedDict, total=False):
    """Typed dictionary for a the values of an OptVariable from a var_dict."""

    value: float
    lower_bound: float
    upper_bound: float
    fixed: bool


VarDictT = Dict[str, OptVarVarDictValueT]


class OptVarDictT(TypedDict):
    """Typed dictionary representation of an OptVariable."""

    name: str
    value: float
    lower_bound: float
    upper_bound: float
    fixed: bool
    description: str


class OptVarSerializedT(TypedDict):
    """Typed dictionary for a serialised OptVariable."""

    value: float
    lower_bound: float
    upper_bound: float
    fixed: bool
    description: str


class OptVariable:
    """
    A bounded variable, uniformly normalised from 0 to 1 w.r.t. its bounds.

    Parameters
    ----------
    name:
        Name of the variable
    value: float
        Value of the variable
    lower_bound:
        Lower bound of the variable
    upper_bound:
        Upper bound of the variable
    fixed:
        Whether or not the variable is to be held constant
    description:
        Description of the variable
    """

    __slots__ = ("name", "_value", "lower_bound", "upper_bound", "fixed", "description")

    def __init__(
        self,
        name: str,
        value: float,
        lower_bound: float,
        upper_bound: float,
        fixed: bool = False,
        description: Optional[str] = None,
    ):
        self.name = name

        self._value = value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.fixed = fixed
        self.description = description

        self._validate_bounds()
        self._validate_value(value)

    @property
    def value(self) -> float:
        """
        The value of the variable.
        """
        return self._value

    @value.setter
    def value(self, value):
        """
        Set the value of the variable, enforcing bounds.
        """
        if self.fixed:
            raise OptVariablesError("Cannot set the value of a fixed variable.")

        self._validate_value(value)
        self._value = value

    @property
    def normalised_value(self) -> float:
        """
        The value uniformly normalised between 0 and 1 w.r.t. its bounds
        """
        return (self.value - self.lower_bound) / (self.upper_bound - self.lower_bound)

    def from_normalised(self, norm: float) -> float:
        """
        The value from a normalised value between [0 -> 1], w.r.t its bounds
        """
        return self.lower_bound + norm * (self.upper_bound - self.lower_bound)

    def fix(self, value: Optional[float] = None):
        """
        Fix the variable at a specified value. Ignores bounds.

        Parameters
        ----------
        value:
            Value at which to fix the variable.
        """
        self.fixed = True
        if value is not None:
            self._value = value

    def adjust(
        self,
        value: Optional[float] = None,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        strict_bounds: bool = True,
    ):
        """
        Adjust the OptVariable.

        Parameters
        ----------
        value:
            Value of the variable to set
        lower_bound:
            Value of the lower bound to set
        upper_bound:
            Value of the upper to set
        strict_bounds:
            If True, will raise errors if values are outside the bounds. If False, the
            bounds are dynamically adjusted to match the value.
        """
        if self.fixed:
            raise OptVariablesError(f"'{self.name}' is fixed and cannot be adjusted.")

        if lower_bound is not None:
            self.lower_bound = lower_bound

        if upper_bound is not None:
            self.upper_bound = upper_bound

        if value is not None:
            if not strict_bounds:
                self._adjust_bounds_to(value)
            self.value = value

        self._validate_bounds()

    def as_dict(self) -> OptVarDictT:
        """Dictionary representation of OptVariable, can be used for serialisation"""
        return {
            "name": self.name,
            "value": self.value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "fixed": self.fixed,
            "description": self.description or "",
        }

    def as_serializable(self) -> OptVarSerializedT:
        """Dictionary representation of OptVariable"""
        return {
            "value": self.value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "fixed": self.fixed,
            "description": self.description or "",
        }

    @classmethod
    def from_serialized(cls, name: str, data: OptVarSerializedT):
        """Create an OptVariable from a dictionary"""
        return cls(
            name=name,
            value=data["value"],
            lower_bound=data["lower_bound"],
            upper_bound=data["upper_bound"],
            fixed=data["fixed"],
            description=data["description"],
        )

    def _adjust_bounds_to(self, value):
        """
        Adjust the bounds to the value
        """
        if value < self.lower_bound:
            bluemira_warn(
                f"OptVariable '{self.name}': value was set to below its lower bound."
                " Adjusting bound."
            )
            self.lower_bound = value

        if value > self.upper_bound:
            bluemira_warn(
                f"OptVariable '{self.name}': value was set to above its upper bound."
                " Adjusting bound."
            )
            self.upper_bound = value

    def _validate_bounds(self):
        if self.lower_bound > self.upper_bound:
            raise OptVariablesError(
                f"OptVariable '{self.name}' - lower bound is higher than upper bound."
            )

    def _validate_value(self, value):
        if not self.lower_bound <= value <= self.upper_bound:
            raise OptVariablesError(
                f"OptVariable '{self.name}' - value {value} is out of bounds:"
                f" [{self.lower_bound}, {self.upper_bound}]"
            )

    def __repr__(self) -> str:
        """
        Representation of OptVariable
        """
        lower_bound, upper_bound, fixed = (
            self.lower_bound,
            self.upper_bound,
            self.fixed,
        )
        return (
            f"{self.__class__.__name__}({self.name}, {self.value}, {lower_bound=},"
            f" {upper_bound=}, {fixed=})"
        )

    def __str__(self) -> str:
        """
        Pretty representation of OptVariable
        """
        bound = (
            f" Bounds: ({self.lower_bound}, {self.upper_bound})"
            if not self.fixed
            else ""
        )
        descr = f' "{self.description}"' if self.description is not None else ""

        return f"{self.name} = {self.value}{bound}{descr}"

    def __add__(self, other: OptVariable):
        """The sum of two OptVariables is the sum of their values"""
        if isinstance(other, OptVariable):
            return self.value + other.value
        if isinstance(other, (int, float, np.floating)):
            return self.value + other
        raise TypeError(f"Cannot add OptVariable with {type(other)}")

    def __sub__(self, other: OptVariable):
        """The subtraction of two OptVariables is the subtraction of their values"""
        if isinstance(other, OptVariable):
            return self.value - other.value
        if isinstance(other, (int, float, np.floating)):
            return self.value - other
        raise TypeError(f"Cannot subtract OptVariable with {type(other)}")

    def __mul__(self, other: OptVariable):
        """
        The multiplication of two OptVariables is
        the multiplication of their values
        """
        if isinstance(other, OptVariable):
            return self.value * other.value
        if isinstance(other, (int, float, np.floating)):
            return self.value * other
        raise TypeError(f"Cannot multiply OptVariable with {type(other)}")


def ov(
    name: str,
    value: float,
    lower_bound: float,
    upper_bound: float,
    fixed: bool = False,
    description: Optional[str] = None,
):
    """Field factory for OptVariable"""
    return field(
        default_factory=lambda: OptVariable(
            name, value, lower_bound, upper_bound, fixed, description
        )
    )


class OptVariablesFrame:
    """
    Class to model the variables for an optimisation
    """

    def __new__(cls, *_args, **_kwargs):
        """
        Prevent instantiation of this class.
        """
        if cls == OptVariablesFrame:
            raise TypeError(
                "Cannot instantiate an OptVariablesFrame directly. It must be"
                " subclassed."
            )
        if not hasattr(cls, "__dataclass_fields__"):
            raise TypeError(f"{cls} must be annotated with '@dataclass'")
        for field_name in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            dcf: Field = cls.__dataclass_fields__[field_name]  # type: ignore[attr-defined]  # noqa: E501
            fact_inst = dcf.default_factory() if dcf.default_factory != MISSING else None
            if fact_inst is None:
                raise TypeError(
                    f"{field_name} must be wrapped in with 'ov' field factory"
                )
            if not isinstance(fact_inst, OptVariable):
                raise TypeError(
                    f"OptVariablesFrame contains non-OptVariable object '{field_name}:"
                    f" {type(fact_inst)}'"
                )
            if field_name != fact_inst.name:
                raise TypeError(
                    "OptVariablesFrame contains OptVariable with incorrect name"
                    f" '{fact_inst.name}', defined as '{field_name}'"
                )

        return super().__new__(cls)

    def __iter__(self) -> Generator[OptVariable, None, None]:
        """
        Iterate over this frame's parameters.

        The order is based on the order in which the parameters were
        declared.
        """
        for field_name in self.__dataclass_fields__:  # type: ignore[attr-defined]
            yield getattr(self, field_name)

    def __getitem__(self, name: str) -> OptVariable:
        """
        Dictionary-like access to variables.

        Parameters
        ----------
        name:
            Name of the variable to get
        """
        return getattr(self, name)

    def adjust_variable(
        self,
        name: str,
        value: Optional[float] = None,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        fixed: bool = False,
        strict_bounds: bool = True,
    ):
        """
        Adjust a variable in the frame.

        Parameters
        ----------
        name:
            Name of the variable to adjust
        value:
            Value of the variable to set
        lower_bound:
            Value of the lower bound to set
        upper_bound:
            Value of the upper to set
        fixed:
            Whether or not the variable is to be held constant
        strict_bounds: bool
            If True, will raise errors if values are outside the bounds. If False, the
            bounds are dynamically adjusted to match the value.
        """
        opt_var = self[name]
        if fixed:
            # sets the bounds and fixes the var ignoring them
            opt_var.adjust(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                strict_bounds=strict_bounds,
            )
            opt_var.fix(value)
        else:
            opt_var.adjust(value, lower_bound, upper_bound, strict_bounds)

    def adjust_variables(
        self,
        var_dict: Optional[VarDictT] = None,
        strict_bounds=True,
    ):
        """
        Adjust multiple variables in the frame.

        Parameters
        ----------
        var_dict:
            Dictionary with which to update the set, of the form
            {"var_name": {"value": v, "lower_bound": lb, "upper_bound": ub}, ...}
        strict_bounds: bool
            If True, will raise errors if values are outside the bounds. If False, the
            bounds are dynamically adjusted to match the value.
        """
        if var_dict is not None:
            for k, v in var_dict.items():
                args = [
                    v.get("value", None),
                    v.get("lower_bound", None),
                    v.get("upper_bound", None),
                    v.get("fixed", None),
                ]
                if all(i is None for i in args):
                    raise OptVariablesError(
                        "When adjusting variables in an OptVariableFrame instance, the"
                        " dictionary must be of the form: {'var_name': {'value': v,"
                        " 'lower_bound': lb, 'upper_bound': ub}, ...}"
                    )
                self.adjust_variable(k, *args, strict_bounds=strict_bounds)

    def fix_variable(self, name: str, value: Optional[float] = None):
        """
        Fix a variable in the frame, removing it from optimisation but preserving a
        constant value.

        Parameters
        ----------
        name:
            Name of the variable to fix
        value:
            Value at which to fix the variable (will default to present value)
        """
        self[name].fix(value)

    def get_normalised_values(self):
        """
        Get the normalised values of all free variables.

        Returns
        -------
        x_norm: np.ndarray
            Array of normalised values
        """
        return np.array([opv.normalised_value for opv in self._opt_vars])

    def set_values_from_norm(self, x_norm):
        """
        Set values from a normalised vector.

        Parameters
        ----------
        x_norm: np.ndarray
            Array of normalised values
        """
        true_values = self.get_values_from_norm(x_norm)
        for opv, value in zip(self._opt_vars, true_values):
            opv.value = value

    def get_values_from_norm(self, x_norm):
        """
        Get actual values from a normalised vector.

        Parameters
        ----------
        x_norm: np.ndarray
            Array of normalised values

        Returns
        -------
        x_true: np.ndarray
            Array of actual values in units
        """
        if len(x_norm) != self.n_free_variables:
            raise OptVariablesError(
                f"Number of normalised variables {len(x_norm)} !="
                f" {self.n_free_variables}."
            )
        return [
            opv.from_normalised(v_norm) for opv, v_norm in zip(self._opt_vars, x_norm)
        ]

    @property
    def names(self):
        """
        All variable names of the variable set.
        """
        return [opv.name for opv in self]

    @property
    def values(self):
        """
        All un-normalised values of the variable set (including fixed variable values).
        """
        return np.array([v.value for v in self])

    @property
    def n_free_variables(self) -> int:
        """
        Number of free variables in the set.
        """
        return len(self._opt_vars)

    @property
    def _opt_vars(self):
        return [v for v in self if not v.fixed]

    @property
    def _fixed_vars(self):
        return [v for v in self if v.fixed]

    @property
    def _fixed_variable_indices(self) -> list:
        """
        Indices of fixed variables in the set.
        """
        # specfically not useing self._fixed_vars here
        # as you need the correct index for the variable
        return [i for i, v in enumerate(self) if v.fixed]

    def as_dict(self) -> Dict[str, OptVarDictT]:
        """
        Dictionary Representation of the frame
        """
        return {opv.name: opv.as_dict() for opv in self}

    def as_serializable(self) -> Dict[str, OptVarSerializedT]:
        """
        Dictionary Representation of the frame
        """
        return {opv.name: opv.as_serializable() for opv in self}

    def to_json(self, file: str, **kwargs):
        """
        Save the OptVariablesFrame to a json file.

        Parameters
        ----------
        path: str
            Path to save the json file to.
        """
        json_writer(self.as_serializable(), file, **kwargs)

    @classmethod
    def from_json(cls, file: Union[Path, str, TextIO]):
        """
        Create an OptVariablesFrame instance from a json file.

        Parameters
        ----------
        file: Union[str, TextIO]
            The path to the file, or an open file handle that supports reading.
        """
        if isinstance(file, (Path, str)):
            with open(file) as fh:
                return cls.from_json(fh)

        d = json.load(file)
        opt_vars = {
            name: OptVariable.from_serialized(name, val) for name, val in d.items()
        }
        return cls(**opt_vars)

    def tabulate(self, tablefmt: str = "fancy_grid") -> str:
        """
        Tabulate OptVariablesFrame

        Parameters
        ----------
        keys: Optional[List]
            table column keys
        tablefmt: str (default="fancy_grid")
            The format of the table - see
            https://github.com/astanin/python-tabulate#table-format

        Returns
        -------
        tabulated: str
            The tabulated data
        """
        records = sorted([val.as_dict() for val in self], key=lambda x: x["name"])

        return f"{self.__class__.__name__}\n" + tabulate(
            records,
            headers="keys",
            tablefmt=tablefmt,
            showindex=False,
            numalign="right",
        )

    def __str__(self) -> str:
        """
        Pretty prints a representation of the OptVariablesFrame inside the console
        """
        return self.tabulate()

    def __repr__(self) -> str:
        """
        Prints a representation of the OptVariablesFrame inside the console
        """
        return (
            f"{self.__class__.__name__}(\n    "
            + "\n    ".join([repr(var) for var in self])
            + "\n)"
        )
