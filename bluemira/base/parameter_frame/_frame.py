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
import json
from contextlib import suppress
from dataclasses import dataclass, fields
from operator import itemgetter
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_type_hints,
)
from typing import _GenericAlias as GenericAlias  # TODO python >=3.9 import from types

import pint
from tabulate import tabulate

from bluemira.base.constants import (
    ANGLE_UNITS,
    base_unit_defaults,
    combined_unit_defaults,
    combined_unit_dimensions,
    raw_uc,
)
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame._parameter import (
    ParamDictT,
    Parameter,
    ParameterValueType,
)

# due to circular import
if TYPE_CHECKING:
    from bluemira.base.reactor_config import ConfigParams

_PfT = TypeVar("_PfT", bound="ParameterFrame")


@dataclass
class ParameterFrame:
    """
    A data class to hold a collection of `Parameter` objects.

    The class should be declared using the following form:

    .. code-block:: python

        @dataclass
        class AnotherFrame(ParameterFrame):
            param_1: Parameter[float]
            param_2: Parameter[int]

    """

    def __new__(cls, *args, **kwargs):  # noqa: ARG003
        """
        Prevent instantiation of this class.
        """
        if cls == ParameterFrame:
            raise TypeError(
                "Cannot instantiate a ParameterFrame directly. It must be subclassed."
            )

        if cls != EmptyFrame and not cls.__dataclass_fields__:
            bluemira_warn(f"{cls} is empty, @dataclass is possibly missing")

        return super().__new__(cls)

    def __post_init__(self):
        """Get types from frame"""
        self._types = self._get_types()

        for field, field_name, value_type in zip(
            self, self.__dataclass_fields__, self._types.values()
        ):
            if not isinstance(field, Parameter):
                raise TypeError(
                    f"ParameterFrame contains non-Parameter object '{field_name}:"
                    f" {type(field)}'"
                )
            if field_name != field.name:
                raise TypeError(
                    "ParameterFrame contains Parameter with incorrect name"
                    f" '{field.name}', defined as '{field_name}'"
                )
            vt = _validate_parameter_field(field, value_type)

            val_unit = {
                "value": Parameter._type_check(field.name, field.value, vt),
                "unit": field.unit,
            }
            _validate_units(val_unit, vt)
            if field.unit != val_unit["unit"]:
                field.set_value(val_unit["value"], "unit enforcement")
                field._unit = pint.Unit(val_unit["unit"])
            else:
                # ensure int-> float conversion
                field._value = val_unit["value"]

    @classmethod
    def _get_types(cls) -> Dict[str, GenericAlias]:
        """Gets types for the frame even with annotations imported"""
        frame_type_hints = get_type_hints(cls)
        return {f.name: frame_type_hints[f.name] for f in fields(cls)}

    def __iter__(self) -> Generator[Parameter, None, None]:
        """
        Iterate over this frame's parameters.

        The order is based on the order in which the parameters were
        declared.
        """
        for field in fields(self):
            yield getattr(self, field.name)

    def update(
        self,
        new_values: Union[Dict[str, ParameterValueType], ParamDictT, ParameterFrame],
    ):
        """Update the given frame"""
        if isinstance(new_values, ParameterFrame):
            self.update_from_frame(new_values)
        else:
            try:
                self.update_from_dict(new_values)
            except TypeError:
                self.update_values(new_values)

    def get_values(self, *names: str) -> Tuple[ParameterValueType, ...]:
        """Get values of a set of Parameters"""
        try:
            return tuple(getattr(self, n).value for n in names)
        except AttributeError as ae:
            raise AttributeError(
                f"Parameters {[n for n in names if not hasattr(self, n)]} not in"
                " ParameterFrame"
            ) from ae

    def update_values(self, new_values: Dict[str, ParameterValueType], source: str = ""):
        """Update the given parameter values."""
        for key, value in new_values.items():
            param: Parameter = getattr(self, key)
            param.set_value(value, source)

    def update_from_dict(self, new_values: ParamDictT):
        """Update from a dictionary representation of a ``ParameterFrame``"""
        for key, value in new_values.items():
            if "name" in value:
                del value["name"]
            self._set_param(
                key,
                Parameter(
                    name=key,
                    **value,
                    _value_types=_validate_parameter_field(key, self._types[key]),
                ),
            )

    def update_from_frame(self, frame: ParameterFrame):
        """Update the frame with the values of another frame"""
        for o_param in frame:
            if hasattr(self, o_param.name):
                self._set_param(o_param.name, o_param)

    def _set_param(self, name: str, o_param: Parameter):
        """
        Sets the information from a Parameter to an existing Parameter in this frame.
        """
        param = getattr(self, name)
        param.set_value(
            (
                o_param.value
                if not param.unit or o_param.value is None
                else o_param.value_as(param.unit)
            ),
            o_param.source,
        )
        if o_param.long_name:
            param._long_name = o_param.long_name
        if o_param.description:
            param._description = o_param.description

    @classmethod
    def from_dict(
        cls: Type[_PfT],
        data: Dict[str, ParamDictT],
        allow_unknown=False,
    ) -> _PfT:
        """Initialize an instance from a dictionary."""
        data = copy.deepcopy(data)
        kwargs: Dict[str, Parameter] = {}
        for member in cls.__dataclass_fields__:
            try:
                param_data = data.pop(member)
            except KeyError as e:
                if cls.__dataclass_fields__[member].type == ClassVar:
                    continue
                raise ValueError(f"Data for parameter '{member}' not found.") from e

            kwargs[member] = cls._member_data_to_parameter(
                member,
                param_data,
            )

        if not allow_unknown and len(data) > 0:
            raise ValueError(f"Unknown parameter(s) {str(list(data))[1:-1]} in dict.")
        return cls(**kwargs)

    @classmethod
    def from_frame(cls: Type[_PfT], frame: ParameterFrame) -> _PfT:
        """Initialise an instance from another ParameterFrame."""
        kwargs = {}
        for field in cls.__dataclass_fields__:
            try:
                kwargs[field] = getattr(frame, field)
            except AttributeError:  # noqa: PERF203
                raise ValueError(
                    "Cannot create ParameterFrame from other. "
                    f"Other frame does not contain field '{field}'."
                ) from None
        return cls(**kwargs)

    @classmethod
    def from_json(cls: Type[_PfT], json_in: Union[str, json.SupportsRead]) -> _PfT:
        """Initialise an instance from a JSON file, string, or reader."""
        if hasattr(json_in, "read"):
            # load from file stream
            return cls.from_dict(json.load(json_in))
        if not isinstance(json_in, str):
            raise TypeError(f"Cannot read JSON from type '{type(json_in).__name__}'.")
        if not json_in.startswith("{"):
            # load from file
            with open(json_in) as f:
                return cls.from_dict(json.load(f))
        # load from a JSON string
        return cls.from_dict(json.loads(json_in))

    @classmethod
    def from_config_params(cls: Type[_PfT], config_params: ConfigParams) -> _PfT:
        """
        Initialise an instance from a
        :class:`~bluemira.base.reactor_config.ConfigParams` object.

        A ConfigParams objects holds a ParameterFrame of global_params
        and a dict of local_params. This function merges the two together
        to form a unified ParameterFrame.

        Parameters in global_params will overwrite those in
        local_params, when defined in both.
        All references to Parameters in global_params are maintained
        (i.e. there's no copying).
        """
        kwargs = {}

        lp = config_params.local_params
        for member in cls.__dataclass_fields__:
            if member not in lp:
                continue
            kwargs[member] = cls._member_data_to_parameter(
                member,
                lp[member],
            )

        gp = config_params.global_params
        for member in cls.__dataclass_fields__:
            if member not in gp.__dataclass_fields__:
                continue
            kwargs[member] = getattr(gp, member)

        # now validate all dataclass_fields are in kwargs
        # (which could be super set)
        for member in cls.__dataclass_fields__:
            try:
                kwargs[member]
            except KeyError as e:  # noqa: PERF203
                raise ValueError(f"Data for parameter '{member}' not found.") from e

        return cls(**kwargs)

    @classmethod
    def _member_data_to_parameter(
        cls,
        member: str,
        member_param_data: Dict,
    ) -> Parameter:
        value_type = _validate_parameter_field(member, cls._get_types()[member])
        try:
            _validate_units(member_param_data, value_type)
        except pint.errors.PintError as pe:
            raise ValueError("Unit conversion failed") from pe
        return Parameter(
            name=member,
            **member_param_data,
            _value_types=value_type,
        )

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Serialize this ParameterFrame to a dictionary."""
        out = {}
        for param_name in self.__dataclass_fields__:
            if self.__dataclass_fields__[param_name].type == ClassVar:
                continue
            param_data = getattr(self, param_name).to_dict()
            # We already have the name of the param, and use it as a
            # key. No need to repeat the name in the data, so pop it.
            param_data.pop("name")
            out[param_name] = param_data
        return out

    def tabulate(
        self,
        keys: Optional[List] = None,
        tablefmt: str = "fancy_grid",
        floatfmt: str = ".5g",
    ) -> str:
        """
        Tabulate the ParameterFrame

        Parameters
        ----------
        keys
            table column keys
        tablefmt
            The format of the table (default="fancy_grid") - see
            https://github.com/astanin/python-tabulate#table-format
        floatfmt
            Format floats to this precision

        Returns
        -------
        The tabulated data
        """
        column_widths = dict(
            zip(list(ParamDictT.__annotations__.keys()), [20, None, 20, 20, 20, 20])
        )
        columns = list(ParamDictT.__annotations__.keys()) if keys is None else keys
        rec_col = copy.deepcopy(columns)
        rec_col.pop(columns.index("name"))
        records = sorted(
            [
                [key, *[param.get(col, "N/A") for col in rec_col]]
                for key, param in self.to_dict().items()
            ]
        )
        # tabulate's floatfmt only works if the whole column is a float
        for r in records:
            if isinstance(r[1], float):
                r[1] = f"{r[1]: {floatfmt}}"

        return tabulate(
            records,
            headers=columns,
            tablefmt=tablefmt,
            showindex=False,
            numalign="right",
            maxcolwidths=list(itemgetter(*columns)(column_widths)),
        )

    def __str__(self) -> str:
        """
        Pretty print ParameterFrame
        """
        return self.tabulate()


def _validate_parameter_field(field, member_type: Type) -> Tuple[Type, ...]:
    if (member_type is not Parameter) and (
        not hasattr(member_type, "__origin__") or member_type.__origin__ is not Parameter
    ):
        raise TypeError(f"Field '{field}' does not have type Parameter.")
    return get_args(member_type)


def _validate_units(param_data: Dict, value_type: Iterable[Type]):
    try:
        quantity = pint.Quantity(param_data["value"], param_data["unit"])
    except KeyError as ke:
        raise ValueError("Parameters need a value and a unit") from ke
    except TypeError:
        if param_data["value"] is None:
            # dummy for None values
            quantity = pint.Quantity(
                1 if param_data["unit"] in (None, "") else param_data["unit"]
            )
        elif isinstance(param_data["value"], (bool, str)):
            param_data["unit"] = "dimensionless"
            return
        else:
            raise

    if dimensionality := quantity.units.dimensionality:
        unit = _fix_combined_units(_remake_units(dimensionality))
    else:
        unit = quantity.units

    unit = _fix_weird_units(unit, quantity.units)

    if unit != quantity.units and param_data["value"] is not None:
        val = raw_uc(quantity.magnitude, quantity.units, unit)
        if isinstance(param_data["value"], int) and int in value_type:
            val = int(val)
        param_data["value"] = val

    param_data["unit"] = f"{unit:~P}"


def _remake_units(dimensionality: Union[Dict, pint.util.UnitsContainer]) -> pint.Unit:
    """Reconstruct unit from its dimensionality"""
    dim_list = list(map(base_unit_defaults.get, dimensionality.keys()))
    dim_pow = list(dimensionality.values())
    return pint.Unit(".".join([f"{j[0]}^{j[1]}" for j in zip(dim_list, dim_pow)]))


def _fix_combined_units(unit: pint.Unit) -> pint.Unit:
    """Converts base unit to a composite unit if they exist in the defaults"""
    dim_keys = list(combined_unit_dimensions.keys())
    dim_val = list(combined_unit_dimensions.values())
    with suppress(ValueError):
        return pint.Unit(
            combined_unit_defaults[dim_keys[dim_val.index(unit.dimensionality)]]
        )
    return pint.Unit(unit)


def _convert_angle_units(
    modified_unit: pint.Unit, orig_unit_str: str, angle_unit: str
) -> pint.Unit:
    """
    Converts angle units to the base unit default for angles.

    Angles are dimensionless therefore dimensionality conversion
    from pint doesn't work. Conversions between angle units is also not
    very robust.

    Parameters
    ----------
    modified_unit
        reconstructed unit without the angle
    orig_unit_str
        the user supplied unit (without spaces)
    angle_unit
        the angle unit in `orig_unit`

    Returns
    -------
        the new unit

    """
    breaking_units = ["steradian", "square_degree"]
    new_angle_unit = base_unit_defaults["[angle]"]
    for b in breaking_units:
        if b in orig_unit_str:
            raise NotImplementedError(f"{breaking_units} not supported for conversion")
    if f"{angle_unit}**" in orig_unit_str:
        raise NotImplementedError("Exponent angles >1, <-1 are not supported")
    unit_list = orig_unit_str.split("/", 1)
    exp = "." if angle_unit in unit_list[0] else "/"
    modified_unit = "".join(str(modified_unit).split(angle_unit))
    return pint.Unit(f"{modified_unit}{exp}{new_angle_unit}")


def _fix_weird_units(modified_unit: pint.Unit, orig_unit: pint.Unit) -> pint.Unit:
    """
    Essentially a crude unit parser for when we have no dimensions
    or non-commutative dimensions.

    Full power years (dimension [time]) and displacements per atom (dimensionless)
    need to be readded to units as they will be removed by the dimensionality conversion.

    Angle units are dimensionless and conversions between them are not robust

    """
    unit_str = f"{orig_unit:C}"

    ang_unit = [ang for ang in ANGLE_UNITS if ang in unit_str]
    if len(ang_unit) > 1:
        raise ValueError(f"More than one angle unit not supported...🤯 {orig_unit}")
    ang_unit = ang_unit[0] if len(ang_unit) == 1 else None

    fpy = "full_power_year" in unit_str
    dpa = "displacements_per_atom" in unit_str

    if not (fpy or dpa) and ang_unit is None:
        return pint.Unit(modified_unit)

    new_unit = _non_comutative_unit_conversion(
        dict(modified_unit.dimensionality), unit_str.split("/")[0], dpa, fpy
    )

    # Deal with angles
    return _convert_angle_units(new_unit, unit_str, ang_unit) if ang_unit else new_unit


def _non_comutative_unit_conversion(dimensionality, numerator, dpa, fpy):
    """
    Full power years (dimension [time]) and displacements per atom (dimensionless)
    need to be readded to units as they will be removed by the dimensionality conversion.

    Full power years even though time based is not the same as straight 'time' and
    is therefore dealt with after other standard unit conversions.

    Only first order of both of these units is dealt with.
    """
    dpa_str = (
        ("dpa." if "displacements_per_atom" in numerator else "dpa^-1.") if dpa else ""
    )
    if fpy:
        if "full_power_year" in numerator:
            dimensionality["[time]"] += -1
            fpy_str = "fpy."
        else:
            dimensionality["[time]"] += 1
            fpy_str = "fpy^-1."

        if dimensionality["[time]"] == 0:
            del dimensionality["[time]"]
    else:
        fpy_str = ""

    return pint.Unit(
        f"{dpa_str}{fpy_str}{_fix_combined_units(_remake_units(dimensionality))}"
    )


@dataclass
class EmptyFrame(ParameterFrame):
    """
    Class to represent an empty `ParameterFrame` (one with no Parameters).

    Can be used when initializing a
    :class:`~bluemira.base.reactor_config.ConfigParams` object with no global params.
    """

    def __init__(self) -> None:
        super().__init__()


def make_parameter_frame(
    params: Union[Dict[str, ParamDictT], ParameterFrame, ConfigParams, str, None],
    param_cls: Type[_PfT],
) -> Union[_PfT, None]:
    """
    Factory function to generate a `ParameterFrame` of a specific type.

    Parameters
    ----------
    params: Union[Dict[str, ParamDictT], ParameterFrame, ConfigParams str, None]
        The parameters to initialise the class with.
        This parameter can be several types:

            * Dict[str, ParamDictT]:
                A dict where the keys are parameter names, and the
                values are the data associated with the corresponding
                name.
            * ParameterFrame:
                A reference to the parameters on this frame will be
                assigned to the new ParameterFrame's parameters. Note
                that this makes no copies, so updates to parameters in
                the new frame will propagate to the old, and vice versa.
            * :class:`~bluemira.base.reactor_config.ConfigParams`:
                An object that holds a `global_params` ParameterFrame
                and a `local_params` dict, which are merged to create
                a new ParameterFrame. Values defined in `local_params`
                will be overwritten by those in `global_params` when
                defined in both.
            * str:
                The path to a JSON file, or, if the string starts with
                '{', a JSON string.
            * None:
                For the case where no parameters are actually required.
                This is intended for internal use, to aid in validation
                of parameters in `Builder`\\s and `Designer`\\s.

    param_cls: Type[ParameterFrame]
        The `ParameterFrame` class to create a new instance of.

    Returns
    -------
        A frame of the type `param_cls`, or `None` if `params` and
        `param_cls` are both `None`.
    """
    from bluemira.base.reactor_config import ConfigParams

    if param_cls is None:
        if params is None:
            # Case for where there are no parameters associated with the object
            return params
        raise ValueError("Cannot process parameters, 'param_cls' is None.")
    if isinstance(params, dict):
        return param_cls.from_dict(params)
    if isinstance(params, param_cls):
        return params
    if isinstance(params, str):
        return param_cls.from_json(params)
    if isinstance(params, ParameterFrame):
        return param_cls.from_frame(params)
    if isinstance(params, ConfigParams):
        return param_cls.from_config_params(params)
    raise TypeError(f"Cannot interpret type '{type(params)}' as {param_cls.__name__}.")
