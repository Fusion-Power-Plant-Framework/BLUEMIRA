# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Classes for computing coils active and reactive powers.

TODO:
    - alter 'name' & 'description' to 'label' and 'name'?
    - relocate classes used by `net.py` and coils.py` to `base.py`
    - ensure every `...Config` class inherits from `Config`, and rename
      other cases as `...Inputs`, `...Scheme`, etc.
    - relocate `CoilSupplySystemError` to `errors.py`
    - implement `_powerloads_from_wallpluginfo` method
    - remove dummy abstract method from `CoilSupplyABC` class
    - modify config/input classes to inherit from bluemira `Parameter`
"""

import sys
from abc import ABC, abstractmethod
from dataclasses import (
    asdict,
    dataclass,
    field,
    fields,
    make_dataclass,
)
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from bluemira.base.look_and_feel import bluemira_print
from bluemira.power_cycle.errors import PowerCycleError
from bluemira.power_cycle.net import (
    Config,
    Descriptor,
    LibraryConfigDescriptor,
)
from bluemira.power_cycle.tools import pp


def _get_module_class_from_str(class_name):
    return getattr(sys.modules[__name__], class_name)


class CoilSupplySystemError(PowerCycleError):
    """
    Exception class for 'CoilSupplySystem' class of the Power Cycle module.
    """


class CoilSupplyABC(ABC):
    """Abstract base class for coil supply systems."""

    @abstractmethod
    def _just_to_stop_ruff_checks(self):
        """Define dummy method to stop ruff checks."""


@dataclass
class CoilSupplyParameterABC:
    """
    Specifier of parameters for a 'CoilSupplySystem' instance.

    Upon creation of a 'CoilSupplyInputs' instance, this class is used
    to specify the structure of parameters applied to methods of the
    'CoilSupplySystem' instance created with the 'CoilSupplyInputs'
    instance.
    """

    subclass_name = "CoilSupplyParameter"
    single_value_types = (int, float, list, tuple, np.ndarray)

    @classmethod
    def validate_parameter(cls, obj):
        """
        Validate 'obj' to be of the same class as the object testing it.

        Substitute method for 'isinstance', since it fails in this
        implementation.
        """
        return obj.__class__.__name__ == cls.__name__

    @classmethod
    def init_subclass(cls, argument: Any = None):
        """
        Create a 'CoilSupplyParameter' subclass instance from argument.

        If 'None' is given to instantiate the class, an empty instance
        is created.
        If an object of this class is given to instantiate the class, it
        is returned as is.
        If a 'dict' is given to instantiate the class, keys must match
        class attributes and their values are distributed.
        If a value of one of the 'single_value_types' classes is given
        to instantiate the class, copy that value to all attributes.
        """
        if argument is None:
            return cls()
        if cls.validate_parameter(argument):
            return argument
        if isinstance(argument, dict):
            return cls(**argument)
        if isinstance(argument, cls.single_value_types):
            args = {}
            all_fields = fields(cls)
            for one_field in all_fields:
                args[one_field.name] = argument
            return cls(**args)
        raise ValueError(
            "A 'CoilSupplyParameter' instance must be initialized "
            f"with 'None' for an empty instance, a '{cls.__name__}' "
            "instance for no alteration, a 'dict' for a distributed "
            "instantiation or any of the following types for a "
            f"single-value instantiation: {cls.single_value_types}. "
            f"Argument was '{type(argument)}' instead."
        )

    def absorb_parameter(
        self,
        other,
        self_key: str = "original",
        other_key: str = "absorbed",
    ):
        """
        Absorbe a 'CoilSupplyParameter' instance into another instance.

        If 'other' is a 'CoilSupplyParameter' instance, the data in its
        attributes are distributed over the attributes of the 'self'
        instance.
        If 'other' is a 'dict' or one of the 'single_value_types' classes,
        copy it to all attributes of the 'self' instance.

        The value stored in each attribute of the 'self' instance is
        always returned as a 'dict'. If it was not a dictionary, its
        value is stored in the 'self_key' key of the new 'dict'.

        The value stored in each attribute of the 'other' instance is
        added to its respective 'dict' in the returned 'self'. If it
        was not a dictionary, its value is stored in the 'other_key'
        key of that attribute's dictionary.
        """
        all_fields = fields(self)
        for one_field in all_fields:
            self_value = getattr(self, one_field.name)
            if self.validate_parameter(other):
                other_value = getattr(other, one_field.name)
            elif isinstance(other, (dict, self.single_value_types)):
                other_value = other
            else:
                raise TypeError(
                    "A 'CoilSupplyParameter' instance must absorb a "
                    f"'{self.__class__.__name__}' instance or any one of "
                    f"the following types: {self.single_value_types}. "
                    f"Argument was a '{type(other)}' instance instead."
                )

            if isinstance(self_value, dict):
                self_dict = self_value
            else:
                self_dict = {self_key: self_value}

            if isinstance(other_value, dict):
                other_dict = other_value
            else:
                other_dict = {other_key: other_value}

            setattr(self, one_field.name, {**self_dict, **other_dict})

        return self


class CoilSupplySubSystem(CoilSupplyABC):
    """Base class for subsystems of 'CoilSupplySystem' class."""

    def _just_to_stop_ruff_checks(self):
        pass


@dataclass
class CoilSupplyConfig:
    """Values that characterize a Coil Supply System."""

    "Description of the 'CoilSupplySystem' instance."
    description: str = "Coil Supply System"

    "Names of the coils to which power is supplied."
    coil_names: Union[None, List[str]] = None

    "Ordered list of names of corrector technologies, found in the"
    "library of 'CoilSupplyCorrectorConfig' entries, used to create the"
    "corresponding 'CoilSupplyCorrector' instances to be applied to"
    "each coil."
    corrector_technologies: Union[None, List[str]] = None

    "Single name of converter technology, found in the library of"
    "'CoilSupplyConverterConfig' entries, used to create the"
    "corresponding 'CoilSupplyConverter' instance to be applied to"
    "each coil."
    converter_technology: Union[None, str] = None


@dataclass
class CoilSupplyCorrectorConfig(Config):
    """Coil supply corrector config."""

    "Description of the 'CoilSupplyCorrector' instance."
    description: str

    "Equivalent resistance of the corrector of each coil. [Ω (ohm)]"
    "Must be a 'dict' with keys that match each 'str' in 'coil_names'"
    "of 'CoilSupplyConfig'. A single value is copied to all coils."
    resistance_set: Union[float, Dict[str, float]]


@dataclass
class CoilSupplyConverterConfig(Config):
    """Coil supply system config."""

    "Class used to build 'CoilSupplyConverter' instance."
    class_name: str

    "Arguments passed to build the 'CoilSupplyConverter' instance."
    class_args: Dict[str, Any]


class CoilSupplyConfigDescriptor(Descriptor):
    """Coil suppply config descriptor for use with dataclasses."""

    def __get__(self, obj: Any, _) -> CoilSupplyConfig:
        """Get the coil supply system config."""
        return getattr(obj, self._name)

    def __set__(self, obj: Any, value: Union[dict, CoilSupplyConfig]):
        """Set the coils supply system config."""
        if not isinstance(value, CoilSupplyConfig):
            value = CoilSupplyConfig(**value)
        setattr(obj, self._name, value)


@dataclass
class CoilSupplyInputs:
    """Values used to characterize a Coil Supply System."""

    config: CoilSupplyConfigDescriptor = CoilSupplyConfigDescriptor()
    corrector_library: LibraryConfigDescriptor = LibraryConfigDescriptor(
        config=CoilSupplyCorrectorConfig,
    )
    converter_library: LibraryConfigDescriptor = LibraryConfigDescriptor(
        config=CoilSupplyConverterConfig,
    )

    def _create_coilsupplyparameter_dataclass(self):
        """
        Dynamically create 'CoilSupplyParameter' dataclass inheriting
        from 'CoilSupplyParameterABC' to contain attributes that match
        'config.coil_names'.

        Based on:
        # https://stackoverflow.com/questions/52534427/dynamically-add-fields-to-dataclass-objects
        """
        parameter_fields = [
            (
                name,
                Any,
                field(default=None),
            )
            for name in self.config.coil_names
        ]
        parameter = CoilSupplyParameterABC()
        parameter.__class__ = make_dataclass(
            parameter.subclass_name,
            fields=parameter_fields,
            bases=(CoilSupplyParameterABC,),
        )
        self.parameter = parameter

    def _transform_resistance_sets_in_coilsupplyparameter(self):
        for config in self.corrector_library.values():
            config.resistance_set = self.parameter.init_subclass(
                config.resistance_set,
            )

    def __post_init__(self):
        """Complete __init__ by ajusting inputs."""
        self._create_coilsupplyparameter_dataclass()
        self._transform_resistance_sets_in_coilsupplyparameter()


class CoilSupplyCorrector(CoilSupplySubSystem):
    """
    Safety and auxiliary sub-systems for coil power supply systems.

    Class to represent safety and auxiliary sub-systems of a
    'CoilSupplySystem' object, that result in a partial voltage
    reduction due to an equivalent resistance.

    Parameters
    ----------
    config: CoilSupplyCorrectorConfig
        Object that characterizes a 'CoilSupplyCorrector' instance.
    """

    def __init__(self, config: CoilSupplyCorrectorConfig):
        self.name = config.name
        self.description = config.description
        all_resistances = asdict(config.resistance_set).values()
        if all(e >= 0 for e in all_resistances):
            self.resistance_set = config.resistance_set
        else:
            raise ValueError("All resistances must be non-negative.")

    def _correct(self, value: np.ndarray):
        return value * (1 + self.factor)

    def compute_correction(self, voltages_parameter, currents_parameter):
        """
        Apply correction to each attribute of a 'CoilSupplyParameter.

        As a first approximation, neglect current reduction due to
        resistance of corrector device, and reduce total voltage
        by contribution to resistance connected in series.
        """
        coil_names = list(asdict(self.resistance_set).keys())

        for name in coil_names:
            initial_voltages = getattr(voltages_parameter, name)
            initial_currents = getattr(currents_parameter, name)
            corrector_resistance = getattr(self.resistance_set, name)
            corrector_voltages = corrector_resistance * initial_currents
            final_voltages = initial_voltages - corrector_voltages
            setattr(voltages_parameter, name, final_voltages)
        return voltages_parameter, currents_parameter


class CoilSupplyConverter(CoilSupplySubSystem):
    """
    Class from which all Converter classes inherit.

    Class to represent power converter technology of a 'CoilSupplySystem'
    object, that computes the "wall-plug" power consumption by the coils
    supply system.
    """

    @property  # Should be abstract property instead?
    def _config(self):
        """Must be defined in subclasses."""
        raise NotImplementedError


@dataclass
class ThyristorBridgesConfig(Config):
    """Config for 'CoilSupplyConverter' using Thyristor Bridges tech."""

    "Description of the 'CoilSupplyConverter' instance."
    description: str

    "Maximum voltage allowed accross single thyristor bridge unit. [V]"
    max_bridge_voltage: float

    "Power loss percentages applied to active power demanded by the"
    "converter from the grid."
    power_loss_percentages: Dict[str, float]


class ThyristorBridges(CoilSupplyConverter):
    """
    Representation of power converter systems using Thyristor Bridges.

    This simplified model computes reactive power loads but does not
    account for power electronics dynamics and its associated control
    systems; it also neglects the following effects:
        - reductions allowed by sequential control of series-connects
            unit (as foreseen in ITER);
        - current circulation mode between bridges connects in parallel
            (since it is only expected at low currents, when reactive
            power is also low);
        - voltage drops in the transformer itself;
        - other non-linearities.

    Parameters
    ----------
    config: CoilSupplyCorrectorConfig
        Object that characterizes a 'CoilSupplyCorrector' instance.
    """

    _config = ThyristorBridgesConfig

    def __init__(self, config: ThyristorBridgesConfig):
        self.name = config.name
        self.description = config.description
        self.max_bridge_voltage = config.max_bridge_voltage
        self.power_loss_percentages = config.power_loss_percentages

    def _convert(self):
        pass

    def compute_conversion(self, voltages_array, currents_array):
        """
        Compute power loads required by converter to feed coils.

        Parameters
        ----------
        voltage: np.ndarray
            Array of voltages in time. [V]
        current: np.ndarray
            Array of currents in time. [A]
        """
        loss_percentages = self.power_loss_percentages
        v_max_bridge = self.max_bridge_voltage
        v_max_coil = np.max(voltages_array)
        if v_max_coil == 0:
            raise ValueError(
                "Voltage array must contain at least one value",
                "different than zero.",
            )
        number_of_bridge_units = np.ceil(v_max_coil / v_max_bridge)
        v_rated = number_of_bridge_units * v_max_bridge
        i_rated = max(currents_array)
        p_rated = v_rated * i_rated

        p_apparent = v_rated * currents_array
        phase = np.arccos(voltages_array / v_rated)
        power_factor = np.cos(phase)

        p_reactive = p_apparent * np.sin(phase)

        p_active = p_apparent * np.cos(phase)
        p_loss_multiplier = 1
        for percentage in loss_percentages:
            p_loss_multiplier *= 1 + loss_percentages[percentage] / 100
        p_losses = p_loss_multiplier * p_active
        p_active = p_active + p_losses

        return {
            "number_of_bridge_units": number_of_bridge_units,
            "voltage_rated": v_rated,
            "current_rated": i_rated,
            "power_rated": p_rated,
            "power_apparent": p_apparent,
            "phase": phase,
            "power_factor": power_factor,
            "power_losses": p_losses,
            "power_active": p_active,
            "power_reactive": p_reactive,
        }


class CoilSupplySystem(CoilSupplyABC):
    """
    Class that represents the complete coil supply systems in a power plant.

    Parameters
    ----------
    scheme: Union[CoilSupplyScheme, Dict]
        Coil Supply System characterization.
    corrector_library: Union[CoilSupplyCorrectorLibrary, Dict]
        Library of inputs for possible CoilSupplyCorrector objects.
    converter_library: Union[CoilSupplyConverterLibrary, Dict]
        Library of inputs for possible CoilSupplyConverter objects.

    Attributes
    ----------
    correctors: Dict[str, Tuple[CoilSupplyCorrector]]
        Ordered list of corrector system instances
    converter:
        blablabla
    """

    _computing_msg = "Computing coils power supply power loads..."

    def _just_to_stop_ruff_checks(self):
        pass

    def __init__(
        self,
        config: Union[CoilSupplyConfig, Dict[str, Any]],
        corrector_library: Dict[str, Tuple[CoilSupplyCorrector]],
        converter_library: Dict[str, Any],
    ):
        self.inputs = CoilSupplyInputs(
            config=config,
            corrector_library=corrector_library,
            converter_library=converter_library,
        )

        self.correctors = self._build_correctors()
        self.converter = self._build_converter()

    def _build_correctors(self) -> Tuple[CoilSupplyCorrector]:
        corrector_list = []
        for name in self.inputs.config.corrector_technologies:
            corrector_config = self.inputs.corrector_library[name]
            corrector_list.append(CoilSupplyCorrector(corrector_config))
        return tuple(corrector_list)

    def _build_converter(self) -> CoilSupplyConverter:
        name = self.inputs.config.converter_technology
        converter_inputs = self.inputs.converter_library[name]
        converter_class_name = converter_inputs.class_name
        converter_class = _get_module_class_from_str(converter_class_name)
        if issubclass(converter_class, CoilSupplyConverter):
            converter_args = converter_inputs.class_args
            converter_config = converter_class._config(
                name=name,
                **converter_args,
            )
        else:
            raise CoilSupplySystemError(
                f"Class '{converter_class_name}' is not an instance of the "
                "'CoilSupplyConverter' class."
            )
        return converter_class(converter_config)

    def validate_parameter(self, obj=None):
        """
        Create parameter compatible with this 'CoilSupplySystem' instance.

        Use this method to transform objects into instances of the
        'CoilSupplyParameter' class, that have attributes that match
        the coil names in 'inputs.config.coil_names'.
        """
        return self.inputs.parameter.init_subclass(obj)

    def _print_computing_message(self, verbose=False):
        if verbose:
            bluemira_print(self._computing_msg)

    def _powerloads_from_wallpluginfo(self, wallplug_info, verbose):
        """TODO: transform converter info into power loads."""
        self._print_computing_message(verbose=verbose)
        active_load = wallplug_info["power_active"]
        reactive_load = wallplug_info["power_reactive"]
        return active_load, reactive_load

    def compute_wallplug_loads(
        self,
        voltages_argument: Any,
        currents_argument: Any,
        verbose: bool = False,
    ):
        """
        Compute power loads required by coil supply system to feed coils.

        Parameters
        ----------
        voltages_parameter: Any
            Array of voltages in time required by the coils. [V]
        currents_parameter: Any
            Array of currents in time required by the coils. [A]
        """
        voltages_parameter = self.validate_parameter(
            np.array(voltages_argument),
        )
        currents_parameter = self.validate_parameter(
            np.array(currents_argument),
        )
        wallplug_parameter = self.validate_parameter()

        for corrector in self.correctors:
            (
                voltages_parameter,
                currents_parameter,
            ) = corrector.compute_correction(
                voltages_parameter,
                currents_parameter,
            )
            if verbose:
                pp(voltages_parameter)

        for name in self.inputs.config.coil_names:
            voltages_array = getattr(voltages_parameter, name)
            currents_array = getattr(currents_parameter, name)

            wallplug_info = self.converter.compute_conversion(
                voltages_array,
                currents_array,
            )
            active_load, reactive_load = self._powerloads_from_wallpluginfo(
                wallplug_info,
                verbose,
            )
            wallplug_info["active_load"] = active_load
            wallplug_info["reactive_load"] = reactive_load
            setattr(wallplug_parameter, name, wallplug_info)

        wallplug_parameter.absorb_parameter(
            voltages_parameter,
            other_key="coil_voltages",
        )
        wallplug_parameter.absorb_parameter(
            currents_parameter,
            other_key="coil_currents",
        )

        return wallplug_parameter
