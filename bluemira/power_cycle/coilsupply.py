# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Classes for computing coils active and reactive powers.

TODO:
    - alter 'name' & 'description' to 'label' and 'name'?
    - relocate classes used by `net.py` and `coilsupply.py` to `base.py`
    - ensure every '...Config' class inherits from 'Config', and rename
      other cases as '...Inputs', '...Scheme', etc.
    - implement '_powerloads_from_wallpluginfo' method with `net.py`
      classes
    - remove dummy abstract method from `CoilSupplyABC` class that
      stops `ruff` complaints
    - alter '_config' in 'CoilSupplyConverter' to abstract property
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
    replace,
)
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from matplotlib import (
    pyplot as plt,
)

from bluemira.base.look_and_feel import bluemira_print
from bluemira.power_cycle.errors import CoilSupplySystemError
from bluemira.power_cycle.net import (
    Config,
    Descriptor,
)
from bluemira.power_cycle.tools import pp


def _get_module_class_from_str(class_name):
    return getattr(sys.modules[__name__], class_name)


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
    single_value_types = (bool, int, float, list, tuple, np.ndarray)

    @classmethod
    def is_parameter(cls, obj):
        """
        Check if the given object is an instance of 'CoilSupplyParameter'.

        This method is a substitute for 'isinstance', which fails in this
        implementation. It compares the name of the class of the object with
        the name of the class that is calling this method.

        Parameters
        ----------
        obj (Any): The object to check.

        Returns
        -------
        bool: True if 'obj' is a 'CoilSupplyParameter', False otherwise.
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
        if cls.is_parameter(argument):
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

    def __len__(self):
        """Get number of attributes in dataclass instance."""
        return len(asdict(self))

    def duplicate(self):
        """Create a duplicate of the dataclass instance."""
        return replace(self)

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
            if self.is_parameter(other):
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
                self_dict = {} if self_value is None else self_dict

            if isinstance(other_value, dict):
                other_dict = other_value
            else:
                other_dict = {other_key: other_value}
                other_dict = {} if other_value is None else other_dict

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


class LibraryDescriptor(Descriptor):
    """Descriptor to define libraries for use with dataclasses."""

    def __init__(self, *, config: Type[Config]):
        """See class docstring."""
        self.config = config

    def __get__(self, obj: Any, _) -> Dict[str, Config]:
        """Get all library entries."""
        return getattr(obj, self._name)

    def __set__(
        self,
        obj: Any,
        value: Dict[str, Union[Config, Dict]],
    ):
        """Set all library entries."""
        for k, v in value.items():
            if not isinstance(v, self.config):
                value[k] = self.config(name=k, **v)
        setattr(obj, self._name, value)


@dataclass
class CoilSupplyInputs:
    """Values used to characterize a Coil Supply System."""

    "Basic configuration for Coil Supply System."
    config: CoilSupplyConfigDescriptor = CoilSupplyConfigDescriptor()

    "Library of inputs for possible CoilSupplyCorrector objects."
    corrector_library: LibraryDescriptor = LibraryDescriptor(
        config=CoilSupplyCorrectorConfig,
    )

    "Library of inputs for possible CoilSupplyConverter objects."
    converter_library: LibraryDescriptor = LibraryDescriptor(
        config=CoilSupplyConverterConfig,
    )

    def _create_coilsupplyparameter_dataclass(self):
        """
        Create 'CoilSupplyParameter' specific for 'CoilSupplySystem'.

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
        """See class docstring."""
        self.name = config.name
        self.description = config.description
        all_resistances = asdict(config.resistance_set).values()
        if all(e >= 0 for e in all_resistances):
            self.resistance_set = config.resistance_set
        else:
            raise ValueError("All resistances must be non-negative.")

    def _correct(self, value: np.ndarray):
        return value * (1 + self.factor)

    def compute_correction(
        self,
        voltages_parameter: CoilSupplyParameterABC,
        currents_parameter: CoilSupplyParameterABC,
        switches_parameter: CoilSupplyParameterABC,
    ):
        """
        Apply the effect of the 'CoilSupplyCorrector'.

        Apply a correction due to the presence of the 'CoilSupplyCorrector'
        in coil circuits. Given a couple of concrete instances of the
        'CoilSupplyParameterABC' class that represent the demanded
        voltages and currents, compute the effect of the corrector to
        each attribute of the parameters.

        As a first approximation, neglect current reduction due to
        resistance of corrector device, and reduce total voltage
        by contribution to resistance connected in series.
        """
        voltages_corrector = voltages_parameter.duplicate()
        currents_corrector = currents_parameter.duplicate()

        voltages_following = voltages_parameter.duplicate()
        currents_following = currents_parameter.duplicate()

        coil_names = list(asdict(self.resistance_set).keys())
        for name in coil_names:
            requested_v = getattr(voltages_parameter, name)
            requested_i = getattr(currents_parameter, name)
            corrector_s = getattr(switches_parameter, name)

            corrector_resistance = getattr(self.resistance_set, name)
            corrector_i = requested_i
            corrector_v = -corrector_resistance * corrector_i
            if corrector_s is not None:
                corrector_i = np.multiply(corrector_i, corrector_s)
                corrector_v = np.multiply(corrector_v, corrector_s)
            setattr(voltages_corrector, name, corrector_v)
            setattr(currents_corrector, name, corrector_i)

            following_v = requested_v - corrector_v
            following_i = requested_i
            setattr(voltages_following, name, following_v)
            setattr(currents_following, name, following_i)

        '''
        plt.figure()
        ax = plt.axes()
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        colors.append("k")
        colors = iter(colors)
        for name in coil_names:
            pp(name)
            color = next(colors)
            corrector_resistance = getattr(self.resistance_set, name)
            requested_i = getattr(currents_parameter, name)
            corrector_i = requested_i
            resistance_v = -corrector_resistance * corrector_i
            # requested_v = getattr(voltages_parameter, name)
            corrector_v = getattr(voltages_corrector, name)
            # following_v = getattr(voltages_following, name)
            """
            corrector_s = getattr(switches_parameter, name)
            if corrector_s is None:
                corrector_s = [0 for v in corrector_v]
            """
            # ax.plot(requested_v, "-", color=color, label=f"{name} (V requested)")
            ax.plot(resistance_v, "-", color=color, label=f"{name} (V resistance)")
            ax.plot(corrector_v, "--", color=color, label=f"{name} (V corrector)")
            # ax.plot(following_v, ":", color=color, label=f"{name} (V following)")
            # ax.plot(corrector_s, "-", color=color, label=f"{name} (corrector switch)")
        plt.legend()
        ax.grid(True)
        ax.title.set_text(f"Corrector: {self.name}")
        # ax.set_ylabel("Switch [-]")
        ax.set_ylabel("Voltage [V]")
        ax.set_xlabel("Vector index [-]")
        '''

        return (
            voltages_following,
            currents_following,
            voltages_corrector,
            currents_corrector,
        )


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
        """See class docstring."""
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
        v_max_coil = np.max(np.absolute(voltages_array))
        if v_max_coil == 0:
            raise ValueError(
                "Voltage array must contain at least one value",
                "different than zero.",
            )
        number_of_bridge_units = np.ceil(v_max_coil / v_max_bridge)
        v_rated = number_of_bridge_units * v_max_bridge
        i_rated = max(currents_array)
        p_rated = v_rated * i_rated  # Probably wrong (maxV * maxI ≠ max power)

        p_apparent = v_rated * currents_array
        power_factor = voltages_array / v_rated
        phase_rad = np.arccos(power_factor)
        phase_deg = phase_rad * 180 / np.pi

        """
        from bluemira.power_cycle.tools import pp
        pp(v_max_coil)
        pp(number_of_bridge_units)
        pp(v_rated)
        pp(i_rated)
        pp(p_rated)
        # pp(power_factor)
        # pp(phase_rad)
        raise False
        """
        p_active = voltages_array * currents_array
        # p_reactive = np.sqrt(np.square(p_apparent) - np.square(p_active))

        # p_reactive = np.absolute(p_apparent * np.sin(phase_rad))  # why?
        p_reactive = np.absolute(p_apparent) * np.sin(phase_rad)

        # p_active = p_apparent * np.cos(phase_rad)  # why not absolute?
        # pp(np.cos(phase_rad))

        p_loss_multiplier = 1
        for percentage in loss_percentages:
            p_loss_multiplier *= 1 + loss_percentages[percentage] / 100
        p_losses = p_loss_multiplier * p_active
        p_active = p_active + p_losses

        return {
            f"{self.name}_voltages": voltages_array,
            f"{self.name}_currents": currents_array,
            "number_of_bridge_units": number_of_bridge_units,
            "voltage_rated": v_rated,
            "current_rated": i_rated,
            "power_rated": p_rated,
            "power_apparent": p_apparent,
            "phase_radians": phase_rad,
            "phase_degrees": phase_deg,
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
    inputs: CoilSupplyInputs
        All inputs for a characterization of a Coil Supply System.

    Attributes
    ----------
    correctors: Tuple[CoilSupplyCorrector]
        Ordered list of corrector system instances.
    converter: CoilSupplyConverter
        Single converter instance.
    """

    _computing_msg = "Computing coils power supply power loads..."

    def _just_to_stop_ruff_checks(self):
        pass

    def __init__(self, inputs: CoilSupplyInputs):
        """See class docstring."""
        self.inputs = inputs
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

    def create_parameter(self, obj=None):
        """
        Create parameter compatible with this 'CoilSupplySystem' instance.

        Use this method to transform objects into instances of the
        'CoilSupplyParameter' class, that have attributes that match
        the coil names in 'inputs.config.coil_names', and transform
        their values into 'numpy.ndarray' instances.
        """
        if isinstance(obj, dict):
            obj = {k: np.array(v) for k, v in obj.items()}
        return self.inputs.parameter.init_subclass(obj)

    def _validate_dict_of_parameters_for_correctors(self, obj=None):
        dict_of_parameters = {}
        for c in self.correctors:
            value = obj.get(c.name, None) if isinstance(obj, dict) else obj
            dict_of_parameters[c.name] = self.create_parameter(value)
        return dict_of_parameters

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
        times_argument: Optional[Any] = None,
        dict_of_switches_argument: Optional[Dict[str, Any]] = None,
        *,
        verbose: bool = False,
    ) -> CoilSupplyParameterABC:
        """
        Compute power loads required by coil supply system to feed coils.

        Arguments are transformed into appropriate CoilSupplyParameter
        instances (dictionaries are distributed by coil, single values
        are turned into arrays and copied for every coil).

        For each coil, voltage and current arrays must have the same
        length, and elements must be sampled at the same points in time.

        The dictionary of "switches argument" should contain one key for
        each corrector defined in the CoilSupplySystem instance. The
        value of each key should be an array of boolean values, with
        the same length as the voltage and current arrays, that specify
        at what intervals the effect of that corrector is applied to the
        voltages and currents arguments. Missing keys in this dictionary
        are filled with False values.

        Parameters
        ----------
        voltages_argument: Any
            Array (collection) of voltages in time required by the coils. [V]
        currents_argument: Any
            Array (collection) of currents in time required by the coils. [A]
        dict_of_switches_argument: Dict[str,Any]
            Arrays of...
        verbose: bool
            Print extra information and converter power factor angles.
        """
        outputs_parameter = self.create_parameter()

        voltages_parameter = self.create_parameter(voltages_argument)
        outputs_parameter.absorb_parameter(
            voltages_parameter,
            other_key="coil_voltages",
        )
        currents_parameter = self.create_parameter(currents_argument)
        outputs_parameter.absorb_parameter(
            currents_parameter,
            other_key="coil_currents",
        )
        if times_argument:
            times_parameter = self.create_parameter(times_argument)
            outputs_parameter.absorb_parameter(
                times_parameter,
                other_key="coil_times",
            )
        all_switches = self._validate_dict_of_parameters_for_correctors(
            dict_of_switches_argument,
        )

        for corrector in self.correctors:
            (
                voltages_parameter,
                currents_parameter,
                voltages_corrector,
                currents_corrector,
            ) = corrector.compute_correction(
                voltages_parameter,
                currents_parameter,
                all_switches[corrector.name],
            )
            outputs_parameter.absorb_parameter(
                voltages_corrector,
                other_key=f"{corrector.name}_voltages",
            )
            if verbose:
                outputs_parameter.absorb_parameter(
                    currents_corrector,
                    other_key=f"{corrector.name}_currents",
                )
        wallplug_parameter = self.create_parameter()

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
        outputs_parameter.absorb_parameter(wallplug_parameter)

        if verbose:
            plt.figure()
            ax = plt.axes()
            for name in self.inputs.config.coil_names:
                pp(name)
                wallplug_info = getattr(wallplug_parameter, name)
                n_units = wallplug_info["number_of_bridge_units"]
                phase_deg = wallplug_info["phase_degrees"]
                ax.plot(phase_deg, label=f"{name} ({n_units} bridge units)")
                pp(name + " number of bridge units: " + str(n_units))
                pp(" ")
            plt.legend()
            ax.grid(True)
            ax.set_ylabel("Phase (phi) [°]")
            ax.set_xlabel("Vector index [-]")

        if verbose:
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            colors_cycle = prop_cycle.by_key()["color"]
            for name in self.inputs.config.coil_names:
                plt.figure()
                ax = plt.axes()
                ax.title.set_text(name)
                colors = iter(colors_cycle)
                coil_parameter = getattr(outputs_parameter, name)
                coil_t = coil_parameter["coil_times"] if times_argument else None
                coil_v = coil_parameter["coil_voltages"]
                converter_v = coil_parameter["THY_voltages"]
                ax.plot(
                    coil_t,
                    coil_v,
                    color=next(colors),
                    label=f"coil: {name}",
                )
                for corrector in self.correctors:
                    color = next(colors)
                    pp(f"{name}: {corrector.name}")
                    corrector_v = coil_parameter[f"{corrector.name}_voltages"]
                    # corrector_i = coil_parameter[f"{corrector.name}_currents"]
                    ax.plot(
                        coil_t,
                        corrector_v,
                        color=color,
                        label=f"corrector: {corrector.name}",
                    )
                ax.plot(
                    coil_t,
                    converter_v,
                    color=next(colors),
                    label=f"converter: {self.converter.name}",
                )
                plt.legend()
                ax.grid(True)
                x_label = "Time [s]" if times_argument else "Vector index [-]"
                ax.set_xlabel(x_label)
                ax.set_ylabel("Voltage [V]")

        return outputs_parameter
