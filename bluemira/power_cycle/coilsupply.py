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
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Tuple, Union

import numpy as np

from bluemira.base.look_and_feel import bluemira_print
from bluemira.power_cycle.errors import PowerCycleError
from bluemira.power_cycle.net import (
    Config,
    Descriptor,
    LibraryConfigDescriptor,
)


def _get_module_class_from_str(class_name):
    return getattr(sys.modules[__name__], class_name)


class CoilSupplySystemError(PowerCycleError):
    """
    Exception class for 'CoilSupplySystem' class of the Power Cycle module.
    """


class CoilVariable(Enum):
    """
    Possible coil variables.

    Members define possible variables demanded by coils, modified by a
    'CoilSupplySystem' object.
    """

    VOLTAGE = "voltage"
    CURRENT = "current"


class CoilSupplyABC(ABC):
    """Abstract base class for coil supply systems."""

    @abstractmethod
    def _just_to_stop_ruff_checks(self):
        """Define dummy method to stop ruff checks."""


class CoilSupplySubSystem(CoilSupplyABC):
    """Base class for subsystems of 'CoilSupplySystem' class."""

    def _just_to_stop_ruff_checks(self):
        pass


@dataclass
class CoilSupplyConfig:
    """Values that characterize a Coil Supply System."""

    "Description of the 'CoilSupplySystem' instance."
    description: str = "Coil Supply System"

    "Ordered labels to identify 'CoilSupplyCorrector' instances needed."
    correctors_tuple: Tuple[Union[None, str]] = ()

    "Label to identify 'CoilSupplyConverter' instance needed."
    converter_technology: Union[None, str] = None


@dataclass
class CoilSupplyCorrectorConfig(Config):
    """Coil supply corrector config."""

    "Description of the 'CoilSupplyCorrector' instance."
    description: str

    "Member of 'CoilVariable' corrected by the 'CoilSupplyCorrector' instance."
    correction_variable: CoilVariable

    "Dimensionless value that quantitavely defines the correction. [-]"
    correction_factor: float


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


class CoilSupplyCorrector(CoilSupplySubSystem):
    """
    Safety and auxiliary sub-systems for coil power supply systems.

    Class to represent safety and auxiliary sub-systems of a
    'CoilSupplySystem' object, that result in a correction of currents
    or voltages demanded by the coils.

    Parameters
    ----------
    config: CoilSupplyCorrectorConfig
        Object that characterizes a 'CoilSupplyCorrector' instance.
    """

    def __init__(self, config: CoilSupplyCorrectorConfig):
        self.name = config.name
        self.description = config.description
        self.variable = CoilVariable(config.correction_variable)
        self.factor = config.correction_factor

    def _correct(self, value: np.ndarray):
        return value * (1 + self.factor)

    def compute_correction(self, voltage, current):
        """Apply correction factor to a member of CoilVariable."""
        if self.variable == CoilVariable.VOLTAGE:
            voltage = self._correct(voltage)
        elif self.variable == CoilVariable.CURRENT:
            current = self._correct(current)
        else:
            raise ValueError(
                f"Unknown routine for correcting variable '{self.variable}'."
            )
        return voltage, current


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

    "Power loss percentages applied to coil power."
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

    def compute_conversion(self, voltage, current):
        """
        Compute power loads required by converter to feed coils.

        Parameters
        ----------
        voltage: np.ndarray
            voltage array
        current: np.ndarray
            current array
        """
        loss_percentages = self.power_loss_percentages
        v_max_bridge = self.max_bridge_voltage
        v_max_coil = np.max(voltage)
        if v_max_coil == 0:
            raise ValueError(
                "Voltage array must contain at least one value",
                "different than zero.",
            )
        number_of_bridge_units = np.ceil(v_max_coil / v_max_bridge)
        v_rated = number_of_bridge_units * v_max_bridge

        p_apparent = v_rated * current
        phase = np.arccos(voltage / v_rated)
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
    correctors_list:
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
        correctors: Dict[str, Any],
        converters: Dict[str, Any],
    ):
        self.inputs = CoilSupplyInputs(
            config=config,
            corrector_library=correctors,
            converter_library=converters,
        )

        self.correctors_list = self._build_correctors_list()
        self.converter = self._build_converter()

    def _build_correctors_list(self):
        correctors_list = []
        for name in self.inputs.config.correctors_tuple:
            corrector_config = self.inputs.corrector_library[name]
            correctors_list.append(CoilSupplyCorrector(corrector_config))
        return correctors_list

    def _build_converter(self):
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

    def _issue_computing_message(self, verbose=False):
        if verbose:
            bluemira_print(self._computing_msg)

    def _powerloads_from_wallpluginfo(self, wallplug_info):
        """TODO: transform converter info into power loads."""
        self._issue_computing_message()
        return wallplug_info

    def compute_wallplug_loads(self, coil_voltage, coil_current):
        """
        Compute power loads required by coil supply system to feed coils.

        Parameters
        ----------
        coil_voltage: Union[np.array, List[float] ]
            Array of voltages in time required by the coils. [V]
        coil_current: Union[np.array, List[float] ]
            Array of currents in time required by the coils. [V]
        """
        voltages = np.array(coil_voltage)
        currents = np.array(coil_current)
        if len(voltages) != len(currents):
            raise CoilSupplySystemError(
                "Current and voltage vectors must have the same length!"
            )
        for corrector in self.correctors_list:
            corrector.compute_correction(voltages, currents)
        wallplug_info = self.converter.compute_conversion(voltages, currents)
        return self._powerloads_from_wallpluginfo(wallplug_info)
