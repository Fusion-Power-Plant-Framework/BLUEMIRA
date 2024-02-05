# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Power cycle net loads"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypedDict,
    Union,
)

import numpy as np
from scipy.interpolate import interp1d
from typing_extensions import NotRequired

from bluemira.base.constants import raw_uc
from bluemira.base.look_and_feel import bluemira_debug
from bluemira.power_cycle.tools import read_json

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt


@dataclass
class Config:
    """Configuration base dataclass"""

    name: str


class LoadType(Enum):
    """Possibly load types"""

    ACTIVE = auto()
    REACTIVE = auto()

    @classmethod
    def from_str(cls, load_type: Union[str, LoadType, None]) -> Union[LoadType, None]:
        """Create loadtype from str"""
        if isinstance(load_type, str):
            return cls[load_type.upper()]
        return load_type

    @classmethod
    def check(
        cls, load_type: Union[str, LoadType, None]
    ) -> Union[Type[LoadType], Set[LoadType]]:
        """Check for all loadtypes or specific loadtype"""
        return (
            cls
            if load_type is None
            else {cls[load_type.upper()] if isinstance(load_type, str) else load_type}
        )


class LoadModel(Enum):
    """
    Members define possible models used.

    Maps model names to 'interp1d' interpolation behaviour.
    """

    RAMP = "linear"
    STEP = "previous"


@dataclass
class Efficiency:
    """Efficiency data container"""

    value: Union[float, Dict[str, float], Dict[LoadType, float]]
    description: str = ""

    def __post_init__(self):
        """Enforce value structure"""
        self.value = efficiency_type(self.value)


class EfficiencyDictType(TypedDict):
    """Typing for efficiency object"""

    value: Dict[Union[LoadType, str], float]
    description: NotRequired[str]


class Descriptor:
    """Data class property descriptor

    See https://docs.python.org/3/library/dataclasses.html#descriptor-typed-fields
    """

    def __set_name__(self, _, name: str):
        """Set the attribute name from a dataclass"""
        self._name = f"_{name}"


class ActiveReactiveDescriptor(Descriptor):
    """Descriptor for setting up active and reactive data dictionaries"""

    def __get__(
        self, obj: Any, _
    ) -> Union[Callable[[], Dict[LoadType, np.ndarray]], Dict[LoadType, np.ndarray]]:
        """Get the config"""
        if obj is None:
            return lambda: {
                LoadType.ACTIVE: np.arange(2),
                LoadType.REACTIVE: np.arange(2),
            }
        return getattr(obj, self._name)

    def __set__(
        self,
        obj: Any,
        value: Union[
            Callable[[], Dict[LoadType, np.ndarray]],
            Dict[Union[LoadType, str], Union[np.ndarray, list]],
            np.ndarray,
        ],
    ):
        """Setup the config"""
        if callable(value):
            value = value()

        if isinstance(value, (np.ndarray, list)):
            value = {
                LoadType.ACTIVE: np.asarray(value),
                LoadType.REACTIVE: np.asarray(value).copy(),
            }
        else:
            ld_t = {LoadType.ACTIVE, LoadType.REACTIVE}
            value = {
                k if isinstance(k, LoadType) else LoadType[k.upper()]: np.asarray(v)
                for k, v in value.items()
            }
            if missing_keys := ld_t - value.keys():
                value[missing_keys.pop()] = np.zeros_like(
                    value[(value.keys() - missing_keys).pop()]
                )
        setattr(obj, self._name, value)


def efficiency_type(
    value: Union[float, Dict[str, float], Dict[LoadType, float]],
) -> Dict[LoadType, float]:
    """Convert efficiency value to the correct structure"""
    if isinstance(value, (float, int)):
        value = {LoadType.ACTIVE: value, LoadType.REACTIVE: value}
    else:
        ld_t = {LoadType.ACTIVE, LoadType.REACTIVE}
        value = {
            k if isinstance(k, LoadType) else LoadType[k.upper()]: v
            for k, v in value.items()
        }
        if missing_keys := ld_t - value.keys():
            value[missing_keys.pop()] = np.ones_like(
                value[(value.keys() - missing_keys).pop()]
            )
    return value


class PhaseEfficiencyDescriptor(Descriptor):
    """Efficiency descriptor for use with dataclasses"""

    def __get__(
        self, obj: Any, _
    ) -> Union[Callable[[], Dict], Dict[str, List[Efficiency]]]:
        """Get the config"""
        if obj is None:
            return dict
        return getattr(obj, self._name)

    def __set__(
        self,
        obj: Any,
        value: Union[
            Callable[[], Dict], Dict[str, List[Union[Efficiency, EfficiencyDictType]]]
        ],
    ):
        """Setup the config"""
        if callable(value):
            value = value()
        for k, val in value.items():
            for no, v in enumerate(val):
                if not isinstance(v, Efficiency):
                    v["value"] = efficiency_type(v["value"])
                    value[k][no] = Efficiency(**v)

        setattr(obj, self._name, value)


class LoadEfficiencyDescriptor(Descriptor):
    """Efficiency descriptor for use with dataclasses"""

    def __get__(self, obj: Any, _) -> Union[Callable[[], list], List[Efficiency]]:
        """Get the config"""
        if obj is None:
            return list
        return getattr(obj, self._name)

    def __set__(
        self,
        obj: Any,
        value: Union[Callable[[], list], List[Union[EfficiencyDictType, Efficiency]]],
    ):
        """Setup the config"""
        if callable(value):
            value = value()
        for no, v in enumerate(value):
            if not isinstance(v, Efficiency):
                v["value"] = efficiency_type(v["value"])
                value[no] = Efficiency(**v)

        setattr(obj, self._name, value)


@dataclass
class ScenarioConfig(Config):
    """Power cycle scenario config"""

    pulses: dict[str, int]
    description: str = ""


@dataclass
class PulseConfig(Config):
    """Power cycle pulse config"""

    phases: list[str]
    description: str = ""


@dataclass
class PhaseConfig(Config):
    """Power cycle phase config"""

    operation: str
    subphases: list[str]
    description: str = ""


@dataclass
class SubPhaseConfig(Config):
    """SubPhase Config"""

    duration: Union[float, str]
    loads: list[str] = field(default_factory=list)
    efficiencies: PhaseEfficiencyDescriptor = PhaseEfficiencyDescriptor()
    unit: str = "s"
    description: str = ""
    reference: str = ""

    def __post_init__(self):
        """Enforce unit conversion"""
        if isinstance(self.duration, (float, int)):
            self.duration = raw_uc(self.duration, self.unit, "second")
            self.unit = "s"


@dataclass
class SystemConfig(Config):
    """Power cycle system config"""

    subsystems: List[str]
    description: str = ""


@dataclass
class SubSystemConfig(Config):
    """Power cycle sub system config"""

    loads: List[str]
    description: str = ""


@dataclass
class LoadConfig(Config):
    """Power cycle load config"""

    time: ActiveReactiveDescriptor = ActiveReactiveDescriptor()
    data: ActiveReactiveDescriptor = ActiveReactiveDescriptor()
    efficiencies: LoadEfficiencyDescriptor = LoadEfficiencyDescriptor()
    model: Union[LoadModel, str] = LoadModel.RAMP
    unit: str = "W"
    description: str = ""
    normalised: bool = True
    consumption: bool = True

    def __post_init__(self):
        """Validate load"""
        if isinstance(self.model, str):
            self.model = LoadModel[self.model.upper()]
        for lt in LoadType:
            if self.data[lt].size != self.time[lt].size:
                raise ValueError(
                    f"time and data must be the same length for {self.name}: "
                    f"{self.data[lt]}"
                )
            if any(np.diff(self.time[lt]) < 0):
                raise ValueError("time must increase")

            self.data[lt] = raw_uc(self.data[lt], self.unit, "W")
        self.unit = "W"

    def interpolate(
        self,
        time: npt.ArrayLike,
        end_time: Optional[float] = None,
        load_type: Union[str, LoadType] = LoadType.ACTIVE,
    ) -> np.ndarray:
        """
        Interpolate load for a given time vector

        Notes
        -----
        The interpolation type is set by load.model.
        Any out-of-bound values are set to zero.
        """
        if isinstance(load_type, str):
            load_type = LoadType[load_type.upper()]
        return interp1d(
            self.time[load_type],
            self.data[load_type],
            kind=self.model.value,
            bounds_error=False,  # turn-off error for out-of-bound
            fill_value=(0, 0),  # below-/above-bounds extrapolations
        )(time if self.normalised or end_time is None else np.array(time) * end_time)


def interpolate_extra(vector: npt.NDArray, n_points: int):
    """
    Add points between each point in a vector.
    """
    if n_points == 0:
        return vector

    return np.concatenate([
        *(
            np.linspace(vector[s], vector[s + 1], n_points + 1, endpoint=False)
            for s in range(len(vector) - 1)
        ),
        np.atleast_1d(vector[-1]),
    ])


def _normalise_timeseries(
    time: npt.ArrayLike,
    end_time: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[float]]:
    time = np.asarray(time)
    if min(time) < 0:
        raise NotImplementedError("Negative time not supported")

    if max(time) > 1:
        mx_time = max(time)
        return time / mx_time, mx_time if end_time is None else end_time
    return time, end_time


class LoadSet:
    """LoadSet of a phase"""

    def __init__(
        self,
        loads: Dict[str, LoadConfig],
    ):
        self._loads = loads

    @staticmethod
    def _consumption_flag(consumption: Optional[bool] = None) -> Set[bool]:
        return {True, False} if consumption is None else {consumption}

    def get_load_data_with_efficiencies(
        self,
        timeseries: np.ndarray,
        load_type: Union[str, LoadType],
        unit: Optional[str] = None,
        end_time: Optional[float] = None,
        *,
        consumption: Optional[bool] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get load data taking into account efficiencies and consumption

        Parameters
        ----------
        timeseries:
            time array
        load_type:
            Type of load
        unit:
            return unit, defaults to [W] or [var]
        end_time:
            for unnormalised loads this assures the load is
            applied at the right point in time
        consumption:
            return only consumption loads
        """
        data = self.get_explicit_data_consumption(
            timeseries, load_type, unit, end_time, consumption=consumption
        )
        load_check = LoadType.check(load_type)
        for ld_name in data:
            load_conf = self._loads[ld_name]
            for eff in load_conf.efficiencies:
                for eff_type, eff_val in eff.value.items():
                    if eff_type in load_check:
                        data[ld_name] *= eff_val

        return data

    def get_explicit_data_consumption(
        self,
        timeseries: np.ndarray,
        load_type: Union[str, LoadType],
        unit: Optional[str] = None,
        end_time: Optional[float] = None,
        *,
        consumption: Optional[bool] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get data with consumption resulting in an oppositely signed load

        Parameters
        ----------
        timeseries:
            time array
        load_type:
            Type of load
        unit:
            return unit, defaults to [W] or [var]
        end_time:
            for unnormalised loads this assures the load is
            applied at the right point in time
        consumption:
            return only consumption loads
        """
        loads = self.get_interpolated_loads(
            timeseries, load_type, unit, end_time, consumption=consumption
        )
        return {
            ld_n: -ld if self._loads[ld_n].consumption else ld
            for ld_n, ld in loads.items()
        }

    def build_timeseries(
        self,
        load_type: Optional[Union[str, LoadType]] = None,
        end_time: Optional[float] = None,
        *,
        consumption: Optional[bool] = None,
    ) -> np.ndarray:
        """Build a combined time series based on loads"""
        times = []
        for load in self._loads.values():
            for time in self._gettime(
                load, load_type, self._consumption_flag(consumption)
            ):
                if load.normalised:
                    times.append(time)
                else:
                    times.append(time / (max(time) if end_time is None else end_time))
        return np.unique(np.concatenate(times))

    @staticmethod
    def _gettime(
        load, load_type: Optional[Union[str, LoadType]], consumption_check: Set[bool]
    ):
        load_type = LoadType.from_str(load_type)
        if load.consumption in consumption_check:
            if load_type is None:
                yield from load.time.values()
            else:
                yield load.time[load_type]

    def get_interpolated_loads(
        self,
        timeseries: np.ndarray,
        load_type: Union[str, LoadType],
        unit: Optional[str] = None,
        end_time: Optional[float] = None,
        *,
        consumption: Optional[bool] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get loads for a given time series

        Parameters
        ----------
        timeseries:
            time array
        load_type:
            Type of load
        unit:
            return unit, defaults to [W] or [var]
        end_time:
            for unnormalised loads this assures the load is
            applied at the right point in time
        consumption:
            return only consumption loads
        """
        timeseries, end_time = _normalise_timeseries(timeseries, end_time)
        load_type = LoadType.from_str(load_type)
        _cnsmptn = self._consumption_flag(consumption)

        return {
            load.name: load.interpolate(timeseries, end_time, load_type)
            if unit is None
            else raw_uc(
                load.interpolate(timeseries, end_time, load_type),
                load.unit,
                unit,
            )
            for load in self._loads.values()
            if load.consumption in _cnsmptn
        }

    def load_total(
        self,
        timeseries: np.ndarray,
        load_type: Union[str, LoadType],
        unit: Optional[str] = None,
        end_time: Optional[float] = None,
        *,
        consumption: Optional[bool] = None,
    ) -> np.ndarray:
        """Total load for each timeseries point for a given load_type

        Parameters
        ----------
        timeseries:
            time array
        load_type:
            Type of load
        unit:
            return unit, defaults to [W] or [var]
        end_time:
            for unnormalised loads this assures the load is
            applied at the right point in time
        consumption:
            return only consumption loads
        """
        return np.sum(
            list(
                self.get_load_data_with_efficiencies(
                    timeseries, load_type, unit, end_time, consumption=consumption
                ).values()
            ),
            axis=0,
        )


@dataclass
class Phase:
    """Phase container"""

    config: PhaseConfig
    subphases: Dict[str, SubPhaseConfig]
    loads: LoadSet

    def __post_init__(self):
        """Validate duration"""
        if self.duration < 0:
            raise ValueError(
                f"{self.config.name} phase duration must be positive: {self.duration}s"
            )

    @property
    def duration(self):
        """Duration of phase"""
        return getattr(np, self.config.operation)([
            s_ph.duration for s_ph in self.subphases.values()
        ])

    def _process_phase_efficiencies(
        self, loads: Dict[str, np.ndarray], load_type: Union[str, LoadType]
    ):
        load_check = LoadType.check(load_type)
        for subphase in self.subphases.values():
            self._find_duplicate_loads(subphase, loads)
            for eff_name, effs in subphase.efficiencies.items():
                for eff in effs:
                    for eff_type, eff_val in eff.value.items():
                        if eff_type in load_check and eff_name in loads:
                            loads[eff_name] *= eff_val
        return loads

    @staticmethod
    def _find_duplicate_loads(subphase: SubPhaseConfig, loads: Dict[str, np.ndarray]):
        """Add duplication efficiency.

        If a load is duplicated in subphase.loads,
        the resultant data array is multiplied by the number of repeats
        """
        u, c = np.unique(subphase.loads, return_counts=True)
        counts = c[c > 1]
        for cnt, dup in enumerate(u[c > 1]):
            if dup in loads:
                eff = counts[cnt]
                bluemira_debug(f"Duplicate load {dup}, duplication efficiency of {eff}")
                loads[dup] *= eff

    def build_timeseries(
        self,
        load_type: Optional[Union[str, LoadType]] = None,
        *,
        consumption: Optional[bool] = None,
    ) -> np.ndarray:
        """Build a combined time series based on loads"""
        return (
            self.loads.build_timeseries(
                load_type=load_type, end_time=self.duration, consumption=consumption
            )
            * self.duration
        )

    def load_total(
        self,
        timeseries: np.ndarray,
        load_type: Union[str, LoadType],
        unit: Optional[str] = None,
        *,
        consumption: Optional[bool] = None,
    ) -> np.ndarray:
        """Total load for each timeseries point for a given load_type

        Parameters
        ----------
        timeseries:
            time array
        load_type:
            Type of load
        unit:
            return unit, defaults to [W] or [var]
        consumption:
            return only consumption loads
        """
        timeseries, _ = _normalise_timeseries(timeseries, self.duration)
        return np.sum(
            list(
                self.get_load_data_with_efficiencies(
                    timeseries, load_type, unit, consumption=consumption
                ).values()
            ),
            axis=0,
        )

    def get_load_data_with_efficiencies(
        self,
        timeseries: np.ndarray,
        load_type: Union[str, LoadType],
        unit: Optional[str] = None,
        *,
        consumption: Optional[bool] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get load data taking into account efficiencies and consumption

        Parameters
        ----------
        timeseries:
            time array
        load_type:
            Type of load
        unit:
            return unit, defaults to [W] or [var]
        consumption:
            return only consumption loads
        """
        timeseries, _ = _normalise_timeseries(timeseries, self.duration)
        return self._process_phase_efficiencies(
            self.loads.get_load_data_with_efficiencies(
                timeseries,
                load_type,
                unit,
                end_time=self.duration,
                consumption=consumption,
            ),
            load_type,
        )


class PulseDictType(TypedDict):
    """Pulse dictionary typing"""

    repeat: int
    data: Dict[str, Phase]


class LibraryConfig:
    """Power Cycle Configuration"""

    def __init__(
        self,
        scenario: ScenarioConfig,
        pulse: Dict[str, PulseConfig],
        phase: Dict[str, PhaseConfig],
        subphase: Dict[str, SubPhaseConfig],
        system: Dict[str, SystemConfig],
        subsystem: Dict[str, SubSystemConfig],
        loads: Dict[str, LoadConfig],
        durations: Optional[Dict[str, float]] = None,
    ):
        self.scenario = scenario
        self.pulse = pulse
        self.phase = phase
        self.subphase = subphase
        self.system = system
        self.subsystem = subsystem
        self.loads = loads
        self._import_subphase_duration(durations)

    def check_config(self):
        """Check powercycle configuration"""
        ph_keys = self.phase.keys()
        sph_keys = self.subphase.keys()
        ss_keys = self.subsystem.keys()
        loads = self.loads.keys()
        # scenario has known pulses
        if unknown_pulse := self.scenario.pulses.keys() - self.pulse.keys():
            raise ValueError(f"Unknown pulses {unknown_pulse}")
        # pulses have known phases
        for pulse in self.pulse.values():
            if unknown_phase := pulse.phases - ph_keys:
                raise ValueError(f"Unknown phases {unknown_phase}")
        # phases have known subphases
        for ph_c in self.phase.values():
            if unknown_s_ph := ph_c.subphases - sph_keys:
                raise ValueError(f"Unknown subphase configurations {unknown_s_ph}")
        # subphases have known loads
        for subphase in self.subphase.values():
            if unknown_load := subphase.loads - loads:
                raise ValueError(f"Unknown loads {unknown_load}")

        # systems have known subsystems
        for sys_c in self.system.values():
            if unknown := sys_c.subsystems - ss_keys:
                raise ValueError(f"Unknown subsystem configurations {unknown}")
        # subsystems have known loads
        for s_sys_c in self.subsystem.values():
            if unknown_load := s_sys_c.loads - loads:
                raise ValueError(f"Unknown loads {unknown_load}")

    def _import_subphase_duration(
        self, subphase_duration_params: Optional[Dict[str, float]] = None
    ):
        """Import subphase data"""
        for s_ph in self.subphase.values():
            if isinstance(s_ph.duration, str):
                key = s_ph.duration.replace("$", "")
                if subphase_duration_params is None:
                    raise KeyError(key)
                s_ph.duration = subphase_duration_params[key]

    def add_load_config(
        self,
        load: LoadConfig,
        subphases: Optional[Union[str, Iterable[str]]] = None,
        subphase_efficiency: Optional[List[Efficiency]] = None,
    ):
        """Add load config"""
        self.loads[load.name] = load
        self.link_load_to_subphase(load.name, subphases or [], subphase_efficiency)

    def link_load_to_subphase(
        self,
        load_name: str,
        subphases: Union[str, Iterable[str]],
        subphase_efficiency: Optional[List[Efficiency]] = None,
    ):
        """Link a load to a specific subphase"""
        if isinstance(subphases, str):
            subphases = [subphases]
        for subphase in subphases:
            self.subphase[subphase].loads.append(load_name)
            if subphase_efficiency is not None:
                self.subphase[subphase].efficiencies[load_name] = subphase_efficiency

    @classmethod
    def from_json(
        cls,
        manager_config_path: Union[Path, str],
        durations: Optional[Dict[str, float]] = None,
    ):
        """Create configuration from pure json"""
        return cls.from_dict(read_json(manager_config_path), durations)

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], durations: Optional[Dict[str, float]] = None
    ):
        """Create configuration from dictionary"""
        return cls(
            scenario=ScenarioConfig(**data["scenario"]),
            pulse={
                k: PulseConfig(name=k, **v) for k, v in data["pulse_library"].items()
            },
            phase={
                k: PhaseConfig(name=k, **v) for k, v in data["phase_library"].items()
            },
            subphase={
                k: SubPhaseConfig(name=k, **v)
                for k, v in data["subphase_library"].items()
            },
            system={
                k: SystemConfig(name=k, **v) for k, v in data["system_library"].items()
            },
            subsystem={
                k: SubSystemConfig(name=k, **v)
                for k, v in data["sub_system_library"].items()
            },
            loads={k: LoadConfig(name=k, **v) for k, v in data["load_library"].items()},
            durations=durations,
        )

    def get_phase(self, phase: str, *, check=True) -> Phase:
        """Create a single phase object"""
        if check:
            self.check_config()

        phase_config = self.phase[phase]
        subphases = {k: self.subphase[k] for k in phase_config.subphases}

        return Phase(
            phase_config,
            subphases,
            LoadSet({
                ld: self.loads[ld]
                for subphase in subphases.values()
                for ld in subphase.loads
            }),
        )

    def get_pulse(self, pulse: str, *, check=True) -> Dict[str, Phase]:
        """Create a pulse dictionary"""
        if check:
            self.check_config()
        return {
            phase: self.get_phase(phase, check=False)
            for phase in self.pulse[pulse].phases
        }

    def get_scenario(self) -> Dict[str, PulseDictType]:
        """Create a scenario dictionary"""
        self.check_config()
        return {
            pulse: {"repeat": reps, "data": self.get_pulse(pulse, check=False)}
            for pulse, reps in self.scenario.pulses.items()
        }
