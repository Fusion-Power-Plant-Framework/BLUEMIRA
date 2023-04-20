# COPYRIGHT PLACEHOLDER

"""
Classes for the definition of power loads in the power cycle model.
"""
import copy
import sys
from enum import Enum
from typing import List, Union

import numpy as np
from scipy.interpolate import interp1d

from bluemira.power_cycle.base import PowerCycleLoadABC
from bluemira.power_cycle.errors import (
    LoadDataError,
    PhaseLoadError,
    PowerLoadError,
    PulseLoadError,
)
from bluemira.power_cycle.time import PowerCyclePhase, PowerCyclePulse
from bluemira.power_cycle.tools import (
    unnest_list,
    validate_axes,
    validate_list,
    validate_numerical,
)

CURVE_TEXT_IN_LABEL = " (curve)"


class LoadData(PowerCycleLoadABC):
    """
    Class to store a set of time and data vectors, to be used in
    creating 'PowerLoad' objects.

    Takes a pair of (time,data) vectors and creates a 'LoadData' object
    used to build power load objects to represent the time evolution
    of a given power in the plant.
    Instances of this class do not specify any dependence between the
    data points it stores, so no method is defined for altering (e.g.
    applying a multiplicative efficiency) or calculating related values
    (e.g. interpolation). Instead, these actions are performed with
    objects of the 'PowerLoad' class, that are built with instances of
    'LoadData'.

    Parameters
    ----------
    name: str
        Description of the 'LoadData' instance.
    time: int | float | list[int | float]
        List of time values that define the LoadData. [s]
    data: int | float | list[int | float]
        List of power values that define the LoadData. [W]

    Properties
    ----------
    intrinsic_time: list[int | float]
        Deep copy of the list stored in the 'time' attribute.
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    def __init__(
        self,
        name,
        time: Union[int, float, List[Union[int, float]]],
        data: Union[int, float, List[Union[int, float]]],
    ):
        super().__init__(name)

        self.data = validate_list(data)
        self.time = validate_list(time)
        self._is_increasing(self.time)

        self._sanity()

        self._norm = []  # Memory for time normalization
        self._shift = []  # Memory for time shifting

    @staticmethod
    def _is_increasing(parameter):
        """
        Validate a parameter for creation of a class instance to be an
        (not necessarily strictly) increasing list.
        """
        if not all(i <= j for i, j in zip(parameter, parameter[1:])):
            raise LoadDataError("increasing")
        return parameter

    def _sanity(self):
        """
        Validate 'data' and 'time' attributes to have both the same
        length, so that they univocally represent power values in time.
        """
        if len(self.data) != len(self.time):
            raise LoadDataError("sanity")

    @classmethod
    def null(cls):
        """
        Instantiates a null version of the class.
        """
        name = "Null LoadData"
        time = [0, 1]
        data = [0, 0]
        null_instance = cls(name, time, data)
        return null_instance

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------
    def _normalize_time(self, new_end_time):
        """
        Normalize values stored in the 'time' attribute, so that the
        last time value coincides with 'new_end_time'.
        Stores the normalization factor in the attribute '_norm', which
        is always initialized as an empty list.
        """
        old_time = self.time
        old_end_time = old_time[-1]
        norm = new_end_time / old_end_time
        new_time = [norm * t for t in old_time]
        self._is_increasing(new_time)
        self.time = new_time
        self._norm.append(norm)

    def _shift_time(self, time_shift):
        """
        Shift all time values in the 'time' attribute by the numerical
        value 'time_shift'.
        Stores the shifting factor in the attribute '_shift', which
        is always initialized as an empty list.
        """
        time_shift = validate_numerical(time_shift)
        time = self.time
        shifted_time = [t + time_shift for t in time]
        self.time = shifted_time
        self._shift.append(time_shift)

    def make_consumption_explicit(self):
        """
        Modifies the instance by turning every non-positive value stored
        in 'data' into its opposite.
        """
        for i, value in enumerate(self.data):
            if value > 0:
                self.data[i] = -value

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    @property
    def intrinsic_time(self):
        """
        Deep copy of the time vector contained in the 'time' attribute.
        """
        time = copy.deepcopy(self.time)
        return time

    @intrinsic_time.setter
    def intrinsic_time(self, value) -> None:
        raise PowerLoadError("time")

    def plot(self, ax=None, **kwargs):
        """
        Plot the points that define the 'LoadData' instance.

        This method applies the 'matplotlib.pyplot.scatter' imported
        method to the vectors that define the 'LoadData' instance. The
        default options for this plot are defined as class attributes,
        but can be overridden.

        Parameters
        ----------
        ax: Axes
            Instance of the 'matplotlib.axes.Axes' class, in which to
            plot. If 'None' is given, a new instance of axes is created.
        **kwargs: dict
            Options for the 'scatter' method.

        Returns
        -------
        ax: Axes
            Instance of the 'matplotlib.axes.Axes' class.
        list_of_plot_objects: list
            List of plot objects created by the 'matplotlib' package.
            The first element of the list is the plot object created
            using the 'pyplot.scatter', while the second element of the
            list is the plot object created using the 'pyplot.text'
            method.
        """
        ax = validate_axes(ax)

        # Set each default options in kwargs, if not specified
        default_scatter_kwargs = self._scatter_kwargs
        final_kwargs = {**default_scatter_kwargs, **kwargs}

        list_of_plot_objects = []

        plot_object = ax.scatter(
            self.time,
            self.data,
            label=self.name + " (data)",
            **final_kwargs,
        )
        list_of_plot_objects.append(plot_object)

        plot_object = self._add_text_to_point_in_plot(
            ax,
            self.name,
            self.time,
            self.data,
            **kwargs,
        )
        list_of_plot_objects.append(plot_object)

        return ax, list_of_plot_objects

    # ------------------------------------------------------------------
    # ARITHMETICS
    # ------------------------------------------------------------------
    def __add__(self, other):
        """
        Addition is not defined for 'LoadData' instances.
        """
        raise LoadDataError("add")


class LoadModel(Enum):
    """
    Members define possible models used by the methods defined in the
    'PowerLoad' class to compute values between load definition points.

    The 'name' of a member is a 'str' that roughly describes the
    interpolation behavior, while its associated 'value' is a 'str' that
    specifies which kind of interpolation is applied when calling the
    imported 'scipy.interpolate.interp1d' method.
    """

    RAMP = "linear"  # 'interp1d' linear interpolation
    STEP = "previous"  # 'interp1d' previous-value interpolation


class PowerLoad(PowerCycleLoadABC):
    """
    Generic representation of a power load.

    Defines a power load with a set of 'LoadData' instances. Each
    instance must be accompanied by a 'LoadModel' specification, used to
    compute additional values between data points. This enables the
    instance to compute time-dependent curves.

    Instances of the 'PowerLoad' class can be added to each other, and
    a list of them can be summed. Instances can also be multiplied and
    divided by scalar numerical values.

    Parameters
    ----------
    name: str
        Description of the 'PowerLoad' instance.
    loaddata_set: LoadData | list[LoadData]
        Collection of instances of the 'LoadData' class that define
        the 'PowerLoad' object.
    loadmodel_set: LoadModel | list[LoadModel]
        Mathematical loadmodel used to compute values between
        'loaddata_set' definition points.

    Properties
    ----------
    intrinsic_time: list[int | float]
        List that contains all values in the 'intrinsic_time' properties
        of the different 'LoadData' objects contained in the
        'loaddata_set' attribute, ordered and with no repetitions.
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------
    _n_points = 100

    def __init__(
        self,
        name,
        loaddata_set: Union[LoadData, List[LoadData]],
        loadmodel_set: Union[LoadModel, List[LoadModel]],
    ):
        super().__init__(name)

        self.loaddata_set = self._validate_loaddata_set(loaddata_set)
        self.loadmodel_set = self._validate_loadmodel_set(loadmodel_set)

        self._sanity()

    @staticmethod
    def _validate_loaddata_set(loaddata_set):
        """
        Validate 'loaddata_set' input to be a list of 'LoadData'
        objects.
        """
        loaddata_set = validate_list(loaddata_set)
        for element in loaddata_set:
            LoadData.validate_class(element)
        return loaddata_set

    @staticmethod
    def _validate_loadmodel_set(loadmodel_set):
        """
        Validate 'loadmodel_set' input to be a list of 'LoadModel'
        objects.
        """
        loadmodel_set = validate_list(loadmodel_set)
        for element in loadmodel_set:
            if not isinstance(element, LoadModel):
                element_class = type(element)
                raise PowerLoadError(
                    "loadmodel",
                    "One of the arguments provided is an instance of "
                    f"the {element_class!r} class instead.",
                )
        return loadmodel_set

    def _sanity(self):
        """
        Validate instance to have 'loaddata_set' and 'loadmodel_set'
        attributes of same length.
        """
        if len(self.loaddata_set) != len(self.loadmodel_set):
            raise PowerLoadError("sanity")

    @classmethod
    def null(cls):
        """
        Instantiates an null version of the class.
        """
        null_loaddata = LoadData.null()

        name = "Null PowerLoad"
        loaddata_set = null_loaddata
        loadmodel_set = LoadModel["RAMP"]
        null_instance = cls(name, loaddata_set, loadmodel_set)
        return null_instance

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    @staticmethod
    def _single_curve(loaddata, loadmodel, time):
        """
        Method that applies the 'scipy.interpolate.interp1d' imported
        method to a single instance of the 'LoadData' class. The kind
        of interpolation is determined by the 'loadmodel' input. Values
        are returned at the times specified in the 'time' input, with
        any out-of-bound values set to zero.
        """
        try:
            interpolation_kind = loadmodel.value
        except AttributeError:
            raise PowerLoadError("loadmodel")

        interpolation_operator = interp1d(
            loaddata.time,
            loaddata.data,
            kind=interpolation_kind,
            bounds_error=False,  # turn-off error for out-of-bound
            fill_value=(0, 0),  # below-/above-bounds extrapolations
        )

        interpolated_curve = list(interpolation_operator(time))
        return interpolated_curve

    @staticmethod
    def _validate_curve_input(time):
        """
        Validate the 'time' input for the 'curve' method to be a list of
        numeric values.
        """
        time = validate_list(time)
        for element in time:
            if not isinstance(element, (int, float)):
                raise PowerLoadError("curve")
        return time

    def curve(self, time):
        """
        Create a curve by calculating power load values at the specified
        times.

        This method applies the 'scipy.interpolate.interp1d' imported
        method to each 'LoadData' object stored in the 'loaddata_set'
        attribute and sums the results. The kind of interpolation is
        determined by each respective value in the 'loadmodel_set'
        attribute. Any out-of-bound values are set to zero.

        Parameters
        ----------
        time: int | float | list[ int | float ]
            List of time values. [s]

        Returns
        -------
        curve: list[float]
            List of power values. [W]
        """
        time = self._validate_curve_input(time)
        preallocated_curve = np.zeros(len(time))

        zip_of_sets = zip(self.loaddata_set, self.loadmodel_set)
        for loaddata, loadmodel in zip_of_sets:
            single_curve = self._single_curve(loaddata, loadmodel, time)
            preallocated_curve += np.array(single_curve)
        curve = preallocated_curve.tolist()

        return curve

    def _normalize_time(self, new_end_time):
        """
        Normalize the time of all 'LoadData' objects stored in the
        'loaddata_set' attribute, so that their last time values
        coincide with 'new_end_time'.
        """
        loaddata_set = self.loaddata_set
        _ = [ld._normalize_time(new_end_time) for ld in loaddata_set]
        self.loaddata_set = loaddata_set

    def _shift_time(self, time_shift):
        """
        Shift the 'time' attribute of all 'LoadData' objects in the
        'loaddata_set' attribute by the numerical value 'time_shift'.
        """
        loaddata_set = self.loaddata_set
        _ = [ld._shift_time(time_shift) for ld in loaddata_set]
        self.loaddata_set = loaddata_set

    def make_consumption_explicit(self):
        """
        Calls 'make_consumption_explicit' on every element of the
        'loaddata_set' attribute.
        """
        self._recursive_make_consumption_explicit(self.loaddata_set)

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    @property
    def intrinsic_time(self):
        """
        Single time vector that contains all values used to define the
        different 'LoadData' objects contained in the 'loaddata_set'
        attribute, ordered and with no repetitions.
        """
        return self._build_time_from_load_set(self.loaddata_set)

    @intrinsic_time.setter
    def intrinsic_time(self, value) -> None:
        raise PowerLoadError("time")

    def plot(self, ax=None, n_points=None, detailed=False, **kwargs):
        """
        Plot a 'PowerLoad' curve, built using the attributes that define
        the instance. The number of points interpolated in each curve
        segment can be specified.

        This method applies the 'matplotlib.pyplot.plot' imported
        method to a list of values built using the 'curve' method.
        The default options for this plot are defined as class
        attributes, but can be overridden.

        This method can also plot the individual 'LoadData' objects
        stored in the 'loaddata_set' attribute that define the
        'PowerLoad' instance.

        Parameters
        ----------
        ax: Axes
            Instance of the 'matplotlib.axes.Axes' class, in which to
            plot. If 'None' is given, a new instance of axes is created.
        n_points: int
            Number of points interpolated in each curve segment. The
            default value is 'None', which indicates to the method
            that the default value should be used, defined as a class
            attribute.
        detailed: bool
            Determines whether the plot will include all individual
            'LoadData' objects (computed with their respective
            'loadmodel_set' entries), that summed result in the normal
            plotted curve. These objects are plotted as secondary plots,
            as defined in 'PowerCycleABC' class. By default this input
            is set to 'False'.
        **kwargs: dict
            Options for the 'plot' method.

        Returns
        -------
        ax: Axes
            Instance of the 'matplotlib.axes.Axes' class.
        list_of_plot_objects: list
            List of plot objects created by the 'matplotlib' package.
            The first element of the list is the plot object created
            using the 'pyplot.plot', while the second element of the
            list is the plot object created using the 'pyplot.text'
            method.
            If the 'detailed' argument is set to 'True', the list
            continues to include the lists of plot objects created by
            the 'LoadData' class, with the addition of plotted curves
            for the visualization of the model selected for each load.
        """
        ax = validate_axes(ax)
        n_points = self._validate_n_points(n_points)

        # Set each default options in kwargs, if not specified
        default_plot_kwargs = self._plot_kwargs
        final_kwargs = {**default_plot_kwargs, **kwargs}

        intrinsic_time = self.intrinsic_time
        computed_time = self._refine_vector(intrinsic_time, n_points)
        computed_curve = self.curve(computed_time)

        list_of_plot_objects = []

        # Plot curve as line
        plot_object = ax.plot(
            computed_time,
            computed_curve,
            label=self.name + CURVE_TEXT_IN_LABEL,
            **final_kwargs,
        )
        list_of_plot_objects.append(plot_object)

        # Add descriptive text next to curve
        plot_object = self._add_text_to_point_in_plot(
            ax,
            self.name,
            computed_time,
            computed_curve,
            **kwargs,
        )
        list_of_plot_objects.append(plot_object)

        if detailed:
            zip_of_sets = zip(self.loaddata_set, self.loadmodel_set)
            for loaddata, loadmodel in zip_of_sets:
                current_curve = self._single_curve(
                    loaddata,
                    loadmodel,
                    computed_time,
                )

                # Plot current LoadData with seconday kwargs
                loaddata._make_secondary_in_plot()
                ax, current_plot_list = loaddata.plot(ax=ax)

                # Plot current curve as line with secondary kwargs
                kwargs.update(loaddata._plot_kwargs)
                plot_object = ax.plot(
                    computed_time,
                    current_curve,
                    **kwargs,
                )
                current_plot_list.append(plot_object)

                list_of_plot_objects.append(current_plot_list)

        return ax, list_of_plot_objects

    # ------------------------------------------------------------------
    # ARITHMETICS
    # ------------------------------------------------------------------
    def __add__(self, other):
        """
        The addition of 'PowerLoad' instances creates a new 'PowerLoad'
        instance with joined 'loaddata_set' and 'loadmodel_set'
        attributes.
        """
        this = copy.deepcopy(self)
        other = copy.deepcopy(other)
        return PowerLoad(
            "Resulting PowerLoad",
            this.loaddata_set + other.loaddata_set,
            this.loadmodel_set + other.loadmodel_set,
        )

    def __mul__(self, number):
        """
        An instance of the 'PowerLoad' class can only be multiplied by
        scalar numerical values.
        The multiplication of a 'PowerLoad' instance by a number
        multiplies all values in the 'data' attributes of 'LoadData'
        objects stored in the 'loaddata_set' by that number.
        """
        number = validate_numerical(number)
        other = copy.deepcopy(self)
        for loaddata in other.loaddata_set:
            multiplied_data = [d * number for d in loaddata.data]
            loaddata.data = multiplied_data
        return other

    def __truediv__(self, number):
        """
        An instance of the 'PowerLoad' class can only be divided by
        scalar numerical values.
        The division of a 'PowerLoad' instance by a number
        divides all values in the 'data' attributes of 'LoadData'
        objects stored in the 'loaddata_set' by that number.
        """
        number = validate_numerical(number)
        other = copy.deepcopy(self)
        for loaddata in other.loaddata_set:
            divided_data = [d / number for d in loaddata.data]
            loaddata.data = divided_data
        return other


class PhaseLoad(PowerCycleLoadABC):
    """
    Generic representation of the total power load during a pulse phase.

    Defines the phase load with a set of 'PowerLoad' instances. Each
    instance must be accompanied by a 'normalize' specification, used to
    indicate whether that power load must have its curve normalized in
    time in respect to the 'duration' of the 'phase' Parameter. This
    enables the instance to adjust the evolution of power loads
    accordingly, if changes occur to the plant pulse.

    Parameters
    ----------
    name: str
        Description of the 'PhaseLoad' instance.
    phase: PowerCyclePhase
        Pulse phase specification, that determines in which phase the
        load happens.
    powerload_set: PowerLoad | list[PowerLoad]
        Collection of instances of the 'PowerLoad' class that define
        the 'PhaseLoad' object.
    normalize: bool | list[bool]
        List of boolean values that defines which elements of
        'powerload_set' have their time-dependence normalized in respect
        to the phase duration. A value of 'True' forces a normalization,
        while a value of 'False' does not and time values beyond the
        phase duration are ignored.

    Properties
    ----------
    intrinsic_time: list[int | float]
        List that contains all values in the 'intrinsic_time' properties
        of the different 'PowerLoad' objects contained in the
        'powerload_set' attribute, ordered and with no repetitions.
    normalized_time: list[int | float]
        List that contains all values in the 'intrinsic_time' properties
        of the different 'PowerLoad' objects contained in the
        '_normalized_set' attribute, ordered and with no repetitions.
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    # Override number of points
    _n_points = 100

    # Override pyplot defaults
    _plot_defaults = {
        "c": "k",  # Line color
        "lw": 2,  # Line width
        "ls": "-",  # Line style
    }

    # Defaults for detailed plots
    _detailed_defaults = {
        "c": "k",  # Line color
        "lw": 1,  # Line width
        "ls": "--",  # Line style
    }

    def __init__(self, name, phase, powerload_set, normalize):
        super().__init__(name)

        self.phase = self._validate_phase(phase)
        self.powerload_set = self._validate_powerload_set(powerload_set)
        self.normalize = self._validate_normalize(normalize)

        self._sanity()

    @staticmethod
    def _validate_phase(phase):
        """
        Validate 'phase' input to be a PowerCycleTimeline instance.
        """
        PowerCyclePhase.validate_class(phase)
        return phase

    @staticmethod
    def _validate_powerload_set(powerload_set):
        """
        Validate 'powerload_set' input to be a list of 'PowerLoad'
        instances.
        """
        powerload_set = validate_list(powerload_set)
        for element in powerload_set:
            PowerLoad.validate_class(element)
        return powerload_set

    @staticmethod
    def _validate_normalize(normalize):
        """
        Validate 'normalize' input to be a list of boolean values.
        """
        normalize = validate_list(normalize)
        for element in normalize:
            if not isinstance(element, (bool)):
                raise PhaseLoadError(
                    "normalize",
                    f"Element {element!r} of 'normalize' list is an "
                    f"instance of the {type(element)!r} class instead.",
                )
        return normalize

    def _sanity(self):
        """
        Validate instance to have 'powerload_set' and 'normalize'
        attributes of same length.
        """
        if len(self.powerload_set) != len(self.normalize):
            raise PhaseLoadError("sanity")

    @classmethod
    def null(cls, phase):
        """
        Instantiates an null version of the class.
        """
        null_powerload = PowerLoad.null()

        name = "Null PhaseLoad for phase " + phase.name
        powerload_set = [null_powerload]
        normalize = True
        null_instance = cls(name, phase, powerload_set, normalize)
        return null_instance

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    @property
    def _normalized_set(self):
        """
        Modified 'powerload_set' attribute, in which all times are
        normalized in respect to the 'duration' of the 'phase'
        attribute.
        """
        normalized_set = copy.deepcopy(self.powerload_set)
        zip_of_set_and_flag = zip(normalized_set, self.normalize)
        for powerload, normalization_flag in zip_of_set_and_flag:
            if normalization_flag:
                powerload._normalize_time(self.phase.duration)
        return normalized_set

    @_normalized_set.setter
    def _normalized_set(self, value) -> None:
        raise PhaseLoadError("normalized_set")

    def _curve(self, time, primary=False):
        """
        If primary, build curve in respect to 'normalized_set'.
        If secondary (called from 'PulseLoad'), plot in respect to
        'powerload_set', since set will already have been normalized.
        """
        load_set = self._normalized_set if primary else self.powerload_set
        resulting_load = sum(load_set)
        return resulting_load.curve(time)

    def curve(self, time):
        """
        Create a curve by calculating power load values at the specified
        times.

        This method applies the 'curve' method of the 'PowerLoad' class
        to the 'PowerLoad' instance that is created by the sum of all
        'PowerLoad' objects stored in the 'powerload_set' attribute.

        Parameters
        ----------
        time: int | float | list[ int | float ]
            List of time values. [s]

        Returns
        -------
        curve: list[float]
            List of power values. [W]
        """
        return self._curve(time, primary=True)

    def make_consumption_explicit(self):
        """
        Calls 'make_consumption_explicit' on every element of the
        'powerload_set' attribute.
        """
        self._recursive_make_consumption_explicit(self.powerload_set)

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    @property
    def intrinsic_time(self):
        """
        Single time vector that contains all values used to define the
        different 'PowerLoad' objects contained in the 'powerload_set'
        attribute (i.e. all times are their original values).
        """
        return self._build_time_from_load_set(self.powerload_set)

    @intrinsic_time.setter
    def intrinsic_time(self, value) -> None:
        raise PhaseLoadError(
            "time",
            "The 'intrinsic_time' is instead built from the 'time' "
            "attributes of the 'PowerLoad' objects stored in "
            "the 'powerload_set' attribute.",
        )

    @property
    def normalized_time(self):
        """
        Single time vector that contains all values used to define the
        different 'PowerLoad' objects contained in the '_normalized_set'
        attribute (i.e. all times are normalized in respect to the phase
        duration).
        """
        return self._build_time_from_load_set(self._normalized_set)

    @normalized_time.setter
    def normalized_time(self, value) -> None:
        raise PhaseLoadError(
            "time",
            "The 'normalized_time' is instead built from the 'time' "
            "attributes of the 'PowerLoad' objects stored in "
            "the 'normalized_set' attribute.",
        )

    def _plot(self, primary=False, ax=None, n_points=None, **kwargs):
        """
        If primary, plot in respect to 'normalized_time'.
        If secondary (called from 'PulseLoad'), plot in respect to
        'intrinsic_time', since set will already have been normalized.
        """
        ax = validate_axes(ax)
        n_points = self._validate_n_points(n_points)

        # Set each default options in kwargs, if not specified
        default_plot_kwargs = self._plot_kwargs
        final_kwargs = {**default_plot_kwargs, **kwargs}

        time_to_plot = self.normalized_time if primary else self.intrinsic_time
        computed_time = self._refine_vector(time_to_plot, n_points)
        computed_curve = self._curve(computed_time, primary=primary)

        list_of_plot_objects = []

        # Plot curve as line
        plot_object = ax.plot(
            computed_time,
            computed_curve,
            label=self.name + CURVE_TEXT_IN_LABEL,
            **final_kwargs,
        )
        list_of_plot_objects.append(plot_object)

        # Add descriptive text next to curve
        plot_object = self._add_text_to_point_in_plot(
            ax,
            self.name,
            computed_time,
            computed_curve,
            **kwargs,
        )
        list_of_plot_objects.append(plot_object)

        return ax, list_of_plot_objects

    def _plot_as_secondary(self, ax=None, n_points=None, **kwargs):
        return self._plot(primary=False, ax=ax, n_points=n_points, **kwargs)

    def plot(self, ax=None, n_points=None, detailed=False, **kwargs):
        """
        Plot a 'PhaseLoad' curve, built using the 'powerload_set' and
        'normalize' attributes that define the instance. The number of
        points interpolated in each curve segment can be specified.

        This method applies the 'plot' method of the 'PowerLoad' class
        to the resulting load created by the 'curve' method.

        This method can also plot the individual 'PowerLoad' objects
        stored in the 'powerload_set' attribute.

        Parameters
        ----------
        n_points: int
            Number of points interpolated in each curve segment. The
            default value is 'None', which indicates to the method
            that the default value should be used, defined as a class
            attribute.
        detailed: bool
            Determines whether the plot will include all individual
            'PowerLoad' objects, that summed result in the normal
            plotted curve. These objects are plotted as secondary plots,
            as defined in 'PowerCycleABC' class. By default this input
            is set to 'False'.
        **kwargs: dict
            Options for the 'plot' method.

        Returns
        -------
        ax: Axes
            Instance of the 'matplotlib.axes.Axes' class.
        list_of_plot_objects: list
            List of plot objects created by the 'matplotlib' package.
            The first element of the list is the plot object created
            using the 'pyplot.plot', while the second element of the
            list is the plot object created using the 'pyplot.text'
            method.
            If the 'detailed' argument is set to 'True', the list
            continues to include the lists of plot objects created by
            the 'PowerLoad' class.
        """
        ax, list_of_plot_objects = self._plot(
            primary=True,
            ax=ax,
            n_points=n_points,
            **kwargs,
        )

        if detailed:
            for normal_load in self._normalized_set:
                normal_load._make_secondary_in_plot()
                ax, plot_list = normal_load.plot(ax=ax)
                list_of_plot_objects.append(plot_list)

        return ax, list_of_plot_objects

    # ------------------------------------------------------------------
    # ARITHMETICS
    # ------------------------------------------------------------------
    def __add__(self, other):
        """
        The addition of 'PhaseLoad' instances creates a new 'PhaseLoad'
        instance with joined 'powerload_set' and 'normalize' attributes,
        but only if its phases are the same.
        """
        this = copy.deepcopy(self)
        other = copy.deepcopy(other)

        if this.phase != other.phase:
            raise PhaseLoadError(
                "addition",
                "The phases of this PhaseLoad addition represent "
                f"{this.phase.name!r} and {other.phase.name!r} "
                "respectively.",
            )

        another_name = f"Resulting PhaseLoad for phase {this.phase.name!r}"
        return PhaseLoad(
            another_name,
            this.phase,
            this.powerload_set + other.powerload_set,
            this.normalize + other.normalize,
        )


class PulseLoad(PowerCycleLoadABC):
    """
    Generic representation of the total power load during a pulse.

    Defines the pulse load with a set of 'PhaseLoad' instances. The list
    of 'PhaseLoad' objects given as a parameter to be stored in the
    'phaseload_set' attribute must be provided in the order they are
    expected to occur during a pulse. This ensures that the correct
    'PowerCyclePulse' object is created and stored in the 'pulse'
    attribute. This enables the instance to shift power loads in time
    accordingly.

    Time shifts of each phase load in the 'phaseload_set' occurs AFTER
    the normalization performed by each 'PhaseLoad' object. In short,
    'PulseLoad' curves are built by joining each individual 'PhaseLoad'
    curve after performing the following the manipulations in the order
    presented below:
        1) normalization, given the 'normalize' attribute of each
            'PhaseLoad', with normalization equal to the 'duration'
            of its 'phase' attribute;
        2) shift, given the sum of the 'duration' attributes of
            the 'phase' of each 'PhaseLoad' that comes before it.

    Parameters
    ----------
    name: str
        Description of the 'PulseLoad' instance.
    pulse: PowerCyclePulse
        Pulse specification, that determines the necessary phases to
        be characterized by 'PhaseLoad' objects.
    phaseload_set: PhaseLoad | list[PhaseLoad]
        Collection of 'PhaseLoad' objects that define the 'PulseLoad'
        instance. Upon initialization, phase loads are permuted to have
        the same order as the phases in the 'phase_set' attribute of
        the 'pulse' parameter. Missing cases are treated with the
        creation of an null phase load.

    Properties
    ----------
    intrinsic_time: list[int | float]
        List that contains all values in the 'intrinsic_time' properties
        of the different 'PhaseLoad' objects contained in the
        'phaseload_set' attribute, ordered and with no repetitions.
    shifted_time: list[int | float]
        List that contains all values in the 'intrinsic_time' properties
        of the different 'PhaseLoad' objects contained in the
        '_shifted_set' attribute, ordered and with no repetitions.
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    # Override number of points
    _n_points = 100

    # Override pyplot defaults
    _plot_defaults = {
        "c": "k",  # Line color
        "lw": 2,  # Line width
        "ls": "-",  # Line style
    }

    # Defaults for detailed plots
    _detailed_defaults = {
        "c": "k",  # Line color
        "lw": 1,  # Line width
        "ls": "--",  # Line style
    }

    # Defaults for delimiter plots
    _delimiter_defaults = {
        "c": "darkorange",  # Line color
    }

    # Minimal shift for time correction in 'curve' method
    epsilon = 1e6 * sys.float_info.epsilon

    def __init__(self, name, pulse, phaseload_set):
        super().__init__(name)

        pulse = self._validate_pulse(pulse)
        phaseload_set = self._validate_phaseload_set(phaseload_set, pulse)

        self.pulse = pulse
        self.phaseload_set = phaseload_set

    @staticmethod
    def _validate_pulse(pulse):
        PowerCyclePulse.validate_class(pulse)
        return pulse

    @staticmethod
    def _validate_phaseload_set(phaseload_set, pulse):
        """
        Validate 'phaseload_set' input to be a list of 'PhaseLoad'
        instances. Multiple phase loads for the same phase are added.
        Mssing phase loads are filled with a null instance.
        """
        phaseload_set = validate_list(phaseload_set)

        validated_phaseload_set = []
        phase_library = pulse.build_phase_library()
        for phase_in_pulse in phase_library.values():
            phaseloads_for_phase = []
            for phaseload in phaseload_set:
                PhaseLoad.validate_class(phaseload)

                phase_of_phaseload = phaseload.phase
                if phase_of_phaseload == phase_in_pulse:
                    phaseloads_for_phase.append(phaseload)

            no_phaseloads_were_added = len(phaseloads_for_phase) == 0
            if no_phaseloads_were_added:
                null_phaseload = PhaseLoad.null(phase_in_pulse)
                phaseloads_for_phase = null_phaseload

            phaseloads_for_phase = validate_list(phaseloads_for_phase)

            single_phaseload = sum(phaseloads_for_phase)
            validated_phaseload_set.append(single_phaseload)

        return validated_phaseload_set

    @classmethod
    def null(cls, pulse):
        """
        Instantiates an null version of the class.
        """
        name = "Null PhaseLoad for pulse " + pulse.name
        library = pulse.build_phase_library()
        phaseload_set = [PhaseLoad.null(phase) for phase in library.values()]
        null_instance = cls(name, pulse, phaseload_set)
        return null_instance

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    def _build_pulse_from_phaseload_set(self):
        """
        Build pulse from 'PowerCyclePhase' objects stored in the
        'phase' attributes of each 'PhaseLoad' instance in the
        'phaseload_set' list.
        """
        name = "Pulse for " + self.name
        phase_set = [phaseload.phase for phaseload in self.phaseload_set]
        pulse = PowerCyclePulse(name, phase_set)
        return pulse

    @property
    def _shifted_set(self):
        """
        Modified 'phaseload_set' attribute, in which all times of each
        'PhaseLoad' object are shifted by the sum of 'duration' values
        of the 'phase' attribute of each 'PhaseLoad' that comes before
        it in the 'phaseload_set' attribute.
        Shifts are applied to the '_normalized_set' property of each
        'PhaseLoad' object.
        """
        phaseload_set = copy.deepcopy(self.phaseload_set)

        time_shift = 0
        shifted_set = []
        for phaseload in phaseload_set:
            normalized_set = phaseload._normalized_set
            for normal_load in normalized_set:
                normal_load._shift_time(time_shift)
            phaseload.powerload_set = normalized_set

            shifted_set.append(phaseload)
            time_shift += phaseload.phase.duration

        return shifted_set

    @_shifted_set.setter
    def _shifted_set(self, value) -> None:
        raise PulseLoadError("shifted_set")

    def curve(self, time):
        """
        Create a curve by calculating phase load values at the specified
        times.

        This method applies the 'curve' method of the 'PhaseLoad' class
        to each object stored in the '_shifted_set' attribute, and
        returns the sum of all individual curves created.

        The last point of each 'PhaseLoad' curve is shifted by a minimal
        time 'epsilon', defined as a class attribute, to avoid an
        overlap with the first point of the curve in the following phase
        and a super-position of loads at that point.

        Parameters
        ----------
        time: int | float | list[ int | float ]
            List of time values. [s]

        Returns
        -------
        curve: list[float]
            List of power values. [W]
        """
        shifted_set = self._shifted_set

        curve = []
        modified_time = []
        for shifted_load in shifted_set:
            intrinsic_time = shifted_load.intrinsic_time

            max_t = max(intrinsic_time)
            min_t = min(intrinsic_time)
            load_time = [t for t in time if (min_t <= t) and (t <= max_t)]

            load_time[-1] = load_time[-1] - self.epsilon
            load_curve = shifted_load._curve(load_time, primary=False)

            modified_time.append(load_time)
            curve.append(load_curve)

        modified_time = unnest_list(modified_time)
        curve = unnest_list(curve)

        return modified_time, curve

    def make_consumption_explicit(self):
        """
        Calls 'make_consumption_explicit' on every element of the
        'phaseload_set' attribute.
        """
        self._recursive_make_consumption_explicit(self.phaseload_set)

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    @property
    def intrinsic_time(self):
        """
        Single time vector that contains all values used to define the
        different 'PhaseLoad' objects contained in the 'phaseload_set'
        attribute (i.e. all times are their original values).
        """
        return self._build_time_from_load_set(self.phaseload_set)

    @intrinsic_time.setter
    def intrinsic_time(self, value) -> None:
        raise PulseLoadError(
            "time",
            "The 'intrinsic_time' is instead built from the "
            "'intrinsic_time' attributes of the 'PhaseLoad' "
            "objects stored in the 'phaseload_set' attribute.",
        )

    @property
    def shifted_time(self):
        """
        Single time vector that contains all values used to define the
        different 'PowerLoad' objects contained in the '_shifted_set'
        attribute (i.e. all times are shifted in respect to the
        duration of previous phases).
        """
        time = self._build_time_from_load_set(self._shifted_set)
        return time

    @shifted_time.setter
    def shifted_time(self, value) -> None:
        raise PulseLoadError(
            "time",
            "The 'shifted_time' is instead built from the "
            "'intrinsic_time' attributes of the 'PowerLoad' "
            "objects stored in the 'shifted_set' attribute.",
        )

    def _plot_phase_delimiters(self, ax=None):
        """
        Add vertical lines to plot to specify where the phases of a
        pulse end.
        """
        ax = validate_axes(ax)
        axis_limits = ax.get_ylim()

        shifted_set = self._shifted_set

        default_delimiter_kwargs = self._delimiter_defaults
        delimiter_kwargs = default_delimiter_kwargs

        default_line_kwargs = self._detailed_defaults
        line_kwargs = {**default_line_kwargs, **delimiter_kwargs}

        default_text_kwargs = self._text_kwargs
        text_kwargs = {**default_text_kwargs, **delimiter_kwargs}

        list_of_plot_objects = []
        for shifted_load in shifted_set:
            intrinsic_time = shifted_load.intrinsic_time
            last_time = intrinsic_time[-1]

            label = "Phase delimiter for " + shifted_load.phase.name

            plot_object = ax.plot(
                [last_time, last_time],
                axis_limits,
                label=label,
                **line_kwargs,
            )
            list_of_plot_objects.append(plot_object)

            plot_object = ax.text(
                last_time,
                axis_limits[-1],
                "End of " + shifted_load.phase.name,
                label=label,
                **text_kwargs,
            )
            list_of_plot_objects.append(plot_object)

        return ax, list_of_plot_objects

    def plot(self, ax=None, n_points=None, detailed=False, **kwargs):
        """
        Plot a 'PulseLoad' curve, built using the attributes that define
        the instance. The number of points interpolated in each curve
        segment can be specified.

        This method applies the 'plot' method of the 'PowerLoad' class
        to the resulting load created by the 'curve' method.

        This method can also plot the individual 'PowerLoad' objects
        stored in each 'PhaseLoad' instance of the 'phaseload_set'
        attribute.

        Parameters
        ----------
        ax: Axes
            Instance of the 'matplotlib.axes.Axes' class, in which to
            plot. If 'None' is given, a new instance of axes is created.
        n_points: int
            Number of points interpolated in each curve segment. The
            default value is 'None', which indicates to the method
            that the default value should be used, defined as a class
            attribute.
        detailed: bool
            Determines whether the plot will include all individual
            'PhaseLoad' instances, that summed result in the normal
            plotted curve. Plotted as secondary plots, as defined in
            'PowerCycleABC' class. By default this input is set to
            'False'.
        **kwargs: dict
            Options for the 'plot' method.

        Returns
        -------
        ax: Axes
            Instance of the 'matplotlib.axes.Axes' class.
        list_of_plot_objects: list
            List of plot objects created by the 'matplotlib' package.
            The first element of the list is the plot object created
            using the 'pyplot.plot', while the second element of the
            list is the plot object created using the 'pyplot.text'
            method.
            If the 'detailed' argument is set to 'True', the list
            continues to include the lists of plot objects created by
            the 'PowerLoad' class.
        """
        ax = validate_axes(ax)
        n_points = self._validate_n_points(n_points)

        # Set each default options in kwargs, if not specified
        default_plot_kwargs = self._plot_kwargs
        final_kwargs = {**default_plot_kwargs, **kwargs}

        time_to_plot = self.shifted_time
        computed_time = self._refine_vector(time_to_plot, n_points)
        modified_time, computed_curve = self.curve(computed_time)

        list_of_plot_objects = []

        # Plot curve as line
        plot_object = ax.plot(
            modified_time,
            computed_curve,
            label=self.name + CURVE_TEXT_IN_LABEL,
            **final_kwargs,
        )
        list_of_plot_objects.append(plot_object)

        # Add descriptive text next to curve
        text_object = self._add_text_to_point_in_plot(
            ax,
            self.name,
            modified_time,
            computed_curve,
            **kwargs,
        )
        list_of_plot_objects.append(text_object)

        if detailed:
            for shifted_load in self._shifted_set:
                shifted_load._make_secondary_in_plot()
                ax, plot_list = shifted_load._plot_as_secondary(ax=ax)
                list_of_plot_objects.append(plot_list)

            # Add phase delimiters
            ax, delimiter_list = self._plot_phase_delimiters(ax=ax)
            list_of_plot_objects = list_of_plot_objects + delimiter_list

        return ax, list_of_plot_objects

    # ------------------------------------------------------------------
    # ARITHMETICS
    # ------------------------------------------------------------------
    def __add__(self, other):
        """
        The addition of 'PulseLoad' instances can only be performed if
        their pulses are equal. It returns a new 'PulseLoad' instance
        with a 'phaseload_set' that contains the addition of the
        respective 'PhaseLoad' objects in each original instance.
        """
        this = copy.deepcopy(self)
        other = copy.deepcopy(other)

        if this.pulse != other.pulse:
            raise PhaseLoadError(
                "addition",
                "The pulses of this PulseLoad addition represent "
                f"{this.pulse.name!r} and {other.pulse.name!r} "
                "respectively.",
            )

        another_name = f"Resulting PulseLoad for pulse {this.pulse.name!r}"
        return PulseLoad(
            another_name,
            this.pulse,
            this.phaseload_set + other.phaseload_set,
        )


class ScenarioLoad(PowerCycleLoadABC):
    """
    Generic representation of the total power load during a scenario.

    Defines the phase load with a set of 'PulseLoad' instances. Each
    instance must be accompanied by a 'repetition' specification, used
    to indicate how many times that pulse load is repeated in the
    scenario before a new set of pulse loads starts. This enables the
    instance to adjust the evolution of pulse loads accordingly, if
    changes occur to the plant scenario.

    Parameters
    ----------
    name: str
        Description of the 'ScenarioLoad' instance.
    scenario: 'PowerCycleScenario'
        Scenario specification, that determines the necessary pulses to
        be characterized by 'PulseLoad' objects.
    pulseload_set: PulseLoad | list[PulseLoad]
        Collection of instances of the 'PulseLoad' class that define
        the 'ScenarioLoad' object.

    Attributes
    ----------
    scenario: PowerCycleScenario
        Scenario specification, determined by the 'pulse' attributes of
        the 'PulseLoad' instances used to define the 'ScenarioLoad'.

    Properties
    ----------
    intrinsic_time: list[int | float]
        List that contains all values in the 'intrinsic_time' properties
        of the different 'PulseLoad' objects contained in the
        'pulseload_set' attribute, ordered and with no repetitions.
    timeline_time: list[int | float]
        List that contains all values in the 'intrinsic_time' properties
        of the different 'PulseLoad' objects contained in the
        '_timeline_set' attribute, ordered and with no repetitions.
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    @property
    def intrinsic_time(self):
        """
        Single time vector that contains all values used to define the
        different 'PulseLoad' objects contained in the 'pulseload_set'
        attribute (i.e. all times are their original values).
        """
        """
        time = self._build_time_from_load_set(self.pulseload_set)
        return time
        """
        pass

    @intrinsic_time.setter
    def intrinsic_time(self, value) -> None:
        pass

    @property
    def timeline_time(self):
        """
        Single time vector that contains all values used to define the
        different 'PowerLoad' objects contained in the '_timeline_set'
        attribute (i.e. all times are shifted in respect to the
        duration of previous pulses).
        """
        """
        time = self._build_time_from_load_set(self._timeline_set)
        return time
        """
        pass

    # ------------------------------------------------------------------
    # ARITHMETICS
    # ------------------------------------------------------------------

    pass