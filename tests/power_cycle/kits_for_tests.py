# COPYRIGHT PLACEHOLDER

import copy
import os

import matplotlib.pyplot as plt
import numpy as np

import bluemira.base.constants as constants
from bluemira.power_cycle.net.loads import LoadData, LoadModel, PhaseLoad, PowerLoad
from bluemira.power_cycle.time import PowerCyclePhase, PowerCyclePulse, ScenarioBuilder
from bluemira.power_cycle.tools import (
    FormattedABC,
    read_json,
    unnest_list,
    validate_axes,
    validate_file,
    validate_nonnegative,
)

test_data_folder_path = (
    "tests",
    "power_cycle",
    "test_data",
)


def assert_value_is_nonnegative(argument):
    possible_errors = (TypeError, ValueError)
    try:
        validate_nonnegative(argument)
    except possible_errors:
        assert False
    else:
        assert True


def copy_dict_with_wrong_key(right_dict, key_to_substitute):
    """
    Make deep copy of dictionary, but substitute one key
    by the 'wrong_key' string.
    """
    wrong_dict = copy.deepcopy(right_dict)
    wrong_dict["wrong_key"] = wrong_dict.pop(key_to_substitute)
    return wrong_dict


def copy_dict_with_wrong_value(right_dict, key, value_to_substitute):
    """
    Make deep copy of dictionary, but substitute the value in 'key'
    by 'value_to_substitute'.
    """
    wrong_dict = copy.deepcopy(right_dict)
    wrong_dict[key] = value_to_substitute
    return wrong_dict


class ToolsTestKit:
    def __init__(self):
        test_file_name = tuple(["test_file.txt"])
        test_file_path = test_data_folder_path + test_file_name
        self.test_file_path = os.path.join(*test_file_path)

    @staticmethod
    def build_list_of_example_arguments():
        example_arguments = [
            None,
            True,  # bool
            "some string",  # string
            70,  # int
            -70,  # negative int
            1.2,  # float
            -1.2,  # negative float
            [True, False],  # bool list
            (True, False),  # bool tuple
            ["1", "2", "3", "4"],  # str list
            ("1", "2", "3", "4"),  # str tuple
            [1, 2, 3, 4],  # int list
            [-1, -2, -3, -4],  # negative int list
            (1, 2, 3, 4),  # int tuple
            (-1, -2, -3, -4),  # negative int tuple
            [1.2, 2.2, 3.2, 4.2],  # float list
            [-1.2, -2.2, -3.2, -4.2],  # negative float list
            (1.2, 2.2, 3.2, 4.2),  # float tuple
            (-1.2, -2.2, -3.2, -4.2),  # negative float tuple
        ]
        return example_arguments

    @staticmethod
    def prepare_figure(figure_title, humor=False):
        """
        Create figure for plot testing. Use 'plt.show()' to display it.
        Run test file with with `pytest --plotting-on` to visualize it.
        Switch 'humor' on for 'xkcd' style plots.
        """
        if humor:
            with plt.xkcd():
                ax = validate_axes()
        else:
            ax = validate_axes()
        plt.grid()
        plt.title(figure_title)
        return ax

    def build_dictionary_examples(self):
        argument_examples = self.build_list_of_example_arguments()

        count = 0
        format_example = dict()
        dictionary_example = dict()
        for argument in argument_examples:
            argument_type = type(argument)

            count += 1
            current_key = "key " + str(count)

            format_example[current_key] = argument_type
            dictionary_example[current_key] = argument

        subdictionaries_example = dict()
        for c in range(count):
            current_key = "key " + str(c)
            subdictionaries_example[current_key] = dictionary_example

        return format_example, dictionary_example, subdictionaries_example

    def inputs_for_format(self):
        """
        Function to create inputs for Format testing.
        """
        right_format_input = {
            "key_1": str,
            "key_2": list,
            "key_3": [int, float],
            "key_4": [None, bool],
            "key_5": dict,
        }
        wrong_format_input = {
            "key_1": "str",
            "key_2": [[int, float], [str, bool]],
            "key_3": 1.2,
            "key_4": True,
            "key_5": copy.deepcopy(right_format_input),
        }
        return (
            right_format_input,
            wrong_format_input,
        )

    def _format_index_inputs(self):
        right_index_input = [
            0,
            0,
            1,
            1,
            0,
        ]

        non_int = 1.2
        out_of_range = 10
        wrong_index_input = [
            0,
            0,
            non_int,
            out_of_range,
            0,
        ]

        return (
            right_index_input,
            wrong_index_input,
        )

    def inputs_for_formatteddict(self):
        """
        Function to create inputs for FormattedDict testing.
        """

        (
            right_format_input,
            wrong_format_input,
        ) = self.inputs_for_format()
        example_format = FormattedABC.Format(right_format_input)
        example_dict = wrong_format_input

        (
            right_index_input,
            wrong_index_input,
        ) = self._format_index_inputs()

        return (
            example_format,
            example_dict,
            right_index_input,
            wrong_index_input,
        )

    def inputs_for_formattedlibrary(self):
        """
        Function to create inputs for FormattedLibrary testing.
        """
        (
            example_format,
            example_dict,
            _,
            _,
        ) = self.inputs_for_formatteddict()

        example_types = [
            str,
            list,
            int,
            float,
            bool,
            dict,
        ]

        example_keys_for_library = [
            "case_1",
            "case_2",
            "case_3",
            "case_4",
        ]

        return (
            example_format,
            example_types,
            example_dict,
            example_keys_for_library,
        )


class TimeTestKit:
    def __init__(self):
        scenario_json_name = tuple(["scenario_config.json"])
        scenario_json_path = test_data_folder_path + scenario_json_name
        self.scenario_json_path = os.path.join(*scenario_json_path)

    def inputs_for_phase(self):
        """
        Function to create inputs for PowerCyclePhase testing.
        The lists 'input_names' and 'input_breakdowns' must have the
        same length.
        """
        input_names = [
            "Dwell",
            "Transition between dwell and flat-top",
            "Flat-top",
            "Transition between flat-top and dwell",
        ]
        input_labels = [
            "dwl",
            "d2f",
            "ftt",
            "f2d",
        ]
        input_breakdowns = [
            {
                "CS-recharge + pumping": constants.raw_uc(10, "minute", "second"),
            },
            {
                "current ramp-up": 157,
                "heating": 19,
            },
            {
                "plasma burn": constants.raw_uc(2, "hour", "second"),
            },
            {
                "cooling": 123,
                "current ramp-down": 157,
            },
        ]
        assert len(input_names) == len(input_breakdowns)
        n_inputs = len(input_names)

        return (
            n_inputs,
            input_names,
            input_breakdowns,
            input_labels,
        )

    def inputs_for_pulse(self):
        """
        Function to create inputs for PowerCyclePulse testing.
        """
        (
            n_inputs,
            input_names,
            input_breakdowns,
            input_labels,
        ) = self.inputs_for_phase()

        input_phases = []
        for i in range(n_inputs):
            name = input_names[i]
            breakdown = input_breakdowns[i]
            label = input_labels[i]
            phase = PowerCyclePhase(name, breakdown, label=label)
            input_phases.append(phase)

        return (
            n_inputs,
            input_phases,
        )

    def inputs_for_scenario(self):
        """
        Function to create inputs for PowerCycleScenario testing.
        """
        (
            _,
            input_phases,
        ) = self.inputs_for_pulse()

        n_pulses = 10

        input_pulses = []
        for p in range(n_pulses):
            name = "Pulse " + str(p)
            phase = input_phases
            pulse = PowerCyclePulse(name, phase)
            input_pulses.append(pulse)

        return (
            n_pulses,
            input_pulses,
        )

    def inputs_for_builder(self):
        scenario_json_path = self.scenario_json_path
        scenario_json_contents = read_json(scenario_json_path)

        return scenario_json_contents


class NetLoadsTestKit:
    time_testkit = TimeTestKit()

    def __init__(self):
        all_colors = [
            "r",
            "b",
            "g",
            "m",
            "c",
            "y",
        ]
        self.color_order_for_plotting = all_colors
        self.n_colors = len(all_colors)

        self.comparison_relative_tolerance = 1e-6
        self.comparison_absolute_tolerance = 1e-12

        manipulation_examples = [10 ** (e - 5) for e in range(10)]
        self.attribute_manipulation_examples = manipulation_examples

    @staticmethod
    def assert_is_interpolation(original_points, curve):
        """
        Confirm that curve is an interpolation with possibility of
        out-of-bounds values.

        Current approach: no curve value is out of the bounds of the
        original defining interpolation points, except if it is a zero
        ('fill_value' argument of 'interp1d').

        Possibly to be substituted by 'unittest.mock'.
        """
        original_max = max(original_points)
        original_min = min(original_points)
        curve_max = max(curve)
        curve_min = min(curve)
        assert (curve_max <= original_max) or (curve_max == 0)
        assert (curve_min >= original_min) or (curve_min == 0)

    def inputs_for_loaddata(self):
        """
        Function to create inputs for LoadData testing.
        The lists 'input_times' and 'input_datas' must have the same
        length.
        """

        input_times = [
            [0, 4, 7, 8],
            [2, 5, 7, 9, 10],
        ]
        input_datas = [
            [6, 9, 7, 8],
            [2, 2, 2, 4, 4],
        ]
        assert len(input_times) == len(input_datas)
        n_inputs = len(input_times)

        input_names = []
        for i in range(n_inputs):
            input_names.append("LoadData " + str(i))

        return (
            n_inputs,
            input_names,
            input_times,
            input_datas,
        )

    def inputs_for_time_interpolation(self):
        """
        Function to create inputs time interpolation testing.
        """
        (
            _,
            _,
            input_times,
            _,
        ) = self.inputs_for_loaddata()

        all_times = unnest_list(input_times)
        minimum_time = min(all_times)
        maximum_time = min(all_times)

        time_extrapolation = 2
        time_step = 0.1

        start = minimum_time - time_extrapolation
        stop = maximum_time + time_extrapolation
        time_vector = np.arange(start, stop, time_step)
        time_list = list(time_vector)
        return time_list

    def inputs_for_powerload(self):
        """
        Function to create inputs for PowerLoad testing, based on the
        function that creates inputs for LoadData testing.
        """
        (
            n_inputs,
            input_names,
            input_times,
            input_datas,
        ) = self.inputs_for_loaddata()

        all_loadmodels = [member.name for member in LoadModel]
        n_loadmodels = len(all_loadmodels)

        input_loadmodels = []
        input_loaddatas = []
        for i in range(n_inputs):
            # Cycle through available loadmodels to create an example
            loadmodel = all_loadmodels[i % n_loadmodels]
            loadmodel = LoadModel[loadmodel]
            input_loadmodels.append(loadmodel)

            name = input_names[i]
            time = input_times[i]
            data = input_datas[i]

            loaddata = LoadData(name, time, data)
            input_loaddatas.append(loaddata)

        old = "LoadData"
        new = "PowerLoad"
        input_names = [name.replace(old, new) for name in input_names]
        return (
            n_inputs,
            input_names,
            input_loaddatas,
            input_loadmodels,
        )

    def inputs_for_phaseload(self):
        """
        Function to create inputs for PhaseLoad testing, based on the
        function that creates inputs for PowerLoad testing.
        """
        (
            n_inputs,
            input_names,
            input_loaddatas,
            input_loadmodels,
        ) = self.inputs_for_powerload()

        (
            n_phases,
            all_phases,
        ) = self.time_testkit.inputs_for_pulse()

        input_phases = []
        input_powerloads = []
        input_normalflags = []
        for i in range(n_inputs):
            # Cycle through phases to pick a phase example
            phase = all_phases[i % n_phases]
            input_phases.append(phase)

            name = input_names[i]
            loaddata = input_loaddatas[i]
            loadmodel = input_loadmodels[i]
            powerload = PowerLoad(name, loaddata, loadmodel)
            input_powerloads.append(powerload)

            # Cycle through True/False to create a flag example
            normal_flag = bool(i % 2)
            input_normalflags.append(normal_flag)

        input_names = [name.replace("Power", "Phase") for name in input_names]
        return (
            n_inputs,
            input_names,
            input_phases,
            input_powerloads,
            input_normalflags,
        )

    def inputs_for_pulseload(self):
        """
        Function to create inputs for PulseLoad testing, based on the
        function that creates inputs for PhaseLoad testing.
        """
        (
            n_inputs,
            input_phases,
        ) = self.time_testkit.inputs_for_pulse()

        (
            _,
            _,
            _,
            input_powerloads,
            input_normalflags,
        ) = self.inputs_for_phaseload()

        input_names = []
        input_phaseloads = []
        for i in range(n_inputs):
            phase = input_phases[i]
            name = "PhaseLoad for " + phase.name + " (phase)"
            powerloads = input_powerloads
            normalflags = input_normalflags
            phaseload = PhaseLoad(name, phase, powerloads, normalflags)
            input_phaseloads.append(phaseload)

            name = "PulseLoad" + str(i)
            input_names.append(name)

        return (
            n_inputs,
            input_names,
            input_phaseloads,
        )


class NetImportersTestKit:
    @staticmethod
    def equilibria_duration_inputs():
        possible_inputs = {
            "desired_data": [
                "CS-recharge-time",
                "ramp-up-time",
                "ramp-down-time",
            ]
        }
        return possible_inputs

    @staticmethod
    def equilibria_phaseload_inputs():
        possible_inputs = {
            "desired_data": [
                "CS-coils",
                "TF-coils",
                "PF-coils",
            ]
        }
        return possible_inputs

    @staticmethod
    def pumping_duration_inputs():
        possible_inputs = {
            "desired_data": [
                "pumpdown-time",
            ]
        }
        return possible_inputs


class NetManagerTestKit:
    time_testkit = TimeTestKit()

    def __init__(self):
        time_testkit = self.time_testkit
        scenario_json_path = time_testkit.scenario_json_path
        self.scenario_json_path = scenario_json_path

        scenario_builder = ScenarioBuilder(scenario_json_path)
        scenario = scenario_builder.scenario
        self.scenario = scenario

        manager_json_name = tuple(["manager_config.json"])
        manager_json_path = test_data_folder_path + manager_json_name
        self.manager_json_path = os.path.join(*manager_json_path)

    def inputs_for_manager(self):
        manager_json_path = self.manager_json_path
        manager_json_contents = read_json(manager_json_path)

        return manager_json_contents

    def inputs_for_groups(self):
        manager_json_contents = self.inputs_for_manager()

        all_group_inputs = dict()
        all_group_labels = manager_json_contents.keys()
        for group_label in all_group_labels:
            group_config = manager_json_contents[group_label]
            group_name = group_config["name"]
            group_systems = group_config["systems"]

            config_path = group_config["config_path"]

            skip_missing_file = False
            try:
                config_path = validate_file(config_path)
            except FileNotFoundError:
                skip_missing_file = True

            if skip_missing_file:
                continue
            systems_config = read_json(config_path)

            group_inputs = dict()
            group_inputs["name"] = group_name
            group_inputs["systems_list"] = group_systems
            group_inputs["systems_config"] = systems_config

            all_group_inputs[group_label] = group_inputs
        return all_group_inputs

    def inputs_for_systems(self):
        all_group_inputs = self.inputs_for_groups()

        all_system_inputs = dict()
        all_group_labels = all_group_inputs.keys()
        for group_label in all_group_labels:
            group_config = all_group_inputs[group_label]

            current_systems_config = group_config["systems_config"]
            all_system_inputs = {
                **all_system_inputs,
                **current_systems_config,
            }
        return all_system_inputs

    @staticmethod
    def _copy_dictionary_with_preceding_str_in_all_keys(dictionary, string):
        d = copy.deepcopy(dictionary)
        new_dictionary = {f"{string}{k}": v for k, v in d.items()}
        return new_dictionary

    def inputs_for_loads(self):
        all_system_inputs = self.inputs_for_systems()

        all_load_types = ["reactive", "active"]

        all_load_inputs = dict()
        all_system_labels = all_system_inputs.keys()
        for system_label in all_system_labels:
            system_config = all_system_inputs[system_label]

            for load_type in all_load_types:
                config_for_type = system_config[load_type]
                preeceding_string = load_type + "-"

                inputs = self._copy_dictionary_with_preceding_str_in_all_keys(
                    config_for_type,
                    preeceding_string,
                )

                all_load_inputs = {**all_load_inputs, **inputs}

        return all_load_inputs
