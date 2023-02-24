# COPYRIGHT PLACEHOLDER

import copy

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bluemira.power_cycle.errors import (
    PhaseLoadError,
    PowerCycleABCError,
    PowerDataError,
    PowerLoadError,
)
from bluemira.power_cycle.net_loads import (
    PhaseLoad,
    PowerData,
    PowerLoad,
    PowerLoadModel,
    PulseLoad,
)
from bluemira.power_cycle.time import PowerCyclePhase, PowerCyclePulse
from bluemira.power_cycle.tools import adjust_2d_graph_ranges
from tests.power_cycle.kits_for_tests import NetLoadsTestKit, TimeTestKit, ToolsTestKit

tools_testkit = ToolsTestKit()
time_testkit = TimeTestKit()
netloads_testkit = NetLoadsTestKit()


class TestPowerData:
    def setup_method(self):
        (
            n_samples,
            sample_names,
            sample_times,
            sample_datas,
        ) = netloads_testkit.inputs_for_powerdata()

        all_samples = []
        for s in range(n_samples):
            time = sample_times[s]
            data = sample_datas[s]
            name = sample_names[s]
            sample = PowerData(name, time, data)
            all_samples.append(sample)
        self.all_samples = all_samples

    @pytest.mark.parametrize("test_attr", ["time", "data"])
    def test_is_increasing(self, test_attr):
        all_samples = self.all_samples
        n_samples = len(all_samples)

        def check_if_increasing(test_list):
            check = []
            for i in range(len(test_list) - 1):
                check.append(test_list[i] <= test_list[i + 1])
            return all(check)

        for s in range(n_samples):
            sample = all_samples[s]
            example_list = getattr(sample, test_attr)
            list_is_increasing = check_if_increasing(example_list)
            if list_is_increasing:
                assert sample._is_increasing(example_list)
            else:
                with pytest.raises(PowerDataError):
                    sample._is_increasing(example_list)

    def test_sanity(self):
        all_samples = self.all_samples
        n_samples = len(all_samples)
        for s in range(n_samples):
            sample = all_samples[s]
            name = sample.name
            list_of_given_length = sample.time
            list_shorter = list_of_given_length[:-1]
            list_longer = list_of_given_length + [10]
            all_lists = [
                list_of_given_length,
                list_shorter,
                list_longer,
            ]
            for time in all_lists:
                for data in all_lists:
                    length_time = len(time)
                    length_data = len(data)
                    if length_time == length_data:
                        test_instance = PowerData(name, time, data)
                        assert isinstance(test_instance, PowerData)
                    else:
                        with pytest.raises(PowerDataError):
                            test_instance = PowerData(name, time, data)

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------
    def test_normalize_time(self):
        list_of_new_end_times = netloads_testkit.attribute_manipulation_examples
        rel_tol = None
        abs_tol = None

        all_samples = self.all_samples
        for sample in all_samples:
            sample_copy = copy.deepcopy(sample)

            for new_end_time in list_of_new_end_times:
                old_time = sample_copy.time
                sample_copy._normalize_time(new_end_time)
                new_time = sample_copy.time

                norm = new_time[-1] / old_time[-1]

                old_time_without_0 = [to for to in old_time if to != 0]
                new_time_without_0 = [tn for tn in new_time if tn != 0]

                time_zip = zip(new_time_without_0, old_time_without_0)
                ratios = [tn / to for (tn, to) in time_zip]
                norm_list = [norm] * len(ratios)

                each_ratio_is_norm = []
                for (r, n) in zip(ratios, norm_list):
                    check = r == pytest.approx(n, rel=rel_tol, abs=abs_tol)
                    each_ratio_is_norm.append(check)

                all_ratios_are_norm = all(each_ratio_is_norm)
                assert all_ratios_are_norm

            all_norms = sample_copy._norm
            assert len(all_norms) == len(list_of_new_end_times)

    def test_shift_time(self):
        list_of_time_shifts = netloads_testkit.attribute_manipulation_examples
        rel_tol = None
        abs_tol = None

        all_samples = self.all_samples
        for sample in all_samples:
            sample_copy = copy.deepcopy(sample)

            for time_shift in list_of_time_shifts:
                old_time = sample_copy.time
                sample_copy._shift_time(time_shift)
                new_time = sample_copy.time

                time_zip = zip(old_time, new_time)
                differences = [(tn - to) for (to, tn) in time_zip]

                ts = time_shift
                each_difference_is_ts = []

                for d in differences:
                    check = d == pytest.approx(ts, rel=rel_tol, abs=abs_tol)
                    each_difference_is_ts.append(check)

                all_differences_are_timeshift = all(each_difference_is_ts)
                assert all_differences_are_timeshift

            all_shifts = sample_copy._shift
            assert len(all_shifts) == len(list_of_time_shifts)

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    def test_plot(self):
        figure_title = "PowerData Plotting"
        ax = tools_testkit.prepare_figure(figure_title)

        colors = netloads_testkit.color_order_for_plotting
        n_colors = netloads_testkit.n_colors

        all_samples = self.all_samples
        n_samples = len(all_samples)
        plot_list = []
        for s in range(n_samples):

            # Cycle through available colors
            sample_color = colors[s % n_colors]

            sample = all_samples[s]
            plot_list.append(sample.plot(ax=ax, c=sample_color))
        adjust_2d_graph_ranges(ax=ax)
        plt.show()  # Run with `pytest --plotting-on` to visualize


class TestPowerLoadModel:
    def test_members(self):
        all_names = [member.name for member in PowerLoadModel]
        all_values = [member.value for member in PowerLoadModel]

        for (name, value) in zip(all_names, all_values):
            assert isinstance(name, str)
            assert isinstance(value, str)


class TestPowerLoad:
    def setup_method(self):
        (
            n_samples,
            sample_names,
            sample_powerdatas,
            sample_models,
        ) = netloads_testkit.inputs_for_powerload()

        all_samples = []
        for s in range(n_samples):
            name = sample_names[s]
            powerdata = sample_powerdatas[s]
            model = sample_models[s]
            sample = PowerLoad(name, powerdata, model)
            all_samples.append(sample)
        self.sample_powerdatas = sample_powerdatas
        self.sample_models = sample_models
        self.all_samples = all_samples

    def construct_multisample(self):
        sample_powerdatas = self.sample_powerdatas
        sample_models = self.sample_models

        multisample = PowerLoad(
            "PowerLoad with multiple powerdata & model arguments",
            sample_powerdatas,
            sample_models,
        )
        return multisample

    def test_constructor_with_multiple_arguments(self):
        multisample = self.construct_multisample()
        assert isinstance(multisample, PowerLoad)

    def assert_constructor_fails(
        self,
        test_name,
        test_powerdata,
        test_model,
    ):
        constructor_exceptions = (PowerCycleABCError, PowerLoadError)
        with pytest.raises(constructor_exceptions):
            wrong_sample = PowerLoad(
                test_name,
                test_powerdata,
                test_model,
            )

    def test_validate_powerdata_set(self):
        sample_powerdatas = self.sample_powerdatas
        sample_models = self.sample_models

        right_name = [
            "PowerLoad with a non-PowerData",
            "element in its 'powerdata_set' list argument",
        ]
        right_name = " ".join(right_name)
        right_powerdatas = sample_powerdatas
        right_models = sample_models

        wrong_powerdatas = copy.deepcopy(right_powerdatas)
        wrong_powerdatas[0] = "non-PowerData"
        self.assert_constructor_fails(
            right_name,
            wrong_powerdatas,
            right_models,
        )

    def test_validate_model(self):
        sample_powerdatas = self.sample_powerdatas
        sample_models = self.sample_models

        right_name = [
            "PowerLoad with a non-PowerLoadModel",
            "element in its 'model' list argument",
        ]
        right_name = " ".join(right_name)
        right_powerdatas = sample_powerdatas
        right_models = sample_models

        wrong_models = copy.deepcopy(right_models)
        wrong_models[0] = "non-PowerLoadModel"
        self.assert_constructor_fails(
            right_name,
            right_powerdatas,
            wrong_models,
        )

    def test_sanity(self):
        sample_powerdatas = self.sample_powerdatas
        sample_models = self.sample_models

        right_name = [
            "PowerLoad with different lengths",
            "of 'powerdata_set' & 'model' arguments",
        ]
        right_name = " ".join(right_name)
        right_powerdatas = sample_powerdatas
        right_models = sample_models

        wrong_powerdatas = copy.deepcopy(right_powerdatas)
        wrong_powerdatas.pop()
        self.assert_constructor_fails(
            right_name,
            wrong_powerdatas,
            right_models,
        )

        wrong_models = copy.deepcopy(right_models)
        wrong_models.pop()
        self.assert_constructor_fails(
            right_name,
            right_powerdatas,
            wrong_models,
        )

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    def test_single_curve(self):
        sample_powerdatas = self.sample_powerdatas
        sample_models = self.sample_models

        test_time = netloads_testkit.inputs_for_time_interpolation()
        time_length = len(test_time)

        for powerdata in sample_powerdatas:
            power_points = powerdata.data

            for powerloadmodel in sample_models:
                curve = PowerLoad._single_curve(
                    powerdata,
                    powerloadmodel,
                    test_time,
                )
                curve_length = len(curve)
                assert curve_length == time_length
                netloads_testkit.assert_is_interpolation(power_points, curve)

    def test_validate_time(self):
        test_arguments = [
            None,
            1.2,
            -1.2,
            70,
            -70,
            "some string",
            [1, 2, 3, 4],
            (1, 2, 3, 4),
        ]
        test_arguments = test_arguments + self.all_samples

        for argument in test_arguments:
            if isinstance(argument, (int, float, list)):
                time = PowerLoad._validate_time(argument)
                assert isinstance(time, list)
            else:
                with pytest.raises(PowerLoadError):
                    time = PowerLoad._validate_time(argument)

    def test_curve(self):
        all_samples = self.all_samples
        test_time = netloads_testkit.inputs_for_time_interpolation()
        time_length = len(test_time)

        all_data = []
        all_sets = []
        all_models = []
        for sample in all_samples:
            curve = sample.curve(test_time)
            curve_length = len(curve)
            assert curve_length == time_length

            all_data.append(sample.powerdata_set[0].data)
            all_sets.append(sample.powerdata_set[0])
            all_models.append(sample.model[0])
        multiset_load = PowerLoad("Multi Load", all_sets, all_models)
        multiset_curve = multiset_load.curve(test_time)

        data_maxima = [max(data) for data in all_data]
        data_minima = [min(data) for data in all_data]
        sum_of_maxima = sum(data_maxima)
        sum_of_minima = sum(data_minima)
        extreme_points = [sum_of_minima, sum_of_maxima]
        netloads_testkit.assert_is_interpolation(extreme_points, multiset_curve)

    new_end_time_examples = netloads_testkit.attribute_manipulation_examples

    @pytest.mark.parametrize("new_end_time", new_end_time_examples)
    def test_normalize_time(self, new_end_time):
        all_samples = self.all_samples
        multisample = self.construct_multisample()
        all_samples.append(multisample)
        for sample in all_samples:
            old_powerdata_set = sample.powerdata_set
            sample._normalize_time(new_end_time)
            new_powerdata_set = sample.powerdata_set

            # Assert length of 'powerdata_set' has not changed
            correct_number_of_powerdatas = len(old_powerdata_set)
            assert len(new_powerdata_set) == correct_number_of_powerdatas

            # Assert 'data' has not changed
            for p in range(correct_number_of_powerdatas):
                old_data = old_powerdata_set[p]
                new_data = new_powerdata_set[p]
                assert old_data == new_data

    def test_shift_time(self):
        pass  # No new functionality to be tested.

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    def test_intrinsic_time(self):
        multisample = self.construct_multisample()
        multisample_powerdatas = multisample.powerdata_set

        intrinsic_time = multisample.intrinsic_time()
        for powerdata in multisample_powerdatas:
            time = powerdata.time
            for t in time:
                assert t in intrinsic_time

    @pytest.mark.parametrize("detailed_plot_flag", [False, True])
    def test_plot(self, detailed_plot_flag):
        figure_title = "'detailed' flag = " + str(detailed_plot_flag)
        figure_title = "PowerLoad Plotting (" + figure_title + ")"
        ax = tools_testkit.prepare_figure(figure_title)

        colors = netloads_testkit.color_order_for_plotting
        n_colors = netloads_testkit.n_colors

        list_of_plot_objects = []

        all_samples = self.all_samples
        n_samples = len(all_samples)
        for s in range(n_samples):

            # Cycle through available colors
            sample_color = colors[s % n_colors]

            sample = all_samples[s]
            current_list_of_plot_objects = sample.plot(
                ax=ax, detailed=detailed_plot_flag, c=sample_color
            )
            list_of_plot_objects.append(current_list_of_plot_objects)

        adjust_2d_graph_ranges(ax=ax)
        plt.show()  # Run with `pytest --plotting-on` to visualize

    # ------------------------------------------------------------------
    # ARITHMETICS
    # ------------------------------------------------------------------

    def test_addition(self):
        """
        Tests both '__add__' and '__radd__'.
        """
        figure_title = "PowerLoad Addition"
        ax = tools_testkit.prepare_figure(figure_title)

        all_samples = self.all_samples
        sample_powerdatas = self.sample_powerdatas
        sample_models = self.sample_models

        result = sum(all_samples)  # requires both __add__ and __radd__
        assert isinstance(result, PowerLoad)

        result_powerdata = result.powerdata_set
        assert result_powerdata == sample_powerdatas

        result_powerloadmodel = result.model
        assert result_powerloadmodel == sample_models

        list_of_plot_objects = result.plot(ax=ax, detailed=True, c="b")
        adjust_2d_graph_ranges(ax=ax)
        plt.show()  # Run with `pytest --plotting-on` to visualize

    @pytest.mark.parametrize("number", [2])
    def test_multiplication(self, number):
        """
        Tests both '__mul__' and '__truediv__'.
        """
        figure_title = "PowerLoad Multiplication"
        ax = tools_testkit.prepare_figure(figure_title)
        rel_tol = None
        abs_tol = None

        colors = netloads_testkit.color_order_for_plotting
        n_colors = netloads_testkit.n_colors

        multisample = self.construct_multisample()
        test_time = multisample.intrinsic_time()
        curve = multisample.curve(test_time)

        lesser_multisample = multisample / number
        lesser_multisample.name = "Divided sample"
        lesser_curve = lesser_multisample.curve(test_time)

        greater_multisample = multisample * number
        greater_multisample.name = "Multiplied sample"
        greater_curve = greater_multisample.curve(test_time)

        length_time = len(test_time)
        for t in range(length_time):
            curve_point = curve[t]
            lesser_point = lesser_curve[t]
            greater_point = greater_curve[t]

            n = number
            cp = curve_point
            lp = lesser_point
            gp = greater_point

            assert lp == pytest.approx(cp / n, rel=rel_tol, abs=abs_tol)
            assert gp == pytest.approx(cp * n, rel=rel_tol, abs=abs_tol)

        samples_to_plot = [
            multisample,
            lesser_multisample,
            greater_multisample,
        ]
        n_samples = len(samples_to_plot)
        list_of_plot_objects = []
        for s in range(n_samples):
            sample = samples_to_plot[s]
            assert isinstance(sample, PowerLoad)

            # Cycle through available colors
            sample_color = colors[s % n_colors]

            sample_plot_list = sample.plot(
                ax=ax,
                detailed=False,
                c=sample_color,
            )
            list_of_plot_objects.append(sample_plot_list)
        adjust_2d_graph_ranges(ax=ax)
        plt.show()  # Run with `pytest --plotting-on` to visualize


class TestPhaseLoad:
    def setup_method(self):
        (
            n_samples,
            sample_names,
            sample_phases,
            sample_powerloads,
            sample_normalflags,
        ) = netloads_testkit.inputs_for_phaseload()

        all_samples = []
        for s in range(n_samples):
            name = sample_names[s]
            phase = sample_phases[s]
            powerload = sample_powerloads[s]
            normalflag = sample_normalflags[s]
            sample = PhaseLoad(name, phase, powerload, normalflag)
            all_samples.append(sample)
        self.sample_phases = sample_phases
        self.sample_powerloads = sample_powerloads
        self.sample_normalflags = sample_normalflags
        self.all_samples = all_samples

    def construct_multisample(self):
        sample_phases = self.sample_phases
        sample_powerloads = self.sample_powerloads
        sample_normalflags = self.sample_normalflags

        name = "PhaseLoad with multiple powerload & flag arguments"
        example_phase = sample_phases[0]
        multisample = PhaseLoad(
            name,
            example_phase,
            sample_powerloads,
            sample_normalflags,
        )
        return multisample

    def test_constructor_with_multiple_arguments(self):
        multisample = self.construct_multisample()
        assert isinstance(multisample, PhaseLoad)

        multisample_phase = multisample.phase
        assert isinstance(multisample_phase, PowerCyclePhase)

    def assert_constructor_fails(
        self,
        test_name,
        test_phase,
        test_powerloads,
        test_normalflags,
    ):
        constructor_exceptions = (PowerCycleABCError, PhaseLoadError)
        with pytest.raises(constructor_exceptions):
            wrong_sample = PhaseLoad(
                test_name,
                test_phase,
                test_powerloads,
                test_normalflags,
            )

    def test_validate_phase(self):
        sample_phases = self.sample_phases
        sample_powerloads = self.sample_powerloads
        sample_normalflags = self.sample_normalflags

        right_name = [
            "PhaseLoad with a non-PowerCyclePhase",
            "element in its 'phase' argument",
        ]
        right_name = " ".join(right_name)
        right_phase = sample_phases[0]
        right_phaseloads = sample_powerloads
        right_normalflags = sample_normalflags

        wrong_phase = "non-PowerCyclePhase"
        self.assert_constructor_fails(
            right_name,
            wrong_phase,
            right_phaseloads,
            right_normalflags,
        )

    def test_validate_powerload_set(self):
        sample_phases = self.sample_phases
        sample_powerloads = self.sample_powerloads
        sample_normalflags = self.sample_normalflags

        right_name = [
            "PhaseLoad with a non-PowerLoad",
            "element in its 'powerload_set' list argument",
        ]
        right_name = " ".join(right_name)
        example_phase = sample_phases[0]
        right_phaseloads = sample_powerloads
        right_normalflags = sample_normalflags

        wrong_phaseloads = copy.deepcopy(right_phaseloads)
        wrong_phaseloads[0] = "non-PowerLoad"
        self.assert_constructor_fails(
            right_name,
            example_phase,
            wrong_phaseloads,
            right_normalflags,
        )

    def test_validate_normalize(self):
        sample_phases = self.sample_phases
        sample_powerloads = self.sample_powerloads
        sample_normalflags = self.sample_normalflags

        right_name = [
            "PhaseLoad with a non-boolean",
            "element in its 'normalize' list argument",
        ]
        right_name = " ".join(right_name)
        example_phase = sample_phases[0]
        right_phaseloads = sample_powerloads
        right_normalflags = sample_normalflags

        wrong_normalflags = copy.deepcopy(right_normalflags)
        wrong_normalflags[0] = "non-boolean"
        self.assert_constructor_fails(
            right_name,
            example_phase,
            right_phaseloads,
            wrong_normalflags,
        )

    def test_sanity(self):
        sample_phases = self.sample_phases
        sample_powerloads = self.sample_powerloads
        sample_normalflags = self.sample_normalflags

        right_name = [
            "PhaseLoad with different lengths",
            "of 'powerload_set' & 'normalize' arguments",
        ]
        right_name = " ".join(right_name)
        example_phase = sample_phases[0]
        right_phaseloads = sample_powerloads
        right_normalflags = sample_normalflags

        wrong_phaseloads = copy.deepcopy(right_phaseloads)
        wrong_phaseloads.pop()
        self.assert_constructor_fails(
            right_name,
            example_phase,
            wrong_phaseloads,
            right_normalflags,
        )

        wrong_normalflags = copy.deepcopy(right_normalflags)
        wrong_normalflags.pop()
        self.assert_constructor_fails(
            right_name,
            example_phase,
            right_phaseloads,
            wrong_normalflags,
        )

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    def test_compute_normalized_set(self):
        multisample = self.construct_multisample()
        normalized_set = multisample._compute_normalized_set()

        normalize = multisample.normalize
        n_normalize = len(normalize)

        for n in range(n_normalize):
            normalization_flag = normalize[n]
            powerload = normalized_set[n]
            powerdata_set = powerload.powerdata_set
            for powerdata in powerdata_set:
                norm = powerdata._norm
                if normalization_flag:
                    assert len(norm) != 0
                else:
                    assert len(norm) == 0

    def test_curve(self):
        pass  # No new functionality to be tested.

    """
    def test_shift_time(self, time_shift):
        list_of_time_shifts = attribute_manipulation_examples

        multisample = self.construct_multisample()
        normalize = multisample.normalize
        n_normalize = len(normalize)

        for time_shift in list_of_time_shifts:
            shifted_multisample = multisample._shift_time(time_shift)

            normalized_set = shifted_multisample._compute_normalized_set()
            for powerload in normalized_set:
                powerdata_set = powerload.powerdata_set
                for powerdata in powerdata_set:
                    shift = powerdata._shift
                    assert len(shift) != 0
    """

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    def test_intrinsic_time(self):
        multisample = self.construct_multisample()
        normalized_set = multisample._compute_normalized_set()
        rel_tol = None
        abs_tol = None

        normalize = multisample.normalize
        n_normalize = len(normalize)

        intrinsic_time = multisample.intrinsic_time()
        n_intrinsic = len(intrinsic_time)
        for n in range(n_normalize):
            normalization_flag = normalize[n]
            powerload = normalized_set[n]
            powerdata_set = powerload.powerdata_set
            for powerdata in powerdata_set:
                norm = np.prod(powerdata._norm)
                time = powerdata.time

                check = []
                for t in time:

                    if normalization_flag:
                        denormalized_t = t / norm
                        denormalized_list = [denormalized_t] * n_intrinsic

                        time_zip = zip(denormalized_list, intrinsic_time)
                        for (dt, it) in time_zip:
                            c = dt == pytest.approx(it, rel=rel_tol, abs=abs_tol)
                            check.append(c)

                    else:
                        normalized_list = [t] * n_intrinsic

                        time_zip = zip(normalized_list, intrinsic_time)
                        for (nt, it) in time_zip:
                            c = nt == pytest.approx(it, rel=rel_tol, abs=abs_tol)
                            check.append(c)

                t_was_in_intrinsic_time = any(check)  # CHECK IS ALL FALSE!
                assert t_was_in_intrinsic_time

    @pytest.mark.parametrize("color_index", [2])
    @pytest.mark.parametrize("detailed_plot_flag", [False, True])
    def test_plot(self, color_index, detailed_plot_flag):

        figure_title = "'detailed' flag = " + str(detailed_plot_flag)
        figure_title = "PhaseLoad Plotting (" + figure_title + ")"
        ax = tools_testkit.prepare_figure(figure_title)

        colors = netloads_testkit.color_order_for_plotting
        sample_color = colors[color_index]

        multisample = self.construct_multisample()
        list_of_plot_objects = multisample.plot(
            ax=ax, detailed=detailed_plot_flag, c=sample_color
        )

        adjust_2d_graph_ranges(ax=ax)
        plt.show()  # Run with `pytest --plotting-on` to visualize


class TestPulseLoad:
    def setup_method(self):
        (
            n_samples,
            sample_names,
            sample_phaseloads,
        ) = netloads_testkit.inputs_for_pulseload()

        all_samples = []
        for s in range(n_samples):
            name = sample_names[s]
            phaseload = sample_phaseloads[s]
            sample = PulseLoad(name, phaseload)
            all_samples.append(sample)
        self.sample_phaseloads = sample_phaseloads
        self.all_samples = all_samples

    def construct_multisample(self):
        sample_phaseloads = self.sample_phaseloads

        name = "PulseLoad with multiple phaseload arguments"
        multisample = PulseLoad(
            name,
            sample_phaseloads,
        )
        return multisample

    def test_constructor_with_multiple_arguments(self):
        multisample = self.construct_multisample()
        assert isinstance(multisample, PulseLoad)

    def assert_constructor_fails(
        self,
        test_name,
        test_phaseloads,
    ):
        constructor_exceptions = PowerCycleABCError
        with pytest.raises(constructor_exceptions):
            wrong_sample = PulseLoad(
                test_name,
                test_phaseloads,
            )

    def test_validate_phaseload_set(self):
        sample_phaseloads = self.sample_phaseloads

        right_name = [
            "PulseLoad with a non-PhaseLoad",
            "element in its 'phaseload_set' list argument",
        ]
        right_name = " ".join(right_name)
        right_phaseloads = sample_phaseloads

        wrong_phaseloads = copy.deepcopy(right_phaseloads)
        wrong_phaseloads[0] = "non-PhaseLoad"
        self.assert_constructor_fails(
            right_name,
            wrong_phaseloads,
        )

    def test_build_pulse(self):
        sample_phaseloads = self.sample_phaseloads
        n_phaseloads = len(sample_phaseloads)

        multisample = self.construct_multisample()
        built_pulse = multisample.pulse
        assert isinstance(built_pulse, PowerCyclePulse)

        pulse_phases = built_pulse.phase_set
        n_phases = len(pulse_phases)

        assert n_phaseloads == n_phases
        for i in range(n_phases):
            current_phaseload = sample_phaseloads[i]
            current_phase = pulse_phases[i]
            assert current_phaseload.phase == current_phase

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    def _compute_shifted_set(self):
        # test that the last time of each phaseload intrinsic time,
        # minus the last time of the previous phase load intrinsic time,
        # is equal to the phase duration
        pass

    def test_curve(self):
        pass  # No new functionality to be tested.

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------
    def test_intrinsic_time(self):
        # Test that each time of original power datas can be found in
        # intrinsic time after normalization+shift
        pass

    @pytest.mark.parametrize("color_index", [3])
    @pytest.mark.parametrize("detailed_plot_flag", [False, True])
    def test_plot(self, color_index, detailed_plot_flag):

        figure_title = "'detailed' flag = " + str(detailed_plot_flag)
        figure_title = "PulseLoad Plotting (" + figure_title + ")"
        ax = tools_testkit.prepare_figure(figure_title)

        colors = netloads_testkit.color_order_for_plotting
        sample_color = colors[color_index]

        multisample = self.construct_multisample()
        list_of_plot_objects = multisample.plot(
            ax=ax, detailed=detailed_plot_flag, c=sample_color
        )

        adjust_2d_graph_ranges(ax=ax)
        plt.show()  # Run with `pytest --plotting-on` to visualize
