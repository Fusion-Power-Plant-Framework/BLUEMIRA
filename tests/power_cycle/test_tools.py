# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.power_cycle.tools import (
    match_domains,
    rms_deviation,
    unique_domain,
    validate_monotonic_increase,
)


def test_validate_monotonic_increase():
    x_strict_monotonic = np.array([1, 2, 3, 4, 5])
    x_monotonic = np.array([1, 2, 2, 4, 5])
    x_not_increasing = np.array([1, 2, 3, 2, 1])
    msg_not_strict = "Vector is not strictly monotonically increasing."
    msg_not_increasing = "Vector is not monotonically increasing."

    validate_monotonic_increase(x_strict_monotonic, strict_flag=True)
    validate_monotonic_increase(x_monotonic, strict_flag=False)

    with pytest.raises(ValueError, match=msg_not_strict):
        validate_monotonic_increase(x_monotonic, strict_flag=True)

    with pytest.raises(ValueError, match=msg_not_increasing):
        validate_monotonic_increase(x_not_increasing, strict_flag=False)


@pytest.mark.parametrize("epsilon", [1e-10, 1e-7, 1e-5])
def test_unique_domain(epsilon):
    x = np.array([1, 1, 2, 2, 3, 3])
    new_x = unique_domain(x, epsilon=epsilon)
    expected_x = np.array([
        1,
        1 + epsilon,
        2,
        2 + epsilon,
        3,
        3 + epsilon,
    ])
    assert all(np.isclose(new_x, expected_x))

    x = np.array([2, 2, 2, 2 + (epsilon / 2)])
    new_x = unique_domain(x, epsilon=epsilon)
    expected_x = np.array([
        2,
        2 + 1 * epsilon,
        2 + 2 * epsilon,
        2 + 2 * epsilon + (epsilon / 2),
    ])
    assert all(np.isclose(new_x, expected_x))

    x = np.array([2, 2, 2, 2 + epsilon, 3, 3])
    new_x = unique_domain(x, epsilon=epsilon)
    expected_x = np.array([
        2,
        2 + 1 * epsilon,
        2 + 2 * epsilon,
        2 + 3 * epsilon,
        3,
        3 + 1 * epsilon,
    ])
    assert all(np.isclose(new_x, expected_x))

    x = np.array([1, 1, 1, 1, 1])
    max_iterations = 3
    with pytest.raises(ValueError, match="Maximum number of iterations"):
        unique_domain(x, max_iterations=max_iterations)


def test_match_domains():
    all_x = [np.array([1, 2, 3]), np.array([2, 3, 4])]
    all_y = [np.array([1, 2, 3]), np.array([4, 6, 8])]
    out_of_bounds_value = 0

    x_expected = np.array([1, 2, 3, 4])
    all_y_expected = [
        np.array([1, 2, 3, out_of_bounds_value]),
        np.array([out_of_bounds_value, 4, 6, 8]),
    ]
    x_matched, all_y_matched = match_domains(all_x, all_y)
    assert np.array_equal(x_matched, x_expected)
    assert np.array_equal(all_y_matched[0], all_y_expected[0])
    assert np.array_equal(all_y_matched[1], all_y_expected[1])


def test_rms_deviation():
    coordinate = [0, 1, 2, 3, 4]
    abscissa_lin = coordinate
    abscissa_sqr = [0, 1, 4, 9, 16]
    fun_linear = [
        coordinate,
        abscissa_lin,
    ]
    fun_square = [
        coordinate,
        abscissa_sqr,
    ]
    expected_err = np.subtract(abscissa_sqr, abscissa_lin)

    expected_rms = np.sqrt(np.mean(np.square(expected_err)))
    rms, _ = rms_deviation(fun_square, fun_linear, normalize=False)
    assert np.isclose(rms, expected_rms)

    norm = np.ptp(abscissa_sqr)
    expected_dev = expected_err / norm
    expected_rms = np.sqrt(np.mean(np.square(expected_dev)))
    rms, _ = rms_deviation(fun_square, fun_linear, normalize=True)
    assert np.isclose(rms, expected_rms)

    norm = np.ptp(abscissa_sqr[2:5])
    expected_err = np.subtract(abscissa_sqr[2:5], abscissa_lin[2:5])
    expected_dev = expected_err / norm
    expected_rms = np.sqrt(np.mean(np.square(expected_dev)))
    rms, trim = rms_deviation(fun_square, fun_linear, x_range=[2, 4])
    assert np.isclose(rms, expected_rms)
    assert np.array_equal(trim[0], coordinate[2:5])
    assert np.array_equal(trim[1], abscissa_sqr[2:5])
    assert np.array_equal(trim[2], abscissa_lin[2:5])
