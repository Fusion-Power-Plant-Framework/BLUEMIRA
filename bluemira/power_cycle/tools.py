# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Utility functions for the power cycle model.
"""

import json
from typing import Any

import matplotlib.pyplot as plt
import numba as nb
import numpy as np

from bluemira.base.constants import EPS


def read_json(file_path) -> dict[str, Any]:
    """
    Returns the contents of a 'json' file.
    """
    with open(file_path) as json_file:
        return json.load(json_file)


def create_axes(ax=None):
    """
    Create axes object.

    If 'None', creates a new 'axes' instance.
    """
    if ax is None:
        _, ax = plt.subplots()
    return ax


@nb.jit
def _nudge_check(x_last, x_this):
    is_close = np.isclose(x_last, x_this, rtol=EPS)
    is_decreasing = x_last > x_this
    return is_close or is_decreasing


@nb.jit
def validate_monotonic_increase(x, strict_flag):
    """Validate that vector is (strictly) monotonically increasing."""
    for x_last, x_this in zip(x[:-1], x[1:]):  # noqa: RUF007, B905
        if strict_flag and not (x_this > x_last):
            raise ValueError("Vector is not strictly monotonically increasing.")
        if not strict_flag and not (x_this >= x_last):
            raise ValueError("Vector is not monotonically increasing.")


@nb.jit
def unique_domain(x: np.ndarray, epsilon: float = 1e-10):
    """
    Ensure x has only unique values to make (Domain: x -> Image: y) a function.

    To be a function domain, x must be strictly monotonically increasing. So x
    must start at least as a monotonically increasing vector. The function then
    ensures strict monotonicity by nudging forward repeated values by a small
    epsilon.

    Epsilon must be small enough so that consecutive values can be considered
    equal within the context/scale in which the function is defined.

    The nudge forward for each non-unique element in x is (N * epsilon), given
    N times, after the first appearance, that value has appeared in x before.
    """
    x = nb.typed.List(x)
    new_x = nb.typed.List()
    validate_monotonic_increase(x, strict_flag=False)
    new_x.append(x[0])
    if len(x) > 1:
        for x_this in x[1:]:
            nudge = 0
            new_x_this = x_this
            needs_nudge = _nudge_check(new_x[-1], x_this)
            if needs_nudge:
                while needs_nudge:
                    nudge += epsilon
                    new_x_this = x_this + nudge
                    needs_nudge = _nudge_check(new_x[-1], new_x_this)
            else:
                new_x_this = x_this
            new_x.append(new_x_this)
    validate_monotonic_increase(new_x, strict_flag=True)
    return np.asarray(new_x)


def match_domains(
    all_x: list[np.ndarray],
    all_y: list[np.ndarray],
    epsilon: float = 1e-10,
):
    """
    Match the domains of multiple functions, each represented by 2 vectors.

    First, for each pair of vectors (x,y) that define a function, the domain
    x is ensured to be unique with 'unique_domain' (otherwise, calling 'unique'
    on 'x_matched' later can neglect step functions).

    Then, for each pair (x,y), values in every 'y' vector are interpolated,
    to ensure that all of them have one element associated to every element
    of the union of all distinct 'x' vectors. Values defined before and after
    the original domain 'x' of the image 'y' are set to zero.
    """
    n_vectors = len(all_x)
    for v in range(n_vectors):
        numba_safe_x = nb.typed.List([float(x) for x in all_x[v]])
        all_x[v] = unique_domain(numba_safe_x, epsilon=epsilon)
    x_matched = np.unique(np.concatenate(all_x))

    all_y_matched = all_y.copy()
    for v in range(n_vectors):
        all_y_matched[v] = np.interp(
            x_matched,
            all_x[v],
            all_y[v],
            left=0,
            right=0,
        )
    return x_matched, all_y_matched
