# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
A collection of miscellaneous tools.
"""

from __future__ import annotations

import operator
import string
import warnings
from collections import Counter
from collections.abc import Callable, Iterable
from functools import wraps
from importlib import import_module as imp
from importlib import machinery as imp_mach
from importlib import util as imp_u
from itertools import permutations
from json import JSONEncoder, dumps
from pathlib import Path
from typing import TYPE_CHECKING, Any

import nlopt
import numpy as np
import numpy.typing as npt
from matplotlib import colors

from bluemira.base.constants import E_I, E_IJ, E_IJK
from bluemira.base.file import force_file_extension
from bluemira.base.look_and_feel import (
    bluemira_debug,
    bluemira_error,
    bluemira_print,
    bluemira_warn,
)

if TYPE_CHECKING:
    from os import PathLike
    from types import ModuleType

    import numpy.typing as npt

    from bluemira.display.palettes import ColorPalette

# =====================================================
# JSON utilities
# =====================================================


class NumpyJSONEncoder(JSONEncoder):
    """
    A JSON encoder that can handle numpy arrays.
    """

    def default(self, o):
        """
        Override the JSONEncoder default object handling behaviour for np.arrays.

        Returns
        -------
        :
            The default json encoder object
        """
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def json_writer(
    data: dict[str, Any],
    file: PathLike | str | None = None,
    *,
    return_output: bool = False,
    cls: type[JSONEncoder] = NumpyJSONEncoder,
    **kwargs,
) -> str | None:
    """
    Write json in the bluemria style.

    Parameters
    ----------
    data:
        dictionary to write to json
    filename:
        filename to write to
    return_output:
        return the json as a string
    cls:
        json encoder child class
    kwargs:
        all further kwargs passed to the json writer

    Returns
    -------
    :
        The json dictionary if requested
    """
    if file is None and not return_output:
        bluemira_warn("No json action to take")
        return None

    if "indent" not in kwargs:
        kwargs["indent"] = 4

    the_json = dumps(data, cls=cls, **kwargs)

    if file is not None:
        with open(file, "w") as fh:
            fh.write(the_json)
            fh.write("\n")

    if return_output:
        return the_json
    return None


# =====================================================
# csv writer utilities
# =====================================================
def write_csv(
    data: np.ndarray,
    base_name: str,
    col_names: list[str],
    metadata: str = "",
    ext: str = ".csv",
    comment_char: str = "#",
):
    """
    Write data in comma-separated value format.

    Parameters
    ----------
    data:
        Array of data to be written to csv file. Will raise an error if the
        dimensionality of the data is not two
    base_name:
        Name of file to write to, minus the extension.
    col_names:
        List of strings for column headings for each data field provided.
    metadata:
        Optional argument for metadata to be written as a header.
    ext:
        Optional argument for file extension, defaults to ".csv".
    comment_char:
        Optional argument to specify character(s) to prepend to metadata lines
        as a comment character (defaults to "#").

    Raises
    ------
    ValueError
        Columns names not available for all columns
    """
    # Fetch number of cols
    shape = data.shape
    n_cols = 1 if len(shape) < 2 else shape[1]  # noqa: PLR2004

    # Write file name
    filename = force_file_extension(base_name, ext)

    # Write column names
    if len(col_names) != n_cols:
        raise ValueError("Column names must be provided for all data fields")

    # Add comment characters and newline to existing metadata
    if metadata:
        comment_prefix = f"{comment_char} "
        metadata = (
            "\n".join([comment_prefix + line for line in metadata.split("\n")]) + "\n"
        )

        # Add column headings
    metadata += ",".join(col_names)

    np.savetxt(
        filename,
        data,
        fmt="%.5e",
        delimiter=",",
        header=metadata or "",
        footer="",
        comments="",
    )
    bluemira_print("Wrote to " + filename)


# =====================================================
# Einsum utilities
# =====================================================
def asciistr(length: int) -> str:
    """
    Get a string of characters of desired length.

    Current max is 52 characters

    Parameters
    ----------
    length:
        number of characters to return

    Returns
    -------
    str of length specified

    Raises
    ------
    ValueError
        String length > 52 characters
    """
    if length > 52:  # noqa: PLR2004
        raise ValueError("Unsupported string length")

    return string.ascii_letters[:length]


def levi_civita_tensor(dim: int = 3) -> np.ndarray:
    """
    N dimensional Levi-Civita Tensor.

    Parameters
    ----------
    dim:
        The number of dimensions for the LCT

    Returns
    -------
    np.array (n_0,n_1,...n_n)

    Notes
    -----
    The Levi-Civita symbol in n dimensions is defined as:

    .. math::
        \\epsilon_{i_1 i_2 \\dots i_n} =
        \\begin{cases} \\\\
            +1 & \\text{for even permutation of } (1, 2, \\dots, n) \\\\
            -1 & \\text{for odd permutation of } (1, 2, \\dots, n) \\\\
            0 & \\text{for indices are equal} \\\\
        \\end{cases}

    For dim=3, this looks like:

    .. math::
        \\epsilon_{ijk} =
        \\begin{cases}
        1 & \\text{if } (i, j, k) \\text{ is } (0, 1, 2), (1, 2, 0), (2, 0, 1) \\\\
        -1 & \\text{if } (i, j, k) \\text{ is } (0, 2, 1), (2, 1, 0), (1, 0, 2) \\\\
        0 & \\text{otherwise}
        \\end{cases}

    """
    perms = np.array(list(set(permutations(np.arange(dim)))))

    e_ijk = np.zeros([dim for d in range(dim)])

    idx = np.triu_indices(n=dim, k=1)

    for perm in perms:
        e_ijk[tuple(perm)] = np.prod(np.sign(perm[idx[1]] - perm[idx[0]]))

    return e_ijk


class EinsumWrapper:
    """
    Preallocator for einsum versions of dot, cross and norm.
    """

    def __init__(self):
        norm_a0 = "ij, ij -> j"
        norm_a1 = "ij, ij -> i"

        self.norm_strs = [norm_a0, norm_a1]

        # Not fool proof for huge no's of dims
        self.dot_1x1 = "i, i -> ..."
        self.dot_1x2 = "i, ik -> k"
        self.dot_2x1 = "ij, j -> i"
        self.dot_2x2 = "ij, jk -> ik"
        self.dot_1xn = "y, {}yz -> {}z"
        self.dot_nx1 = "{}z, z -> {}"
        self.dot_nxn = "{}y, {}yz -> {}z"

        cross_2x1 = "i, i, i -> i"
        cross_2x2 = "xy, ix, iy -> i"
        cross_2x3 = "xyz, ix, iy -> iz"

        self.cross_strs = [cross_2x1, cross_2x2, cross_2x3]
        self.cross_lcts = [E_I, E_IJ, E_IJK]

    def norm(self, ix: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        Emulates some of the functionality of np.linalg.norm for 2D arrays.

        Specifically:
        np.linalg.norm(ix, axis=0)
        np.linalg.norm(ix, axis=1)

        For optimum speed and customisation use np.einsum modified for your use case.

        Parameters
        ----------
        ix:
            Array to perform norm on
        axis:
            axis for the norm to occur on

        Returns
        -------
        :
            The norm of the array

        Raises
        ------
        ValueError
            Matrix dimensions >2 unsupported
        """
        try:
            return np.sqrt(np.einsum(self.norm_strs[axis], ix, ix))
        except IndexError:
            raise ValueError("matrices dimensions >2d Unsupported") from None

    def dot(
        self, ix: np.ndarray, iy: np.ndarray, out: np.ndarray | None = None
    ) -> np.ndarray:
        """
        A dot product emulation using np.einsum.

        For optimum speed and customisation use np.einsum modified for your use case.

        Should follow the same mechanics as np.dot, a few examples:

        ein_str = 'i, i -> ...'
        ein_str = 'ij, jk -> ik' # Classic dot product
        ein_str = 'ij, j -> i'
        ein_str = 'i, ik -> k'
        ein_str = 'aij, ajk -> aik' # for loop needed with np.dot

        Parameters
        ----------
        ix:
            First array
        iy:
            Second array
        out:
            output array for inplace dot product

        Returns
        -------
        :
            The dot product of the array

        Raises
        ------
        ValueError
            Undefined dot product behaviour for array shapes
        """
        # Ordered hopefully by most used
        dim_sw = 2
        if ix.ndim == dim_sw and iy.ndim == dim_sw:
            out_str = self.dot_2x2
        elif ix.ndim > dim_sw and iy.ndim > dim_sw:
            ix_str = asciistr(ix.ndim - 1)
            iy_str = asciistr(iy.ndim - 2)
            out_str = self.dot_nxn.format(ix_str, iy_str, ix_str)
        elif ix.ndim < dim_sw and iy.ndim == dim_sw:
            out_str = self.dot_1x2
        elif ix.ndim >= dim_sw and iy.ndim < dim_sw:
            ix_str = asciistr(ix.ndim - 1)
            out_str = self.dot_nx1.format(ix_str, ix_str)
        elif iy.ndim >= dim_sw or ix.ndim == dim_sw:
            raise ValueError(
                f"Undefined behaviour ix.shape:{ix.shape}, iy.shape:{iy.shape}"
            )
        else:
            out_str = self.dot_1x1

        return np.einsum(out_str, ix, iy, out=out)

    def cross(
        self, ix: np.ndarray, iy: np.ndarray, out: np.ndarray | None = None
    ) -> np.ndarray:
        """
        A row-wise cross product of a 2D matrices of vectors.

        This function mirrors the properties of np.cross
        such as vectors of 2 or 3 elements. 1D is also accepted
        but just do x * y.
        Only 7D has similar orthogonal properties above 3D.

        For optimum speed and customisation use np.einsum modified for your use case.

        Parameters
        ----------
        ix:
            1st array to cross
        iy:
            2nd array to cross
        out:
            output array for inplace cross product

        Returns
        -------
        :
            the cross product of the arrays

        Raises
        ------
        ValueError
            If the dimensions of the cross product are > 3

        """
        dim = ix.shape[-1] - 1 if ix.ndim > 1 else 0

        try:
            return np.einsum(self.cross_strs[dim], self.cross_lcts[dim], ix, iy, out=out)
        except IndexError:
            raise ValueError("Incompatible dimension for cross product") from None


wrap = EinsumWrapper()

norm = wrap.norm
dot = wrap.dot
cross = wrap.cross

# =====================================================
# Misc utilities
# =====================================================


def floatify(x: npt.ArrayLike) -> float:
    """
    Converts the np array or float into a float by returning
    the first element or the element itself.

    Notes
    -----
    This function aims to avoid numpy warnings for float(x) for >0 rank scalars
    it emulates the functionality of float conversion

    Returns
    -------
    :
        the extracted float

    Raises
    ------
    ValueError
        If array like object has more than 1 element
    TypeError
        If object is None
    """
    if x is None:
        raise TypeError("The argument cannot be None")
    return np.asarray(x, dtype=float).item()


class ColourDescriptor:
    """Colour Descriptor for use with dataclasses"""

    def __init__(self):
        self._default = colors.to_hex((0.5, 0.5, 0.5))

    def __set_name__(self, _, name: str):
        """Set the attribute name from a dataclass"""
        self._name = "_" + name

    def __get__(self, obj: Any, _) -> str:
        """Get the hex colour

        Returns
        -------
        :
            The hex colour string
        """
        if obj is None:
            return self._default

        return colors.to_hex(getattr(obj, self._name, self._default))

    def __set__(self, obj: Any, value: str | tuple[float, ...] | ColorPalette):
        """
        Set the colour

        Notes
        -----
        The value can be anything accepted by matplotlib.colors.to_hex
        """
        if hasattr(value, "as_hex"):
            value = value.as_hex()
            if isinstance(value, list):
                value = value[0]
        setattr(obj, self._name, value)


def iterable_to_list(obj: Any | Iterable[Any]) -> list[Any]:
    """Convert object(s) to an explicit list of objects

    Returns
    -------
    :
        The object converted to a list
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, Iterable):
        return [*obj]
    return [obj]


def is_num(thing: Any) -> bool:
    """
    Determine whether or not the input is a number.

    Parameters
    ----------
    thing:
        The input which we need to determine is a number or not

    Returns
    -------
    :
        Whether or not the input is a number
    """
    if thing is True or thing is False:
        return False
    try:
        thing = floatify(thing)
    except (ValueError, TypeError):
        return False
    else:
        return not np.isnan(thing)


def is_num_array(thing: Any) -> bool:
    """
    :func:`~bluemira.utilities.tools.is_num` but also includes arrays

    Returns
    -------
    :
        Whether or not the input is a number
    """
    if isinstance(thing, np.ndarray) and thing.dtype in {float, int, complex}:
        return ~np.isnan(thing)
    return is_num(thing)


def abs_rel_difference(v2: float, v1_ref: float) -> float:
    """
    Calculate the absolute relative difference between a new value and an old
    reference value.

    Parameters
    ----------
    v2:
        The new value to compare to the old
    v1_ref:
        The old reference value

    Returns
    -------
    :
        The absolute relative difference between v2 and v1ref
    """
    return abs((v2 - v1_ref) / v1_ref)


def set_random_seed(seed_number: int):
    """
    Sets the random seed number in numpy and NLopt. Useful when repeatable
    results are desired in Monte Carlo methods and stochastic optimisation
    methods.

    Parameters
    ----------
    seed_number:
        The random seed number, preferably a very large integer
    """
    np.random.default_rng(seed_number)
    nlopt.srand(seed_number)


def compare_dicts(
    d1: dict[str, Any],
    d2: dict[str, Any],
    *,
    almost_equal: bool = False,
    verbose: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """
    Compares two dictionaries. Will print information about the differences
    between the two to the console. Dictionaries are compared by length, keys,
    and values per common keys

    Parameters
    ----------
    d1:
        The reference dictionary
    d2:
        The dictionary to be compared with the reference
    almost_equal:
        Whether or not to use np.isclose and np.allclose for numbers and arrays
    verbose:
        Whether or not to print to the console
    rtol:
        The relative tolerance parameter, used if ``almost_equal`` is True
    atol:
        The absolute tolerance parameter, used if ``almost_equal`` is True

    Returns
    -------
    Whether or not the dictionaries are the same
    """
    nkey_diff = len(d1) - len(d2)
    k1 = set(d1.keys())
    k2 = set(d2.keys())
    intersect = k1.intersection(k2)
    new_diff = k1 - k2
    old_diff = k2 - k1
    same, different = [], []

    # Define functions to use for comparison in either the array, dict, or
    # numeric cases.
    def dict_eq(value_1, value_2):
        return compare_dicts(
            value_1,
            value_2,
            almost_equal=almost_equal,
            verbose=verbose,
            rtol=rtol,
            atol=atol,
        )

    def array_almost_eq(val1, val2):
        return np.allclose(val1, val2, rtol, atol)

    def num_almost_eq(val1, val2):
        return np.isclose(val1, val2, rtol, atol)

    def array_is_eq(val1, val2):
        return (np.asarray(val1) == np.asarray(val2)).all()

    def list_eq(val1, val2):
        return Counter(val1) == Counter(val2)

    if almost_equal:
        array_eq = array_almost_eq
        num_eq = num_almost_eq
    else:
        array_eq = array_is_eq
        num_eq = operator.eq

    # Map the comparison functions to the keys based on the type of value in d1.
    comp_map = {
        key: (
            array_eq
            if isinstance(val, np.ndarray)
            or (isinstance(val, list) and is_num(next(flatten_iterable(val))))
            else (
                dict_eq
                if isinstance(val, dict)
                else list_eq
                if isinstance(val, list)
                else num_eq
                if is_num(val)
                else operator.eq
            )
        )
        for key, val in d1.items()
    }

    # Do the comparison
    for k in intersect:
        v1, v2 = d1[k], d2[k]
        try:
            if comp_map[k](v1, v2):
                same.append(k)
            else:
                different.append(k)
        except ValueError:  # One is an array and the other not
            different.append(k)

    the_same = False
    result = "===========================================================\n"
    if nkey_diff != 0:
        compare = "more" if nkey_diff > 0 else "fewer"
        result += f"d1 has {nkey_diff} {compare} keys than d2" + "\n"
    if new_diff != set():
        result += "d1 has the following keys which d2 does not have:\n"
        new_diff = ["\t" + str(i) for i in new_diff]
        result += "\n".join(new_diff) + "\n"
    if old_diff != set():
        result += "d2 has the following keys which d1 does not have:\n"
        old_diff = ["\t" + str(i) for i in old_diff]
        result += "\n".join(old_diff) + "\n"
    if different:
        result += "the following shared keys have different values:\n"
        different = ["\t" + str(i) for i in different]
        result += "\n".join(different) + "\n"
    if nkey_diff == 0 and new_diff == set() and old_diff == set() and different == []:
        the_same = True
    else:
        result += "==========================================================="
        if verbose:
            bluemira_error(result)
    return the_same


def flatten_iterable(iters: Iterable[Any]):
    """
    Expands a nested iterable structure, flattening it into one iterable

    Parameters
    ----------
    lists: set of Iterables
        The object(s) to de-nest

    Yields
    ------
        elements of iterable

    Notes
    -----
    Does not cater for nested dictionaries

    """
    for _iter in iters:
        if isinstance(_iter, Iterable) and not isinstance(_iter, str | bytes | dict):
            yield from flatten_iterable(_iter)
        else:
            yield _iter


def consec_repeat_elem(arr: np.ndarray, num_rep: int) -> np.ndarray:
    """
    Get array of repeated elements with n or more repeats

    Parameters
    ----------
    arr:
        array to find repeats in
    num_rep:
        number of repetitions to find

    Returns
    -------
    :
        The repeated element array
    """
    if num_rep <= 1:
        raise NotImplementedError("Not implemented for less than 2 repeat elements")
    n = num_rep - 1
    m = arr[:-1] == arr[1:]
    return np.flatnonzero(np.convolve(m, np.ones(n, dtype=int)) == n) - n + 1


def slope(arr: np.ndarray) -> float:
    """Calculate gradient of a 2x2 point array

    Returns
    -------
    :
        The gradient of the array
    """
    b = arr[1, 0] - arr[0, 0]
    return np.inf if b == 0 else (arr[1, 1] - arr[0, 1]) / b


def yintercept(arr: np.ndarray) -> tuple[float, float]:
    """Calculate the y intercept and gradient of an array

    Returns
    -------
    :
        The y-intercept of the array
    """
    s = slope(arr)
    return arr[0, 1] - s * arr[0, 0], s


def ten_power(x):
    """Get the power for the base ten notation, set 0 to 0.

    Returns
    -------
    :
        the nearest log10 power
    """
    x = np.atleast_1d(x)
    tp = np.zeros_like(x)
    ind = np.nonzero(x != 0)
    tp[ind] = np.floor(np.log10(np.abs(x[ind])))
    return tp if x.size > 1 else tp.item()


def sig_fig_round(x, s, low_lim=-16):
    """
    Fuction to round to a given number of significant figures,
    with any number below a lower limit set to zero.

    Parameters
    ----------
    x:
        value or values to round.
    s:
        number of significant figures
    low_lim:
        power below which values are set to 0,
        default: low_lim = -16 (i.e, numbers below 1e-16)

    Returns
    -------
    :
        Rounded value

    """
    tp = ten_power(x)
    x_round = np.round(x / 10.0**tp, s - 1) * 10.0**tp
    return x_round * (tp >= low_lim)


# ======================================================================================
# Coordinate system transformations
# ======================================================================================


def cartesian_to_polar(
    x: np.ndarray, z: np.ndarray, x_ref: float = 0.0, z_ref: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert from 2-D Cartesian coordinates to polar coordinates about a reference point.

    Parameters
    ----------
    x:
        Radial coordinates
    z:
        Vertical coordinates
    x_ref:
        Reference radial coordinate
    z_ref:
        Reference vertical coordinate

    Returns
    -------
    r:
        Polar radial coordinates
    phi:
        Polar angle coordinates
    """
    xi, zi = x - x_ref, z - z_ref
    r = np.hypot(xi, zi)
    phi = np.arctan2(zi, xi)
    return r, phi


def polar_to_cartesian(
    r: np.ndarray, phi: np.ndarray, x_ref: float = 0.0, z_ref: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert from 2-D polar to Cartesian coordinates about a reference point.

    Parameters
    ----------
    r:
        Polar radial coordinates
    phi:
        Polar angle coordinates
    x_ref:
        Reference radial coordinate
    z_ref:
        Reference vertical coordinate

    Returns
    -------
    x:
        Radial coordinates
    z:
        Vertical coordinate
    """
    x = x_ref + r * np.cos(phi)
    z = z_ref + r * np.sin(phi)
    return x, z


def toroidal_to_cylindrical(R_0: float, z_0: float, tau: np.ndarray, sigma: np.ndarray):
    """
    Convert from toroidal coordinates to cylindrical coordinates in the poloidal plane
    Toroidal coordinates are denoted by (\\tau, \\sigma, \\phi)
    Cylindrical coordinates are denoted by (r, z, \\phi)
    We are in the poloidal plane so take the angle \\phi = 0

    .. math::
        R = R_{0} \\frac{\\sinh{\\tau}}{\\cosh{\\tau} - \\cos{\\sigma}}
        z = R_{0} \\frac{\\sin{\\tau}}{\\cosh{\\tau} - \\cos{\\sigma}} + z_{0}

    Parameters
    ----------
    R_0:
        r coordinate of focus in poloidal plane
    z_0:
        z coordinate of focus in poloidal plane
    tau:
        the tau coordinates to transform
    sigma:
        the sigma coordinates to transform

    Returns
    -------
    R, Z:
        Tuple of transformed coordinates in cylindrical form
    """
    R = R_0 * np.sinh(tau) / (np.cosh(tau) - np.cos(sigma))  # noqa: N806
    Z = R_0 * np.sin(sigma) / (np.cosh(tau) - np.cos(sigma)) + z_0  # noqa: N806
    return R, Z


def cylindrical_to_toroidal(R_0: float, z_0: float, R: np.ndarray, Z: np.ndarray):  # noqa: N803
    """
    Convert from cylindrical coordinates to toroidal coordinates in the poloidal plane
    Toroidal coordinates are denoted by (\\tau, \\sigma, \\phi)
    Cylindrical coordinates are denoted by (r, z, \\phi)
    We are in the poloidal plane so take the angle \\phi = 0

    .. math::
        \\tau = \\ln\\frac{d_{1}}{d_{2}}
        \\sigma = sign(z - z_{0}) \\arccos\\frac{d_{1}^2 + d_{2}^2
                                        - 4 R_{0}^2}{2 d_{1} d_{2}}

        d_{1}^2 = (R + R_{0})^2 + (z - z_{0})^2
        d_{2}^2 = (R - R_{0})^2 + (z - z_{0})^2

    Parameters
    ----------
    R_0:
        r coordinate of focus in poloidal plane
    z_0:
        z coordinate of focus in poloidal plane
    R:
        the r coordinates to transform
    Z:
        the z coordinates to transform

    Returns
    -------
    tau, sigma:
        Tuple of transformed coordinates in toroidal form
    """
    d_1 = np.sqrt((R + R_0) ** 2 + (Z - z_0) ** 2)
    d_2 = np.sqrt((R - R_0) ** 2 + (Z - z_0) ** 2)
    tau = np.log(d_1 / d_2)
    sigma = np.sign(Z - z_0) * np.arccos(
        np.clip((d_1**2 + d_2**2 - 4 * R_0**2) / (2 * d_1 * d_2), -1, 1)
    )
    return tau, sigma


# ======================================================================================
# Dynamic module loading
# ======================================================================================


def get_module(name: str) -> ModuleType:
    """
    Load module dynamically.

    Parameters
    ----------
    name:
        Filename or python path (a.b.c) of module to import

    Returns
    -------
    Loaded module
    """
    try:
        module = imp(name)
    except ImportError:
        module = _loadfromspec(name)
    bluemira_debug(f"Loaded module {module.__name__}")
    return module


def _loadfromspec(name: str) -> ModuleType:
    """
    Load module from filename.

    Parameters
    ----------
    name:
        Filename of module to import

    Returns
    -------
    Loaded module

    Raises
    ------
    ImportError
        Unable to import module
    FileNotFoundError
        Cant find specified module file
    ModuleNotFoundError
        Cant find module
    """
    full_dirname = name.rsplit("/", 1)
    dirname = Path("." if len(full_dirname[0]) == 0 else full_dirname[0])
    path = Path(full_dirname[1])

    try:
        mod_files = [
            file for file in dirname.iterdir() if file.name.startswith(full_dirname[1])
        ]
    except FileNotFoundError:
        raise FileNotFoundError(f"Can't find module file '{name}'") from None

    if len(mod_files) == 0:
        raise FileNotFoundError(f"Can't find module file '{name}'")

    requested = path if path in mod_files else Path(dirname, mod_files[0])

    if len(mod_files) > 1:
        bluemira_warn(
            "{}{}".format(
                f"Multiple files start with '{path}'\n",
                f"Assuming module is '{requested}'",
            )
        )

    name, ext = (
        requested.name.rsplit(".", 1) if "." in requested.name else (requested.name, "")
    )
    if ext not in imp_mach.SOURCE_SUFFIXES:
        if ext and not ext.startswith("."):
            ext = f".{ext}"
        n_suffix = True
        imp_mach.SOURCE_SUFFIXES.append(ext)
    else:
        n_suffix = False

    try:
        spec = imp_u.spec_from_file_location(name, requested)
        module = imp_u.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ModuleNotFoundError:
        raise
    except (AttributeError, ImportError, SyntaxError) as err:
        raise ImportError(f"File '{mod_files[0]}' is not a module") from err

    if n_suffix:
        imp_mach.SOURCE_SUFFIXES.pop()

    return module


def get_class_from_module(name: str, default_module: str = "") -> type:
    """
    Load a class from a module dynamically.

    Parameters
    ----------
    name:
        Filename or python path (a.b.c) of module to import, with specific class to load
        appended following :: e.g. my_package.my_module::my_class. If the default_module
        is provided then only the class name (e.g. my_class) needs to be provided.
    default_module:
        The default module to search for the class, by default "". If provided then if
        name does not contain a module path then this the default module will be used to
        search for the class. Can be overridden if the name provides a module path.

    Returns
    -------
    Loaded class

    Raises
    ------
    ImportError
        Unable to import class from module
    """
    module = default_module
    class_name = name
    if "::" in class_name:
        module, class_name = class_name.split("::")
    try:
        output = getattr(get_module(module), class_name)
    except AttributeError as ae:
        raise ImportError(
            f"Unable to load class {class_name} - not in module {module}"
        ) from ae

    bluemira_debug(f"Loaded class {output.__name__}")
    return output


def array_or_num(array: Any) -> np.ndarray | float:
    """
    Always returns a numpy array or a float

    Parameters
    ----------
    array:
        The value to convert into a numpy array or number.

    Returns
    -------
    The value as a numpy array or number.

    Raises
    ------
    TypeError
        If the value cannot be converted to a numpy or number.
    """
    if is_num(array):
        return floatify(array)
    if isinstance(array, np.ndarray):
        return array
    raise TypeError


def deprecation_wrapper(
    message: Callable[[Any], Any] | str | None,
) -> Callable[[Any], Any]:
    """Deprecate any callable.

    Parameters
    ----------
    message:
        The callable to deprecate or the message to show

    Returns
    -------
    :
        The wrapped function
    """

    def _decorate(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
        """Deprecate any callable.

        Parameters
        ----------
        func:
            The callable to deprecate

        Returns
        -------
        :
            wrapped function
        """

        @wraps(func)
        def deprecator(*args, **kwargs) -> Any:
            warnings.warn(
                (
                    message
                    if isinstance(message, str)
                    else (
                        f"'{func.__name__}' is deprecated and will be removed in the"
                        " next major release"
                    )
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return deprecator

    if callable(message):
        return _decorate(message)

    return _decorate
