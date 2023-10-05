# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
A collection of miscellaneous tools.
"""
from __future__ import annotations

import operator
import string
import warnings
from collections.abc import Iterable
from functools import wraps
from importlib import import_module as imp
from importlib import machinery as imp_mach
from importlib import util as imp_u
from itertools import permutations
from json import JSONEncoder, dumps
from os import listdir
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Type, Union

import nlopt
import numpy as np
from matplotlib import colors

from bluemira.base.constants import E_I, E_IJ, E_IJK
from bluemira.base.look_and_feel import bluemira_debug, bluemira_warn

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

    from bluemira.display.palettes import ColorPalette

# =====================================================
# JSON utilities
# =====================================================


class NumpyJSONEncoder(JSONEncoder):
    """
    A JSON encoder that can handle numpy arrays.
    """

    def default(self, obj):  # noqa: PLR6301
        """
        Override the JSONEncoder default object handling behaviour for np.arrays.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def json_writer(
    data: Dict[str, Any],
    file: Optional[Union[Path, str]] = None,
    return_output: bool = False,
    *,
    cls: JSONEncoder = NumpyJSONEncoder,
    **kwargs,
):
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

    """
    if length > 52:  # noqa: PLR2004
        raise ValueError("Unsupported string length")

    return string.ascii_letters[:length]


def levi_civita_tensor(dim: int = 3) -> np.ndarray:
    """
    N dimensional Levi-Civita Tensor.

    For dim=3 this looks like:

    e_ijk = np.zeros((3, 3, 3))
    e_ijk[0, 1, 2] = e_ijk[1, 2, 0] = e_ijk[2, 0, 1] = 1
    e_ijk[0, 2, 1] = e_ijk[2, 1, 0] = e_ijk[1, 0, 2] = -1

    Parameters
    ----------
    dim:
        The number of dimensions for the LCT

    Returns
    -------
    np.array (n_0,n_1,...n_n)

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
        """
        try:
            return np.sqrt(np.einsum(self.norm_strs[axis], ix, ix))
        except IndexError:
            raise ValueError("matrices dimensions >2d Unsupported") from None

    def dot(
        self, ix: np.ndarray, iy: np.ndarray, out: Optional[np.ndarray] = None
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
        self, ix: np.ndarray, iy: np.ndarray, out: Optional[np.ndarray] = None
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


class ColourDescriptor:
    """Colour Descriptor for use with dataclasses"""

    def __init__(self):
        self._default = colors.to_hex((0.5, 0.5, 0.5))

    def __set_name__(self, _, name: str):
        """Set the attribute name from a dataclass"""
        self._name = "_" + name

    def __get__(self, obj: Any, _) -> str:
        """Get the hex colour"""
        if obj is None:
            return self._default

        return colors.to_hex(getattr(obj, self._name, self._default))

    def __set__(self, obj: Any, value: Union[str, Tuple[float, ...], ColorPalette]):
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


def is_num(thing: Any) -> bool:
    """
    Determine whether or not the input is a number.

    Parameters
    ----------
    thing: unknown type
        The input which we need to determine is a number or not

    Returns
    -------
    Whether or not the input is a number
    """
    if thing is True or thing is False:
        return False
    if thing is np.nan:
        return False
    try:
        float(thing)
    except (ValueError, TypeError):
        return False
    else:
        return True


def is_num_array(thing: Any) -> bool:
    """
    :func:is_num but also includes arrays
    """
    if isinstance(thing, np.ndarray) and thing.dtype in [float, int, complex]:
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
    d1: Dict[str, Any],
    d2: Dict[str, Any],
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
        return compare_dicts(value_1, value_2, almost_equal, verbose, rtol, atol)

    def array_almost_eq(val1, val2):
        return np.allclose(val1, val2, rtol, atol)

    def num_almost_eq(val1, val2):
        return np.isclose(val1, val2, rtol, atol)

    def array_is_eq(val1, val2):
        return (np.asarray(val1) == np.asarray(val2)).all()

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
            if isinstance(val, (np.ndarray, list))
            else (
                dict_eq
                if isinstance(val, dict)
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
            print(result)
    return the_same


def clip(
    val: Union[float, np.ndarray],
    val_min: Union[float, np.ndarray],
    val_max: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Clips (limits) val between val_min and val_max.
    This function wraps the numpy core umath minimum and maximum functions
    in order to avoid the standard numpy clip function, as described in:
    https://github.com/numpy/numpy/issues/14281

    Handles scalars using built-ins.

    Parameters
    ----------
    val:
        The value to be clipped.
    val_min:
        The minimum value.
    val_max:
        The maximum value.

    Returns
    -------
    The clipped values.
    """
    if isinstance(val, np.ndarray):
        np.core.umath.clip(val, val_min, val_max, out=val)
    else:
        val = val_min if val < val_min else val_max if val > val_max else val
    return val


def flatten_iterable(iters):
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
        if isinstance(_iter, Iterable) and not isinstance(_iter, (str, bytes, dict)):
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
    """
    if num_rep <= 1:
        raise NotImplementedError("Not implemented for less than 2 repeat elements")
    n = num_rep - 1
    m = arr[:-1] == arr[1:]
    return np.flatnonzero(np.convolve(m, np.ones(n, dtype=int)) == n) - n + 1


def slope(arr: np.ndarray) -> float:
    """Calculate gradient of a 2x2 point array"""
    b = arr[1, 0] - arr[0, 0]
    return np.inf if b == 0 else (arr[1, 1] - arr[0, 1]) / b


def yintercept(arr: np.ndarray) -> Tuple[float]:
    """Calculate the y intercept and gradient of an array"""
    s = slope(arr)
    return arr[0, 1] - s * arr[0, 0], s


# ======================================================================================
# Coordinate system transformations
# ======================================================================================


def cartesian_to_polar(
    x: np.ndarray, z: np.ndarray, x_ref: float = 0.0, z_ref: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
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
) -> Tuple[np.ndarray, np.ndarray]:
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
    """
    full_dirname = name.rsplit("/", 1)
    dirname = "." if len(full_dirname[0]) == 0 else full_dirname[0]

    try:
        mod_files = [
            file for file in listdir(dirname) if file.startswith(full_dirname[1])
        ]
    except FileNotFoundError:
        raise FileNotFoundError(f"Can't find module file '{name}'") from None

    if len(mod_files) == 0:
        raise FileNotFoundError(f"Can't find module file '{name}'")

    requested = full_dirname[1] if full_dirname[1] in mod_files else mod_files[0]

    if len(mod_files) > 1:
        bluemira_warn(
            "{}{}".format(
                f"Multiple files start with '{full_dirname[1]}'\n",
                f"Assuming module is '{requested}'",
            )
        )

    mod_file = f"{dirname}/{requested}"

    name, ext = requested.rsplit(".", 1) if "." in requested else (requested, "")
    if ext not in imp_mach.SOURCE_SUFFIXES:
        if ext and not ext.startswith("."):
            ext = f".{ext}"
        n_suffix = True
        imp_mach.SOURCE_SUFFIXES.append(ext)
    else:
        n_suffix = False

    try:
        spec = imp_u.spec_from_file_location(name, mod_file)
        module = imp_u.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ModuleNotFoundError:
        raise
    except (AttributeError, ImportError, SyntaxError) as err:
        raise ImportError(f"File '{mod_files[0]}' is not a module") from err

    if n_suffix:
        imp_mach.SOURCE_SUFFIXES.pop()

    return module


def get_class_from_module(name: str, default_module: str = "") -> Type:
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


def list_array(list_: Any) -> np.ndarray:
    """
    Always returns a numpy array
    Can handle int, float, list, np.ndarray

    Parameters
    ----------
    list_:
        The value to convert into a numpy array.

    Returns
    -------
    he value as a numpy array.

    Raises
    ------
    TypeError
        If the value cannot be converted to a numpy array.
    """
    if isinstance(list_, list):
        return np.array(list_)
    if isinstance(list_, np.ndarray):
        try:  # This catches the odd np.array(8) instead of np.array([8])
            len(list_)
        except TypeError:
            return np.array([list_])
        else:
            return list_
    elif is_num(list_):
        return np.array([list_])
    raise TypeError("Could not convert input type to list_array to a np.array.")


def array_or_num(array: Any) -> Union[np.ndarray, float]:
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
        return float(array)
    if isinstance(array, np.ndarray):
        return array
    raise TypeError


def deprecation_wrapper(
    message: Union[Optional[Callable[[Any], Any]], Optional[str]]
) -> Callable[[Any], Any]:
    """Deprecate any callable.

    Parameters
    ----------
    message:
        The callable to deprecate or the message to show
    """

    def _decorate(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
        """Deprecate any callable.

        Parameters
        ----------
        func:
            The callable to deprecate
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
