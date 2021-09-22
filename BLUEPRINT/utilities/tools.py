# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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
Generic miscellaneous tools, including some amigo port-overs
"""
from functools import partial
import numpy as np
import operator
import string
from scipy.interpolate import griddata, interp1d
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import cdist
from itertools import permutations
from json.encoder import _make_iterencode
import nlopt
import re
from os import listdir
from json import JSONEncoder, JSONDecoder
from collections import OrderedDict
from collections.abc import Mapping, Iterable
from typing import List, Union
from unittest.mock import patch
from importlib import util as imp_u, import_module as imp

from bluemira.base.constants import ABS_ZERO_C, ABS_ZERO_K, E_IJK, E_IJ, E_I
from bluemira.base.parameter import Parameter
from bluemira.base.look_and_feel import bluemira_print, bluemira_warn

from BLUEPRINT.geometry.geomtools import lengthnorm


CROSS_P_TOL = 1e-14  # Cross product tolerance


def levi_civita_tensor(dim=3):
    """
    N dimensional Levi-Civita Tensor.

    For dim=3 this looks like:

    e_ijk = np.zeros((3, 3, 3))
    e_ijk[0, 1, 2] = e_ijk[1, 2, 0] = e_ijk[2, 0, 1] = 1
    e_ijk[0, 2, 1] = e_ijk[2, 1, 0] = e_ijk[1, 0, 2] = -1

    Parameters
    ----------
    dim: int
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


class NumpyJSONEncoder(JSONEncoder):
    """
    Une lacune ennuyante dans json...
    """

    def default(self, obj):
        """
        Override the JSONEncoder default object handling behaviour for np.arrays.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    @staticmethod
    def floatstr(_floatstr, obj, *args, **kwargs):
        """
        Awaiting python bugs:

        https://github.com/python/cpython/pull/13233
        https://bugs.python.org/issue36841
        https://bugs.python.org/issue42434
        https://bugs.python.org/issue31466
        """
        if isinstance(obj, Parameter):
            obj = obj.value
        return _floatstr(obj, *args, **kwargs)

    def iterencode(self, o, _one_shot=False):
        """
        Patch iterencode type checking
        """
        with patch("json.encoder._make_iterencode", new=_patcher):
            return super().iterencode(o, _one_shot=_one_shot)


def _patcher(markers, _default, _encoder, _indent, _floatstr, *args, **kwargs):
    """
    Modify the json encoder to be less strict on
    type checking.
    Pythons built in types (float, int) have __repr__ written in c
    and json encoder doesn't yet allow custom type checking

    For example
    p = Parameter(var='hi', value=1, source='here')
    repr(p) == '1' # True
    isinstance(p, int) # True
    int.__repr__(p) # TypeError

    Currently there is a comment in the _make_iterencode function that
    calls itself a hack therefore this is ok...
    """
    _floatstr = partial(NumpyJSONEncoder.floatstr, _floatstr)
    kwargs["_intstr"] = repr
    return _make_iterencode(
        markers, _default, _encoder, _indent, _floatstr, *args, **kwargs
    )


class CommentJSONDecoder(JSONDecoder):
    """
    Decode JSON with comments

    Notes
    -----
    Regex does the following for comments:

        - starts with // followed by most chr (not ")
        - if not followed by " and any of (whitespace , }) and \\n

    and removes extra commas from the end of dict like objects
    """

    comments = re.compile(r'[/]{2}(\s*\w*[$-/:-?{-~!^_`\[\]]*)*(?!["]\s*[,]*[\}]*\n)')
    comma = re.compile(r"[,]\n\s*[\}]")

    def decode(self, s, *args, **kwargs):
        """Return the Python representation of ``s`` (a ``str`` instance
        containing a JSON document).
        """
        s = self.comma.sub("}", self.comments.sub("", s))
        s = self.comma.sub("}", self.comments.sub("", s)).strip()
        if s.endswith(","):
            s = s[:-1] + "}"

        return super().decode(s, *args, **kwargs)


class PowerLawScaling:
    """
    Simple power law scaling object, of the form:

    \t:math:`c~\\pm~cerr \\times {a_{1}}^{n1\\pm err1}{a_{2}}^{n2\\pm err2}...`

    if cerr is specified, or of the form:

    \t:math:`ce^{\\pm cexperr} \\times {a_{1}}^{n1\\pm err1}{a_{2}}^{n2\\pm err2}...`

    Parameters
    ----------
    c: float
        The constant of the equation
    cerr: float
        The error on the constant
    cexperr: Union[float, None]
        The exponent error on the constant (cannot be specified with cerr)
    exponents: Union[np.array, List, None]
        The ordered list of exponents
    err: Union[np.array, List, None]
        The ordered list of errors of the exponents
    """  # noqa (W505)

    def __init__(self, c=1, cerr=0, cexperr=None, exponents=None, err=None):
        self._len = len(exponents)
        self.c = c
        self.cerr = cerr
        self.cexperr = cexperr
        self.exponents = np.array(exponents)
        self.errors = np.array(err)

    def __call__(self, *args):
        """
        Call the PowerLawScaling object for a set of arguments.
        """
        if len(args) != len(self):
            raise ValueError(
                "Number of arguments should be the same as the "
                f"power law length. {len(args)} != {len(self)}"
            )
        return self.calculate(*args)

    def calculate(self, *args, exponents=None):
        """
        Call the PowerLawScaling object for a set of arguments.
        """
        if exponents is None:
            exponents = self.exponents
        return self.c * np.prod(np.power(args, exponents))

    def error(self, *args):
        """
        Calculate the error of the PowerLawScaling for a set of arguments.
        """
        if self.cexperr is None:
            c = [(self.c + self.cerr) / self.c, (self.c - self.cerr) / self.c]
        else:
            if self.cerr != 0:
                bluemira_warn("PowerLawScaling object overspecified, ignoring cerr.")
            c = [np.exp(self.cexperr), np.exp(-self.cexperr)]
        up = max(c) * self.calculate(*args, exponents=self.exponents + self.errors)
        down = min(c) * self.calculate(*args, exponents=self.exponents - self.errors)
        return self.calculate(*args), min(down, up), max(down, up)

    def __len__(self):
        """
        Get the length of the PowerLawScaling object.
        """
        return self._len


def set_random_seed(seed_number: int):
    """
    Sets the random seed number in numpy and NLopt. Useful when repeatable
    results are desired in Monte Carlo methods and stochastic optimisation
    methods.

    Parameters
    ----------
    seed_number: int
        The random seed number, preferably a very large integer
    """
    np.random.seed(seed_number)
    nlopt.srand(seed_number)


def latin_hypercube_sampling(dimensions: int, samples: int):
    """
    Classic Latin Hypercube sampling function

    Parameters
    ----------
    dimensions: int
        The number of design dimensions
    samples: int
        The number of samples points within the dimensions

    Returns
    -------
    lhs: np.array(samples, dimensions)
        The array of 0-1 normed design points

    Notes
    -----
    Simon's rule of thumb for a good number of samples was that
    samples >= 3**dimensions
    """
    intervals = np.linspace(0, 1, samples + 1)

    r = np.random.rand(samples, dimensions)
    a = intervals[:samples]
    b = intervals[1 : samples + 1]

    points = np.zeros((samples, dimensions))

    for j in range(dimensions):
        points[:, j] = r[:, j] * (b - a) + a

    lhs = np.zeros((samples, dimensions))

    for j in range(dimensions):
        order = np.random.permutation(range(samples))
        lhs[:, j] = points[order, j]

    return lhs


def expand_nested_list(*lists):
    """
    Expands a nested iterable structure, flattening it into one iterable

    Parameters
    ----------
    lists: set of Iterables
        The object(s) to de-nest

    Returns
    -------
    expanded: list
        The fully flattened list of iterables
    """
    expanded = []
    for obj in lists:
        if isinstance(obj, Iterable):
            for o in obj:
                expanded.extend(expand_nested_list(o))
        else:
            expanded.append(obj)
    return expanded


def map_nested_dict(obj, function):
    """
    Wendet eine Funktion auf ein verschachteltes Wörterbuch an

    Parameters
    ----------
    obj: dict
        Nested dictionary object to apply function to
    function: callable
        Function to apply to all non-dictionary objects in dictionary.

    Note
    ----
    In place modification of the dict
    """
    for k, v in obj.items():
        if isinstance(v, Mapping):
            map_nested_dict(v, function)
        else:
            obj[k] = function(v)


def compare_dicts(d1, d2, almost_equal=False, verbose=True):
    """
    Compares two dictionaries. Will print information about the differences
    between the two to the console. Dictionaries are compared by length, keys,
    and values per common keys

    Parameters
    ----------
    d1: dict
        The reference dictionary
    d2: dict
        The dictionary to be compared with the reference
    almost_equal: bool (default = False)
        Whether or not to use np.isclose and np.allclose for numbers and arrays
    verbose: bool (default = True)
        Whether or not to print to the console

    Returns
    -------
    the_same: bool
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
        return compare_dicts(value_1, value_2, almost_equal, verbose)

    if almost_equal:
        array_eq, num_eq = np.allclose, np.isclose
    else:
        array_eq, num_eq = lambda val1, val2: (val1 == val2).all(), operator.eq

    # Map the comparison functions to the keys based on the type of value in d1.
    comp_map = {
        key: array_eq
        if isinstance(val, np.ndarray)
        else dict_eq
        if isinstance(val, dict)
        else num_eq
        if is_num(val)
        else operator.eq
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


def get_max_PF(coil_dict):  # noqa (N802)
    """
    Returns maximum external radius of the largest PF coil
    takes a nova ordered dict of PFcoils
    """
    x = []
    for _, coil in coil_dict.items():
        try:  # New with equilibria
            x.append(coil.x + coil.dx)
        except AttributeError:  # Old nova
            x.append(coil["x"] + coil["dx"] / 2)
    return max(x)


def furthest_perp_point(p1, p2, point_array):
    """
    Returns arg of furthest point from vector p2-p1 in point_array
    """
    v = p2 - p1
    d = v / np.sqrt(v.dot(v))
    dot = np.dot((point_array - p2), d) * d[:, None]
    xyz = p2 + dot.T
    perp_vec = xyz - point_array
    perp_mags = np.linalg.norm(perp_vec, axis=1)
    n = np.argmax(perp_mags)
    return n, perp_mags[n]


def furthest_point_arg(point, loop, coords=None, closest=False):
    """
    Pode ser que esta funcionando mas tenho que confirmar.
    """
    if coords is None:
        coords = ["x", "z"]
    pnts = np.array([loop[coords[0]], loop[coords[1]]]).T
    pnt = np.array([point[0], point[1]]).reshape(2, 1).T
    distances = cdist(pnts, pnt, "euclidean")
    if closest:
        return np.argmin(distances)

    return np.argmax(distances)


def innocent_smoothie(x, y, n=500, s=0):
    """
    Pillaged from S. McIntosh geom.py "rzSLine" and modded
    sharing the spline love :)
    """
    length_norm = lengthnorm(x, y)
    n = int(n)
    l_interp = np.linspace(0, 1, n)
    if s == 0:
        x = interp1d(length_norm, x)(l_interp)
        y = interp1d(length_norm, y)(l_interp)
    else:
        x = UnivariateSpline(length_norm, x, s=s)(l_interp)
        y = UnivariateSpline(length_norm, y, s=s)(l_interp)
    return x, y


def ellipse(a, b, n=100):
    """
    Calculates an ellipse shape

    Parameters
    ----------
    a: float
        Ellipse major radius
    b: float
        Ellipse minor radius
    n: int
        The number of points in the ellipse
    \t:math: `y=\\pm\\sqrt{\\frac{a^2-x^2}{\\kappa^2}}`
    """
    k = a / b
    x = np.linspace(-a, a, n)
    y = ((a ** 2 - x ** 2) / k ** 2) ** 0.5
    x = np.append(x, x[:-1][::-1])
    y = np.append(y, -y[:-1][::-1])
    return x, y


def is_num(s):
    """
    Determines whether or not the input is a number

    Parameters
    ----------
    s: unknown type
        The input which we need to determine is a number or not

    Returns
    -------
    num: bool
        Whether or not the input is a number
    """
    if s is True or s is False:
        return False
    if s is np.nan:
        return False
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def findnoisylocals(f, x_bins=50, mode="min"):
    """
    Esta función encuentra los puntos máximos en cada basura!
    """
    n = len(f)
    bin_size = round(n / x_bins)
    y_bins = [f[i : i + bin_size] for i in range(0, n, bin_size)]
    localm = []
    localmidx = []
    for i, y_bin in enumerate(y_bins):
        if mode == "max":
            localm.append(np.max(y_bin))
            localmidx.append(np.argmax(y_bin) + i * bin_size)
        else:
            localm.append(np.min(y_bin))
            localmidx.append(np.argmin(y_bin) + i * bin_size)
    return localmidx, localm


def discretise_1d(x, y, n, method="linear"):
    """
    Isso aqui é uma função para discretizar uma função 1D com n puntos. \n
    Se você quiser uma lista, pode pedir!
    """
    x = np.array(x)
    y = np.array(y)
    x_1d = np.linspace(x[0], x[-1], n)
    y_1d = griddata(x, y, xi=x_1d, method=method)
    return [x_1d, y_1d]


def delta(v2, v1ref):
    """
    Calculates the absolute relative difference between a new value and an old
    reference value.

    Parameters
    ----------
    v2: float
        The new value to compare to the old
    v1ref: float
        The old reference value

    Returns
    -------
    delta: float
        The absolute relative difference between v2 and v1ref
    """
    return abs((v2 - v1ref) / v1ref)


def perc_change(v2, v1ref, verbose=False):
    """
    Calculates the percentage difference between a new value and an old
    reference value

    Parameters
    ----------
    v2: float
        The new value to compare to the old
    v1ref: float
        The old reference value
    verbose: bool
        Whether or not to print information

    Returns
    -------
    perc_change: float
        The percentage difference between v2 and v1ref
    """
    perc = 100 * (v2 - v1ref) / abs(v1ref)
    if verbose:
        if perc < 0:
            change = "decrease"
        else:
            change = "increase"
        bluemira_print("This is a {0} % ".format(round(perc, 3)) + change)
        factor = v2 / v1ref
        if factor < 1:
            fchange = "lower"
        else:
            fchange = "higher"
        bluemira_print("This is " + fchange + f" by a factor of {factor:.2f}")
    return perc


def tokelvin(temp_in_celsius):
    """
    Convert a temperature in Celsius to Kelvin.

    Parameters
    ----------
    temp_in_celsius: Union[float, np.array, List[float]]
        The temperature to convert [°C]

    Returns
    -------
    temp_in_kelvin: Union[float, np.array]
        The temperature [K]
    """
    if (is_num(temp_in_celsius) and temp_in_celsius < ABS_ZERO_C) or np.any(
        np.less(temp_in_celsius, ABS_ZERO_C)
    ):
        raise ValueError("Negative temperature in K specified.")
    return array_or_num(list_array(temp_in_celsius) - ABS_ZERO_C)


def tocelsius(temp_in_kelvin):
    """
    Convert a temperature in Celsius to Kelvin.

    Parameters
    ----------
    temp_in_kelvin: Union[float, np.array, List[float]]
        The temperature to convert [K]

    Returns
    -------
    temp_in_celsius: Union[float, np.array]
        The temperature [°C]
    """
    if (is_num(temp_in_kelvin) and temp_in_kelvin < ABS_ZERO_K) or np.any(
        np.less(temp_in_kelvin, ABS_ZERO_K)
    ):
        raise ValueError("Negative temperature in K specified.")
    return array_or_num(list_array(temp_in_kelvin) + ABS_ZERO_C)


def kgm3togcm3(density):
    """
    Convert a density in kg/m3 to g/cm3

    Parameters
    ----------
    density : Union[float, np.array, List[float]]
        The density [kg/m3]

    Returns
    -------
    density_gcm3 : Union[float, np.array]
        The density [g/cm3]
    """
    if density is not None:
        return array_or_num(list_array(density) / 1000.0)


def gcm3tokgm3(density):
    """
    Convert a density in g/cm3 to kg/m3

    Parameters
    ----------
    density : Union[float, np.array, List[float]]
        The density [g/cm3]

    Returns
    -------
    density_kgm3 : Union[float, np.array]
        The density [kg/m3]
    """
    if density is not None:
        return array_or_num(list_array(density) * 1000.0)


def tomols(flow_in_pam3_s):
    """
    Convert a flow in Pa.m^3/s to a flow in mols.

    Parameters
    ----------
    flow_in_pam3_s: Union[float, np.array]
        The flow in Pa.m^3/s to convert

    Returns
    -------
    flow_in_mols: Union[float, np.array]
        The flow in mol/s

    Notes
    -----
    At 273.15 K for a diatomic gas
    """
    return flow_in_pam3_s / 2270


def to_Pam3_s(flow_in_mols):  # noqa (N802)
    """
    Convert a flow in Pa.m^3/s to a flow in mols.

    Parameters
    ----------
    flow_in_mols: Union[float, np.array]
        The flow in mol/s to convert

    Returns
    -------
    flow_in_pam3_s: Union[float, np.array]
        The flow in Pa.m^3/s

    Notes
    -----
    At 273.15 K for a diatomic gas
    """
    return flow_in_mols * 2270


def print_format_table():
    """
    Prints table of formatted text format options.
    """
    for style in range(0, 10):
        for fg in range(26, 38):
            s1 = ""
            for bg in range(38, 48):
                formatt = ";".join([str(style), str(fg), str(bg)])
                s1 += "\x1b[%sm %s \x1b[0m" % (formatt, formatt)
            print(s1)
        print("\n")


def _apply_rule(a_r, op, b_r):
    return {
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        "=": lambda a, b: a == b,
    }[op](float(a_r), float(b_r))


def _apply_rules(rule):
    if len(rule) == 3:
        return _apply_rule(*rule)
    if len(rule) == 5:
        return _apply_rule(*rule[:3]) and _apply_rule(*rule[2:])


def _split_rule(rule):
    return re.split("([<>=]+)", rule)


def nested_dict_search(odict, rules: Union[str, List[str]]):
    """
    Returns sub-set of nested dictionary which meet str rules for keys in each
    sub-dict.
    Use-case: R.PF.coil searching
    """
    r = []
    if isinstance(rules, str):
        rules = [rules]  # Handles single input if no list
    for rule in rules:
        r.append(_split_rule(rule))
    sub = OrderedDict()
    for n, c in odict.items():
        rules = [[c[i] if i in ["x", "z"] else i for i in j] for j in r]
        if all(_apply_rules(rule) for rule in rules):
            sub[n] = c
    if len(sub) == 0:
        return None
    return sub


def clip(val, val_min, val_max):
    """
    Clips (limits) val between val_min and val_max.
    This function wraps the numpy core umath minimum and maximum functions
    in order to avoid the standard numpy clip function, as described in:
    https://github.com/numpy/numpy/issues/14281

    Handles scalars using built-ins.

    Parameters
    ----------
    val: scalar or array
        The value to be clipped.
    val_min: scalar or array
        The minimum value.
    val_max: scalar or array
        The maximum value.

    Returns
    -------
    clipped_val: scalar or array
        The clipped values.
    """
    if isinstance(val, np.ndarray):
        np.core.umath.clip(val, val_min, val_max, out=val)
    else:
        val = val_min if val < val_min else val_max if val > val_max else val
    return val


def maximum(val, val_min):
    """
    Gets the maximum of val and val_min.
    This function wraps the numpy core umath maximum function
    in order to avoid the standard numpy clip function, as described in:
    https://github.com/numpy/numpy/issues/14281

    Handles scalars using built-ins.

    Parameters
    ----------
    val: scalar or array
        The value to be floored.
    val_min: scalar or array
        The minimum value.

    Returns
    -------
    maximum_val: scalar or array
        The maximum values.
    """
    if isinstance(val, np.ndarray):
        np.core.umath.maximum(val, val_min, out=val)
    else:
        val = val_min if val < val_min else val
    return val


def asciistr(length):
    """
    Get a string of characters of desired length.

    Current max is 52 characters

    Parameters
    ----------
    length: int
        number of characters to return

    Returns
    -------
    str of length specified

    """
    if length > 52:
        raise ValueError("Unsupported string length")

    return string.ascii_letters[:length]


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

    def norm(self, ix, axis=0):
        """
        Emulates some of the functionality of np.linalg.norm for 2D arrays.

        Specifically:
        np.linalg.norm(ix, axis=0)
        np.linalg.norm(ix, axis=1)

        For optimum speed and customisation use np.einsum modified for your use case.

        Parameters
        ----------
        ix: np.array
            Array to perform norm on
        axis: int
            axis for the norm to occur on

        Returns
        -------
        np.array

        """
        try:
            return np.sqrt(np.einsum(self.norm_strs[axis], ix, ix))
        except IndexError:
            raise ValueError("matrices dimensions >2d Unsupported")

    def dot(self, ix, iy, out=None):
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
        ix: np.array
            First array
        iy: np.array
            Second array
        out: np.array
            output array for inplace dot product

        Returns
        -------
        np.array

        """
        # Ordered hopefully by most used
        if ix.ndim == 2 and iy.ndim == 2:
            out_str = self.dot_2x2
        elif ix.ndim > 2 and iy.ndim > 2:
            ix_str = asciistr(ix.ndim - 1)
            iy_str = asciistr(iy.ndim - 2)
            out_str = self.dot_nxn.format(ix_str, iy_str, ix_str)
        elif ix.ndim < 2 and iy.ndim == 2:
            out_str = self.dot_1x2
        elif ix.ndim >= 2 and iy.ndim < 2:
            ix_str = asciistr(ix.ndim - 1)
            out_str = self.dot_nx1.format(ix_str, ix_str)
        elif iy.ndim >= 2 or ix.ndim == 2:
            raise ValueError(
                f"Undefined behaviour ix.shape:{ix.shape}, iy.shape:{iy.shape}"
            )
        else:
            out_str = self.dot_1x1

        return np.einsum(out_str, ix, iy, out=out)

    def cross(self, ix, iy, out=None):
        """
        A row-wise cross product of a 2D matrices of vectors.

        This function mirrors the properties of np.cross
        such as vectors of 2 or 3 elements. 1D is also accepted
        but just do x * y.
        Only 7D has similar orthogonal properties above 3D.

        For optimum speed and customisation use np.einsum modified for your use case.

        Parameters
        ----------
        ix: np.array
            1st array to cross
        iy: np.array
            2nd array to cross
        out: np.array
            output array for inplace cross product

        Returns
        -------
        np.array (ix.shape)

        Raises
        ------
        ValueError
            If the dimensions of the cross product are > 3

        """
        dim = ix.shape[-1] - 1 if ix.ndim > 1 else 0

        try:
            return np.einsum(self.cross_strs[dim], self.cross_lcts[dim], ix, iy, out=out)
        except IndexError:
            raise ValueError("Incompatible dimension for cross product")


# =====================================================
# Einsum string preallocation
# =====================================================
wrap = EinsumWrapper()

norm = wrap.norm
dot = wrap.dot
cross = wrap.cross
# ====================================================


def list_array(list_):
    """
    Always returns a numpy array
    Can handle int, float, list, np.ndarray

    Parameters
    ----------
    list_ : Any
        The value to convert into a numpy array.

    Returns
    -------
    result : np.ndarray
        The value as a numpy array.

    Raises
    ------
    TypeError
        If the value cannot be converted to a numpy array.
    """
    if isinstance(list_, list):
        return np.array(list_)
    elif isinstance(list_, np.ndarray):
        try:  # This catches the odd np.array(8) instead of np.array([8])
            len(list_)
            return list_
        except TypeError:
            return np.array([list_])
    elif is_num(list_):
        return np.array([list_])
    else:
        raise TypeError("Could not convert input type to list_array to a np.array.")


def array_or_num(array):
    """
    Always returns a numpy array or a float

    Parameters
    ----------
    array : Any
        The value to convert into a numpy array or number.

    Returns
    -------
    result : Union[np.ndarray, float]
        The value as a numpy array or number.

    Raises
    ------
    TypeError
        If the value cannot be converted to a numpy or number.
    """
    if is_num(array):
        return float(array)
    elif isinstance(array, np.ndarray):
        return array
    else:
        raise TypeError


def get_module(name):
    """
    Load module dynamically.

    Parameters
    ----------
    name: string
        Filename or python path (a.b.c) of module to import

    Returns
    -------
    output: module
        Loaded module

    """
    try:
        module = imp(name)
    except ImportError:
        module = _loadfromspec(name)
    bluemira_print(f"Loaded {module.__name__}")
    return module


def _loadfromspec(name):
    """
    Load module from filename.

    Parameters
    ----------
    name: string
        Filename of module to import

    Returns
    -------
    output: module
        Loaded module

    """
    full_dirname = name.rsplit("/", 1)
    dirname = "." if len(full_dirname[0]) == 0 else full_dirname[0]

    try:
        mod_files = [
            file for file in listdir(dirname) if file.startswith(full_dirname[1])
        ]
    except FileNotFoundError:
        raise FileNotFoundError("Can't find module file '{}'".format(name))

    if len(mod_files) == 0:
        raise FileNotFoundError("Can't find module file '{}'".format(name))

    requested = full_dirname[1] if full_dirname[1] in mod_files else mod_files[0]

    if len(mod_files) > 1:
        bluemira_warn(
            "{}{}".format(
                "Multiple files start with '{}'\n".format(full_dirname[1]),
                "Assuming module is '{}'".format(requested),
            )
        )

    mod_file = f"{dirname}/{requested}"

    try:
        spec = imp_u.spec_from_file_location(
            mod_file.rsplit("/")[-1].split(".")[0], mod_file
        )
        module = imp_u.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (AttributeError, ImportError):
        raise ImportError("File '{}' is not a module".format(mod_files[0]))

    return module


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
