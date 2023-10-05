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
Coil and coil grouping objects
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from operator import attrgetter
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Type, Union

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes

    from bluemira.equilibria.file import EQDSKInterface

import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.coils._coil import Coil, CoilType
from bluemira.equilibria.coils._field import CoilGroupFieldsMixin, CoilSetFieldsMixin
from bluemira.equilibria.coils._tools import (
    _get_symmetric_coils,
    check_coilset_symmetric,
    get_max_current,
)
from bluemira.equilibria.constants import I_MIN
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.plotting import CoilGroupPlotter
from bluemira.utilities.tools import flatten_iterable, yintercept


def symmetrise_coilset(coilset: CoilSet) -> CoilSet:
    """
    Symmetrise a CoilSet by converting any coils that are up-down symmetric about
    z=0 to SymmetricCircuits.

    Parameters
    ----------
    coilset:
        CoilSet to symmetrise

    Returns
    -------
    New CoilSet with SymmetricCircuits where appropriate
    """
    if not check_coilset_symmetric(coilset):
        bluemira_warn(
            "Symmetrising a CoilSet which is not purely symmetric about z=0. This can"
            " result in undesirable behaviour."
        )
    coilset = deepcopy(coilset)

    sym_stack = _get_symmetric_coils(coilset)
    counts = np.array(sym_stack, dtype=object).T[1]

    new_coils = []
    for coil, count in zip(coilset._coils, counts):
        if count == 1:
            new_coils.append(coil)
        elif count == 2:  # noqa: PLR2004
            if isinstance(coil, SymmetricCircuit):
                new_coils.append(coil)
            elif isinstance(coil, Coil):
                new_coils.append(SymmetricCircuit(coil))
            else:
                raise EquilibriaError(f"Unrecognised class {coil.__class__.__name__}")
        else:
            raise EquilibriaError("There are super-posed Coils in this CoilSet.")

    return CoilSet(*new_coils)


class CoilGroup(CoilGroupFieldsMixin):
    """
    Coil Grouping object

    Allow nested coils or groups of coils with access to the methods and properties
    in the same way to a vanilla Coil

    Parameters
    ----------
    coils:
        Coils and groups of Coils to group
    """

    __slots__ = ("_coils", "_pad_size")

    def __init__(self, *coils: Union[Coil, CoilGroup[Coil]]):
        if any(not isinstance(c, (Coil, CoilGroup)) for c in coils):
            raise TypeError("Not all arguments are a Coil or CoilGroup.")
        self._coils = coils
        self._pad_discretisation(self.__list_getter("_quad_x"))

    def __repr__(self):
        """
        Pretty print
        """
        coils_repr = "\n    "
        coils_repr += "".join(f"{c.__repr__()}\n    " for c in self._coils)
        coils_repr = coils_repr.replace("\n", "\n    ")
        return f"{type(self).__name__}({coils_repr[:-5]})"

    def n_coils(self, ctype: Optional[Union[str, CoilType]] = None) -> int:
        """
        Get number of coils

        Parameters
        ----------
        ctype:
            get number of coils of a specific type

        Returns
        -------
        Number of coils
        """
        if ctype is None:
            return len(self.x)

        if not isinstance(ctype, CoilType):
            ctype = CoilType[ctype]

        return Counter(self.ctype)[ctype]

    def plot(
        self,
        ax: Optional[Axes] = None,
        subcoil: bool = True,
        label: bool = False,
        force: Optional[Iterable] = None,
        **kwargs,
    ):
        """
        Plot a CoilGroup

        Parameters
        ----------
        ax:
            Matplotlib axis object
        subcoil:
            plot coil discretisations
        label:
            show coil labels on plot
        force:
            force arrows iterable
        kwargs:
            passed to matplotlib's Axes.plot
        """
        return CoilGroupPlotter(
            self, ax=ax, subcoil=subcoil, label=label, force=force, **kwargs
        )

    def fix_sizes(self):
        """
        Fix the sizes of coils in CoilGroup
        """
        self.__run_func("fix_size")

    def resize(self, currents: Union[float, List, np.ndarray]):
        """
        Resize coils based on their current if their size is not fixed
        """
        self.__run_func("resize", currents)

    def _resize(self, currents: Union[float, List, np.ndarray]):
        """
        Resize coils based on their current

        Notes
        -----
        Ignores any protections on their size
        """
        self.__run_func("_resize", currents)

    def __run_func(self, func: str, *args, **kwargs):
        """
        Runs a function with no outputs that exists on a coil or coilgroup

        This function aims to deal with the edge cases that are around nested coilgroups
        If kwargs are passed they will be passed to all function calls as is.
        If args are passed an attempt is made to push the right shaped argument to a
        given function.
        """
        if not args:
            for ff in self.__list_getter(func):
                ff(**kwargs)
        else:
            args = list(args)
            funclist = self.__list_getter(func)
            len_funclist = len(funclist)
            for no, arg in enumerate(args):
                if isinstance(arg, (float, int)):
                    args[no] = np.full(len_funclist, arg)
                elif len(arg) != len_funclist:
                    raise ValueError(
                        f"length of {arg} != number of coilgroups ({len_funclist})"
                    )
            for ff, *_args in zip(funclist, *args):
                ff(*_args, **kwargs)

    def add_coil(self, *coils: Union[Coil, CoilGroup[Coil]]):
        """Add coils to the coil group"""
        self._coils = (*self._coils, *coils)

    def remove_coil(self, *coil_name: str, _top_level: bool = True) -> Union[None, List]:
        """
        Remove coil from CoilGroup

        Parameters
        ----------
        coil_name:
            coil(s) to remove
        _top_level:
            FOR INTERNAL USE, flags if at top level of nested coilgroup stack

        Returns
        -------
        Removed names if not at top level of nested stack

        Notes
        -----
        If a nested coilgroup is empty it is also removed from the parent coilgroup

        """
        names = self.name

        if _top_level and (remainder := set(coil_name) - set(names)):
            raise EquilibriaError(f"Unknown coils {remainder}")

        c_names = []
        removed_names = []

        for c in self._coils:
            if isinstance(c, CoilGroup):
                removed_names.extend(c.remove_coil(*coil_name, _top_level=False))
            elif c.name in coil_name:
                c_names.append(c.name)

        to_remove = [names.index(c_n) for c_n in c_names]

        coils = np.array(self._coils, dtype=object)
        mask = np.full(coils.size, True, dtype=bool)
        mask[to_remove] = False

        for no, c in enumerate(self._coils):
            if c.name == []:
                # if a coil group is empty the name list will be empty
                mask[no] = False

        self._coils = tuple(coils[mask])

        if not _top_level:
            return removed_names + to_remove
        return None

    @classmethod
    def from_group_vecs(cls, eqdsk: EQDSKInterface):
        """
        Initialises an instance of CoilSet from group vectors.

        This has been implemented as a dict operation, because it will
        occur for eqdsks only.
        Future dict instantiation methods will likely differ, hence the
        confusing name of this method.
        """
        pfcoils = []
        cscoils = []
        passivecoils = []
        for i in range(eqdsk.ncoil):
            dx = eqdsk.dxc[i]
            dz = eqdsk.dzc[i]
            if abs(eqdsk.Ic[i]) < I_MIN:
                # Some eqdsk formats (e.g., CREATE) contain 'quasi-coils'
                # with currents very close to 0.
                # Catch these cases and make sure current is set to zero.
                passivecoils.append(
                    Coil(
                        eqdsk.xc[i],
                        eqdsk.zc[i],
                        current=0,
                        dx=dx,
                        dz=dz,
                        ctype="NONE",
                        control=False,
                    )
                )
            elif dx != dz:  # Rough and ready
                cscoils.append(
                    Coil(
                        eqdsk.xc[i],
                        eqdsk.zc[i],
                        current=eqdsk.Ic[i],
                        dx=dx,
                        dz=dz,
                        ctype="CS",
                    )
                )
            else:
                coil = Coil(
                    eqdsk.xc[i],
                    eqdsk.zc[i],
                    current=eqdsk.Ic[i],
                    dx=dx,
                    dz=dz,
                    ctype="PF",
                )
                coil.fix_size()  # Oh ja
                pfcoils.append(coil)

        coils = pfcoils
        coils.extend(cscoils)
        coils.extend(passivecoils)
        return cls(*coils)

    def to_group_vecs(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Output CoilGroup properties as numpy arrays

        Returns
        -------
        x:
            The x-positions of coils
        z:
            The z-positions of coils.
        dx:
            The coil size in the x-direction.
        dz:
            The coil size in the z-direction.
        currents:
            The coil currents.
        """
        return self.x, self.z, self.dx, self.dz, self.current

    def __list_getter(self, attr: str) -> List:
        """Get attributes from coils tuple"""
        return np.frompyfunc(attrgetter(attr), 1, 1)(self._coils)

    def __getter(self, attr: str) -> np.ndarray:
        """Get attribute from coils and convert to flattened numpy array"""
        arr = np.array([*flatten_iterable(self.__list_getter(attr))])
        arr.flags.writeable = False
        return arr

    def __quad_getter(self, attr: str) -> np.ndarray:
        """Get quadratures and autopad to create non ragged array"""
        _quad_list = self.__list_getter(attr)
        self._pad_discretisation(_quad_list)

        for i, d in enumerate(self._pad_size):
            if _quad_list[i].ndim > 1:
                pad = (*tuple((0, 0) for _ in range(_quad_list[i].ndim - 1)), (0, d))
            else:
                pad = (0, d)
            _quad_list[i] = np.pad(_quad_list[i], pad)

        return np.vstack(_quad_list)

    def __setter(
        self,
        attr: str,
        values: Union[CoilType, float, Iterable[Union[CoilType, float]]],
        dtype: Union[Type, None] = None,
    ):
        """Set attributes on coils"""
        values = np.atleast_1d(values)
        if dtype not in (None, object):
            values.dtype = np.dtype(dtype)
        no_val = values.size
        no = 0
        for coil in flatten_iterable(self._coils):
            end_no = no + coil.n_coils()
            if end_no > no_val:
                if no_val == 1:
                    setattr(coil, attr, np.repeat(values[0], end_no - no))
                elif isinstance(coil, Circuit):
                    setattr(coil, attr, np.repeat(values[-1], coil.n_coils()))
                else:
                    raise ValueError(
                        "The number of elements is less than the number of coils"
                    )
            else:
                setattr(coil, attr, values[no:end_no])
                no = end_no

    def __getitem__(self, item):
        """Get coils"""
        return self._find_coil(item)

    def __copy__(self):
        """Copy dunder method, needed because attribute setter fails for quadratures"""
        cls = self.__class__
        result = cls.__new__(cls)
        for k in self.__slots__:
            setattr(result, k, getattr(self, k))
        return result

    def __deepcopy__(self, memo):
        """
        Deepcopy dunder method, needed because attribute setter fails for
        quadratures
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k in self.__slots__:
            setattr(result, k, deepcopy(getattr(self, k), memo))
        result._einsum_str = self._einsum_str
        return result

    def _pad_discretisation(
        self,
        _to_pad: List[np.ndarray],
    ):
        """
        Convert quadrature list of array to rectangular arrays.
        Padding quadrature arrays with zeros to allow array operations
        on rectangular matricies.

        Parameters
        ----------
        _to_pad:
            x quadratures

        Notes
        -----
        Padding exists for coils with different discretisations or sizes within a
        coilgroup.
        There are a few extra calculations of the greens functions where padding
        exists in the :func:_combined_control method of CoilGroupFieldMixin.

        """
        all_len = np.array([q.shape[-1] for q in _to_pad])
        max_len = max(all_len)
        self._pad_size = max_len - all_len

        self._einsum_str = "...j, ...j -> ..."

    def _find_coil(self, name):
        """Find coil by name"""
        for c in self._coils:
            if isinstance(c, CoilGroup):
                try:
                    return c._find_coil(name)
                except KeyError:
                    pass
            elif c.name == name:
                return c

        raise KeyError(f"Coil '{name}' not found in Group")

    def _get_coiltype(self, ctype):
        """Find coil by type"""
        coils = []
        if isinstance(ctype, str):
            ctype = CoilType[ctype]
        for c in self._coils:
            if isinstance(c, CoilGroup):
                coils.extend(c._get_coiltype(ctype))
            elif c.ctype == ctype:
                coils.append(c)
        return coils

    def get_coiltype(self, ctype: Union[str, CoilType]):
        """Get coil by coil type"""
        return CoilGroup(*self._get_coiltype(ctype))

    def assign_material(self, ctype, j_max, b_max):
        """Assign material J and B to Coilgroup"""
        cg = self.get_coiltype(ctype)
        cg.j_max = j_max
        cg.b_max = b_max

    def get_max_current(self, max_current: float = np.infty) -> np.ndarray:
        """
        Get max currents

        If a max current argument is provided and the max current isn't set, the value
        will be as input.

        Parameters
        ----------
        max_current:
            max current value if j_max == nan

        Returns
        -------
        Maximum currents
        """
        return np.where(
            np.isnan(self.j_max) | ~self._flag_sizefix,  # or not
            max_current,
            get_max_current(self.dx, self.dz, self.j_max),
        )

    @property
    def name(self) -> List:
        """Get coil names"""
        return self.__getter("name").tolist()

    @property
    def x(self) -> np.ndarray:
        """Get coil x positions"""
        return self.__getter("x")

    @property
    def z(self) -> np.ndarray:
        """Get coil z positions"""
        return self.__getter("z")

    @property
    def position(self):
        """Get coil x, z positions"""
        return np.array([self.x, self.z])

    @property
    def ctype(self) -> List:
        """Get coil types"""
        return self.__getter("ctype").tolist()

    @property
    def dx(self) -> np.ndarray:
        """Get coil widths (half)"""
        return self.__getter("dx")

    @property
    def dz(self) -> np.ndarray:
        """Get coil heights (half)"""
        return self.__getter("dz")

    @property
    def current(self) -> np.ndarray:
        """Get coil currents"""
        return self.__getter("current")

    @property
    def j_max(self) -> np.ndarray:
        """Get coil max current density"""
        return self.__getter("j_max")

    @property
    def b_max(self) -> np.ndarray:
        """Get coil max field"""
        return self.__getter("b_max")

    @property
    def discretisation(self) -> np.ndarray:
        """Get coil discretisations"""
        return self.__getter("discretisation")

    @property
    def n_turns(self) -> np.ndarray:
        """Get coil number of turns"""
        return self.__getter("n_turns")

    @property
    def area(self) -> np.ndarray:
        """Get coil areas"""
        return self.__getter("area")

    @property
    def volume(self) -> np.ndarray:
        """Get coil volumes"""
        return self.__getter("volume")

    @property
    def x_boundary(self) -> np.ndarray:
        """Get coil x coordinate boundary"""
        xb = self.__getter("x_boundary")
        if self.n_coils() > 1:
            return xb.reshape(-1, 4)
        return xb

    @property
    def z_boundary(self) -> np.ndarray:
        """Get coil z coordinate boundary"""
        zb = self.__getter("z_boundary")
        if self.n_coils() > 1:
            return zb.reshape(-1, 4)
        return zb

    @property
    def _flag_sizefix(self) -> np.ndarray:
        """Get coil current radius"""
        return self.__getter("_flag_sizefix")

    @property
    def _current_radius(self) -> np.ndarray:
        """Get coil current radius"""
        return self.__getter("_current_radius")

    @property
    def _quad_x(self):
        """Get coil x quadratures"""
        return self.__quad_getter("_quad_x")

    @property
    def _quad_z(self):
        """Get coil z quadratures"""
        return self.__quad_getter("_quad_z")

    @property
    def _quad_dx(self):
        """Get coil dx quadratures"""
        return self.__quad_getter("_quad_dx")

    @property
    def _quad_dz(self):
        """Get coil dz quadratures"""
        return self.__quad_getter("_quad_dz")

    @property
    def _quad_weighting(self):
        """Get coil quadrature weightings"""
        return self.__quad_getter("_quad_weighting")

    @property
    def _quad_boundary(self):
        """Get coil quadrature boundaries"""
        return [*self.__list_getter("_quad_boundary")]

    @x.setter
    def x(self, values: Union[float, Iterable[float]]):
        """Set coil x positions"""
        self.__setter("x", values)

    @z.setter
    def z(self, values: Union[float, Iterable[float]]):
        """Set coil z positions"""
        self.__setter("z", values)

    @position.setter
    def position(self, values: np.ndarray):
        """Set coil positions"""
        self.__setter("x", values[0])
        self.__setter("z", values[1])

    @ctype.setter
    def ctype(self, values: Union[CoilType, Iterable[CoilType]]):
        """Set coil types"""
        self.__setter("ctype", values, dtype=object)

    @dx.setter
    def dx(self, values: Union[float, Iterable[float]]):
        """Set coil dx sizes"""
        self.__setter("dx", values)

    @dz.setter
    def dz(self, values: Union[float, Iterable[float]]):
        """Set coil dz sizes"""
        self.__setter("dz", values)

    @current.setter
    def current(self, values: Union[float, Iterable[float]]):
        """Set coil currents"""
        self.__setter("current", values)

    @j_max.setter
    def j_max(self, values: Union[float, Iterable[float]]):
        """Set coil max current densities"""
        self.__setter("j_max", values)

    @b_max.setter
    def b_max(self, values: Union[float, Iterable[float]]):
        """Set coil max fields"""
        self.__setter("b_max", values)

    @discretisation.setter
    def discretisation(self, values: Union[float, Iterable[float]]):
        """Set coil discretisations"""
        self.__setter("discretisation", values)
        self._pad_discretisation(self.__list_getter("_quad_x"))

    @n_turns.setter
    def n_turns(self, values: Union[float, Iterable[float]]):
        """Set coil number of turns"""
        self.__setter("n_turns", values)


class Circuit(CoilGroup):
    """
    A CoilGroup where all coils have the same current

    Parameters
    ----------
    coils:
        coils in circuit
    current:
        The current value, if not provided the first coil current is used
    """

    __slots__ = ()

    def __init__(
        self, *coils: Union[Coil, CoilGroup[Coil]], current: Optional[float] = None
    ):
        super().__init__(*coils)
        self.current = self._get_current() if current is None else current

    def _get_current(self):
        current = self._coils[0].current
        if isinstance(current, Iterable):
            current = current[0]
        return current

    def add_coil(self, *coils: Union[Coil, CoilGroup[Coil]]):
        """
        Add coil to circuit forcing the same current
        """
        super().add_coil(coils)
        self.current = self._get_current()

    @CoilGroup.current.setter
    def current(self, values: Union[float, Iterable[float]]):
        """
        Set current for circuit
        """
        if isinstance(values, Iterable):
            # Force the same value of current for all coils
            values = values[0]
        self._CoilGroup__setter("current", values)


class SymmetricCircuit(Circuit):
    """
    A Circuit with positional symmetry

    Parameters
    ----------
    coils:
        2 coil or coil group objects to symmetrise in a circuit


    Notes
    -----
    Although two groups can be defined any movement of the coils is
    achieved by offsets to the mean position of the coilgroups.

    Currently only symmetric about z = 0 see gh issue #210
    """

    __slots__ = ("_symmetry_line", "_shift", "sym_mat", *CoilGroup.__slots__)

    def __init__(
        self,
        *coils: Union[Coil, CoilGroup[Coil]],
    ):
        symmetry_line: Union[Tuple, np.ndarray] = ((0, 0), (1, 0))

        if len(coils) == 1:
            coils = (coils[0], deepcopy(coils[0]))
        if len(coils) != 2:  # noqa: PLR2004
            raise EquilibriaError(
                f"Wrong number of coils to create a {type(self).__name__}"
            )

        super().__init__(*coils)

        self.modify_symmetry(symmetry_line)
        diff = self._symmetrise()
        self._coils[1].x = self._coils[1].x - diff[0]
        self._coils[1].z = self._coils[1].z - diff[1]

    def modify_symmetry(self, symmetry_line: np.ndarray):
        """
        Create a unit vector for the symmetry of the circuit

        Parameters
        ----------
        symmetry_line:
            two points making a symmetry line [[float, float], [float, float]]
        """
        self._symmetry_line = (
            np.array(symmetry_line)
            if isinstance(symmetry_line, tuple)
            else symmetry_line
        )
        self._symmetry_matrix()

    def _symmetry_matrix(self):
        """
        Symmetry matrix

        .. math::

            \\frac{1}{1 + m} \\left[ {\\begin{array}{cc}
                                        1 - m^2 & 2m \\\\
                                        2m & -(1 - m^2) \\\\
                                      \\end{array} } \\right]
        """
        self._shift, grad = yintercept(self._symmetry_line)
        grad2 = grad**2
        mgrad2 = 1 - grad2
        gradsq = 2 * grad
        self.sym_mat = 1 / (1 + grad2) * np.array([[mgrad2, gradsq], [gradsq, -mgrad2]])

    def _symmetrise(self) -> np.ndarray:
        """
        Calculate the change in position to the symmetric coil,
        twice the distance to the line of symmetry.
        """
        cp = self._get_group_centre()
        cp[1] -= self._shift
        mirror = np.dot(self.sym_mat, cp.T)
        mirror[1] += self._shift
        return np.array([np.mean(self._coils[1].x), np.mean(self._coils[1].z)]) - mirror

    @Circuit.x.setter
    def x(self, new_x: float):
        """
        Set x coordinate of each coil
        """
        if isinstance(new_x, np.ndarray):
            new_x = np.mean(new_x[0])
        self._coils[0].x = self._coils[0].x + new_x - self._get_group_x_centre()
        self._coils[1].x = self._coils[1].x - self._symmetrise()[0]

    @Circuit.z.setter
    def z(self, new_z: float):
        """
        Set z coordinate of each coil
        """
        if isinstance(new_z, np.ndarray):
            new_z = np.mean(new_z[0])
        self._coils[0].z = self._coils[0].z + new_z - self._get_group_z_centre()
        self._coils[1].z = self._coils[1].z + self._symmetrise()[1]

    def _get_group_x_centre(self) -> np.ndarray:
        """Get the x centre of the first coil group"""
        return np.mean(self._coils[0].x)

    def _get_group_z_centre(self) -> np.ndarray:
        """Get the z centre of the first coil group"""
        return np.mean(self._coils[0].z)

    def _get_group_centre(self) -> np.ndarray:
        """Get the centre of the first coil group"""
        return np.array([self._get_group_x_centre(), self._get_group_z_centre()])


class CoilSet(CoilSetFieldsMixin, CoilGroup):
    """
    CoilSet is a CoilGroup with the concept of control coils

    A CoilSet will return the total volume and area of the coils
    and the respective psi or field calculations can optionally sum over the coils
    or only sum over the control coils.

    By default all coils are controlled

    Parameters
    ----------
    coils:
        The coils to be added to the set
    control_names:
        List of coil names to be controlled

    """

    __slots__ = ("_control", "_control_ind", *CoilGroup.__slots__)

    def __init__(
        self,
        *coils: Union[Coil, CoilGroup[Coil]],
        control_names: Optional[Union[List, bool]] = None,
    ):
        super().__init__(*coils)
        self.control = control_names

    def remove_coil(self, *coil_name: str, _top_level: bool = True) -> Union[None, List]:
        """
        Remove coil from CoilSet
        """
        super().remove_coil(*coil_name, _top_level=_top_level)
        self.control = list(set(self.control) & set(self.name))

    @property
    def control(self) -> List:
        """Get control coil names"""
        return self._control

    @control.setter
    def control(self, control_names: Optional[Union[List, bool]] = None):
        """Set control coils"""
        names = self.name
        if isinstance(control_names, List):
            self._control_ind = [names.index(c) for c in control_names]
        elif control_names or control_names is None:
            self._control_ind = np.arange(len(names)).tolist()
        else:
            self._control_ind = []
        self._control = [names[c] for c in self._control_ind]

    def get_control_coils(self):
        """Get Control coils"""
        coils = []
        for c in self._coils:
            names = c.name
            if isinstance(names, List):
                # is subset of list
                if isinstance(c, Circuit) and any(n in self.control for n in names):
                    coils.append(c)
                else:
                    coils.extend(c.get_control_coils()._coils)
            elif names in self.control:
                coils.append(c)
        return CoilSet(*coils)

    def get_coiltype(self, ctype):  # noqa: PLR6301
        """Get coils by coils type"""
        return CoilSet(*super()._get_coiltype(ctype))

    @property
    def area(self) -> float:
        """
        Cross sectional area of CoilSet
        """
        return np.sum(super().area)

    @property
    def volume(self) -> float:
        """
        Volume of Coilset
        """
        return np.sum(super().volume)

    def _sum(
        self, output: np.ndarray, sum_coils: bool = False, control: bool = False
    ) -> np.ndarray:
        """
        Get responses of coils optionally only control and/or sum over the responses

        Parameters
        ----------
        output:
            Output of calculation
        sum_coils:
            sum over coils
        control:
            operations on control coils only

        Returns
        -------
        Summed response output
        """
        ind = self._control_ind if control else slice(None)

        return np.sum(output[..., ind], axis=-1) if sum_coils else output[..., ind]
