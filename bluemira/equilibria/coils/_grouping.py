# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Coil and coil grouping objects
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from operator import attrgetter
from typing import TYPE_CHECKING

import numpy as np

from bluemira.base.constants import CoilType
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.coils._coil import Coil
from bluemira.equilibria.coils._field import CoilGroupFieldsMixin, CoilSetFieldsMixin
from bluemira.equilibria.coils._tools import (
    _get_symmetric_coils,
    check_coilset_symmetric,
    get_max_current,
)
from bluemira.equilibria.constants import I_MIN
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.plotting import CoilGroupPlotter
from bluemira.utilities.tools import flatten_iterable, yintercept

if TYPE_CHECKING:
    import numpy.typing as npt
    from eqdsk import EQDSKInterface
    from matplotlib.axes import Axes


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
    :
        New CoilSet with SymmetricCircuits where appropriate

    Raises
    ------
    EquilibriaError
        Superposition of coils or unrecognised type
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
    for coil, count in zip(coilset._coils, counts, strict=False):
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

    __slots__ = (
        "_Bx_analytic",
        "_Bz_analytic",
        "_coils",
        "_pad_size",
        "_psi_analytic",
    )

    def __init__(
        self,
        *coils: Coil | CoilGroup,
        psi_analytic: bool = False,
        Bx_analytic: bool = True,
        Bz_analytic: bool = True,
    ):
        if any(not isinstance(c, Coil | CoilGroup) for c in coils) or not coils:
            raise TypeError("Not all arguments are a Coil or CoilGroup.")
        self._coils = coils
        self._pad_discretisation(self.__list_getter("_quad_x"))
        super().__init__(
            psi_analytic=psi_analytic, Bx_analytic=Bx_analytic, Bz_analytic=Bz_analytic
        )

    def __repr__(self):
        """
        Pretty print
        """  # noqa: DOC201
        coils_repr = "\n    "
        coils_repr += "".join(f"{c!r}\n    " for c in self._coils)
        coils_repr = coils_repr.replace("\n", "\n    ")
        return f"{type(self).__name__}({coils_repr[:-5]})"

    def n_coils(self, ctype: str | CoilType | None = None) -> int:
        """
        Get number of coils

        Parameters
        ----------
        ctype:
            get number of coils of a specific type

        Returns
        -------
        :
            Number of coils
        """
        if ctype is None:
            return len(self.x)

        if not isinstance(ctype, CoilType):
            ctype = CoilType[ctype]

        return Counter(self.ctype)[ctype]

    def plot(
        self,
        ax: Axes | None = None,
        *,
        subcoil: bool = True,
        label: bool = False,
        force: Iterable | None = None,
        **kwargs,
    ) -> CoilGroupPlotter:
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
        """  # noqa: DOC201
        if self.ctype == CoilType.DUM:
            # Do not plot if it is a dummy coil
            return None
        return CoilGroupPlotter(
            self, ax=ax, subcoil=subcoil, label=label, force=force, **kwargs
        )

    def fix_sizes(self):
        """
        Fix the sizes of coils in CoilGroup
        """
        self.__run_func("fix_size")

    def resize(self, currents: float | list | np.ndarray):
        """
        Resize coils based on their current if their size is not fixed
        """
        self.__run_func("resize", currents)

    def _resize(self, currents: float | list | np.ndarray):
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

        Raises
        ------
        ValueError
            Number of arguments not equal to number og coil groups
        """
        if not args:
            for ff in self.__list_getter(func):
                ff(**kwargs)
        else:
            args = list(args)
            funclist = self.__list_getter(func)
            len_funclist = len(funclist)
            for no, arg in enumerate(args):
                if isinstance(arg, float | int):
                    args[no] = np.full(len_funclist, arg)
                elif len(arg) != len_funclist:
                    raise ValueError(
                        f"length of {arg} != number of coilgroups ({len_funclist})"
                    )
            for ff, *_args in zip(funclist, *args, strict=False):
                ff(*_args, **kwargs)

    def add_coil(self, *coils: Coil | CoilGroup):
        """Add coils to the coil group"""
        self._coils = (*self._coils, *coils)

    def remove_coil(self, *coil_name: str, _top_level: bool = True) -> list[str] | None:
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
        :
            Removed names if not at top level of nested stack

        Raises
        ------
        EquilibriaError
            Unknown coils

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
        mask = np.full(coils.size, fill_value=True, dtype=bool)
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
    def from_group_vecs(cls, eqdsk: EQDSKInterface) -> CoilGroup:
        """
        Initialises an instance of CoilSet from group vectors.

        This has been implemented as a dict operation, because it will
        occur for eqdsks only.
        Future dict instantiation methods will likely differ, hence the
        confusing name of this method.

        There should always be a coilset but...
        if for some reason the coilset is missing, and the user has not
        provided 'user_coils' as an input for 'from_eqdsk', then a dummy
        coilset is used and a warning message is printed.
        """  # noqa: DOC201
        pfcoils = []
        cscoils = []
        passivecoils = []
        if eqdsk.ncoil < 1:
            grid = Grid.from_eqdsk(eqdsk)
            dum_xc = [np.min(grid.x), np.max(grid.x), np.max(grid.x), np.min(grid.x)]
            dum_zc = [np.min(grid.z), np.min(grid.z), np.max(grid.z), np.max(grid.z)]
            for i in range(4):
                coil = Coil(
                    dum_xc[i],
                    dum_zc[i],
                    current=0,
                    dx=0,
                    dz=0,
                    ctype=CoilType.DUM,
                    j_max=0,
                    b_max=0,
                )
                coil.fix_size()
                pfcoils.append(coil)
            coils = pfcoils
            bluemira_warn(
                "EQDSK coilset empty - dummy coilset in use. "
                "Please replace with an appropriate coilset."
            )
            return cls(*coils)

        def _get_val(lst: npt.ArrayLike | None, idx: int, default=None):
            if lst is None:
                return None
            try:
                return lst[idx]
            except IndexError:
                return default

        for i in range(eqdsk.ncoil):
            dx = eqdsk.dxc[i]
            dz = eqdsk.dzc[i]
            cn = _get_val(eqdsk.coil_names, i)
            ct = None if (v := _get_val(eqdsk.coil_types, i)) is None else CoilType(v)
            if ct is CoilType.NONE or (abs(eqdsk.Ic[i]) < I_MIN and ct is None):
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
                        ctype=ct or CoilType.NONE,
                        name=cn,
                    )
                )
            elif ct is CoilType.CS or (dx != dz and ct is None):  # Rough and ready
                cscoils.append(
                    Coil(
                        eqdsk.xc[i],
                        eqdsk.zc[i],
                        current=eqdsk.Ic[i],
                        dx=dx,
                        dz=dz,
                        ctype=CoilType.CS,
                        name=cn,
                    )
                )
            else:
                coil = Coil(
                    eqdsk.xc[i],
                    eqdsk.zc[i],
                    current=eqdsk.Ic[i],
                    dx=dx,
                    dz=dz,
                    ctype=ct or CoilType.PF,
                    name=cn,
                )
                coil.fix_size()  # Oh ja
                pfcoils.append(coil)

        coils = pfcoils
        coils.extend(cscoils)
        coils.extend(passivecoils)
        return cls(*coils)

    def to_group_vecs(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    def __list_getter(self, attr: str) -> list:
        """Get attributes from coils tuple"""  # noqa: DOC201
        return np.frompyfunc(attrgetter(attr), 1, 1)(self._coils)

    def __getter(self, attr: str) -> np.ndarray:
        """
        Returns
        -------
        :
            Attribute from coils and convert to flattened numpy array
        """
        arr = np.array([*flatten_iterable(self.__list_getter(attr))])
        arr.flags.writeable = False
        return arr

    def __quad_getter(self, attr: str) -> np.ndarray:
        """
        Returns
        -------
        :
            Quadratures and autopad to create non ragged array
        """
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
        values: CoilType | float | Iterable[CoilType | float],
        dtype: type | None = None,
    ):
        """Set attributes on coils

        Raises
        ------
        ValueError
            Number of elements < number of coils
        """
        values = np.atleast_1d(values)
        if dtype not in {None, object}:
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
        """Get coils"""  # noqa: DOC201
        return self._find_coil(item)

    def __copy__(self):
        """
        Copy dunder method, needed because attribute setter fails for quadratures
        """  # noqa: DOC201
        cls = self.__class__
        result = cls.__new__(cls)
        for k in self.__slots__:
            setattr(result, k, getattr(self, k))
        return result

    def __deepcopy__(self, memo):
        """
        Deepcopy dunder method, needed because attribute setter fails for
        quadratures
        """  # noqa: DOC201
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k in self.__slots__:
            setattr(result, k, deepcopy(getattr(self, k), memo))
        result._einsum_str = self._einsum_str
        return result

    def _pad_discretisation(
        self,
        _to_pad: list[np.ndarray],
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
        There are a few extra calculations of the greens functions where padding exists
        in
        :meth:`~bluemira.equilibria.coils._field.CoilGroupFieldsMixin._combined_control`.

        """
        all_len = np.array([q.shape[-1] for q in _to_pad])
        max_len = max(all_len)
        self._pad_size = max_len - all_len

        self._einsum_str = "...j, ...j -> ..."

    def _find_coil(self, name):
        """Find coil by name

        Raises
        ------
        KeyError
            Coil not found in group
        """  # noqa: DOC201
        for c in self._coils:
            if isinstance(c, CoilGroup):
                try:
                    return c._find_coil(name)
                except KeyError:
                    pass
            elif c.name == name:
                return c

        raise KeyError(f"Coil '{name}' not found in Group")

    def _get_coiltype(self, *ctype: CoilType | str) -> list[Coil]:
        """Find coil by type"""  # noqa: DOC201
        coils = []
        ctype = tuple(CoilType(ct) for ct in ctype)
        for c in self._coils:
            if isinstance(c, CoilGroup):
                coils.extend(c._get_coiltype(*ctype))
            elif c.ctype in ctype:
                coils.append(c)
        return coils

    def all_coils(self) -> list[Coil]:
        """
        Returns
        -------
        :
            All coils as a flattened list (no CoilGroups)
        """
        return [self[n] for n in self.name]

    def _get_type_index(self, *ctype: CoilType | str) -> npt.NDArray[int]:
        coil_type = tuple(CoilType(ct) for ct in ctype) if ctype else CoilType

        return np.asarray(
            [no for no, coil in enumerate(self.all_coils()) if coil.ctype in coil_type],
            dtype=int,
        )

    def get_coiltype(self, *ctype: str | CoilType) -> CoilGroup | None:
        """
        Returns
        -------
        :
            Coils matching coil type
        """
        if coiltype := self._get_coiltype(*ctype):
            return CoilGroup(*coiltype)
        return None

    def assign_material(self, ctype, j_max, b_max):
        """Assign material J and B to Coilgroup"""
        cg = self.get_coiltype(ctype)
        cg.j_max = j_max
        cg.b_max = b_max

    def get_max_current(self, max_current: float = np.inf) -> np.ndarray:
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
        :
            Maximum currents
        """
        return np.where(
            np.isnan(self.j_max) | ~self._flag_sizefix,  # or not
            max_current,
            get_max_current(self.dx, self.dz, self.j_max),
        )

    @property
    def name(self) -> list:
        """Get coil names"""
        return self.__getter("name").tolist()

    @property
    def primary_coil(self) -> Coil:
        """Get primary coil, which is arbitrarily taken to
        be the first coil in the group.

        Will recurse if the first coil itself is a CoilGroup.
        """
        c = self._coils[0]
        if isinstance(c, CoilGroup):
            return c.primary_coil
        return c

    def get_coil_or_group_with_coil_name(self, coil_name: str) -> Coil | CoilGroup:
        """
        Get the coil or coil group with the coil with `coil_name` in it.

        This will be the lowest level group that contains the coil.

        Parameters
        ----------
        coil_name:
            The coil name to search for

        Returns
        -------
        :
            The coil or coil group with the given coil name in it

        Raises
        ------
        ValueError
            No coil found with given name
        """
        for coil_or_group in self._coils:
            if isinstance(coil_or_group, CoilGroup):
                try:
                    c_rtn = coil_or_group.get_coil_or_group_with_coil_name(coil_name)
                except ValueError:
                    continue
                # if it's a CoilGroup, return it
                # (we want the lowest level group that contains the coil)
                if isinstance(c_rtn, CoilGroup):
                    return c_rtn
                # otherwise it's a coil,
                # so return the group it's in
                return coil_or_group

            if coil_or_group.name == coil_name:
                return coil_or_group
        raise ValueError(f"No coil or coil group with primary coil name {coil_name}")

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
    def ctype(self) -> list:
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
    def resistance(self) -> np.ndarray:
        """Get coil resistance"""
        return self.__getter("resistance")

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
        n = []
        qbs = self.__list_getter("_quad_boundary")
        for qb in qbs:
            # basically checking if not a tuple
            # but this ensures the type
            if isinstance(qb, np.ndarray | list):
                n.extend(qb)
            else:
                n.append(qb)
        return n

    @x.setter
    def x(self, values: float | Iterable[float]):
        """Set coil x positions"""
        self.__setter("x", values)

    @z.setter
    def z(self, values: float | Iterable[float]):
        """Set coil z positions"""
        self.__setter("z", values)

    @position.setter
    def position(self, values: np.ndarray):
        """Set coil positions"""
        self.__setter("x", values[0])
        self.__setter("z", values[1])

    @ctype.setter
    def ctype(self, values: CoilType | Iterable[CoilType]):
        """Set coil types"""
        self.__setter("ctype", values, dtype=object)

    @dx.setter
    def dx(self, values: float | Iterable[float]):
        """Set coil dx sizes"""
        self.__setter("dx", values)

    @dz.setter
    def dz(self, values: float | Iterable[float]):
        """Set coil dz sizes"""
        self.__setter("dz", values)

    @current.setter
    def current(self, values: float | Iterable[float]):
        """Set coil currents"""
        self.__setter("current", values)

    @j_max.setter
    def j_max(self, values: float | Iterable[float]):
        """Set coil max current densities"""
        self.__setter("j_max", values)

    @b_max.setter
    def b_max(self, values: float | Iterable[float]):
        """Set coil max fields"""
        self.__setter("b_max", values)

    @resistance.setter
    def resistance(self, values: float | Iterable[float]):
        """Set coil resistance"""
        self.__setter("b_max", values)

    @discretisation.setter
    def discretisation(self, values: float | Iterable[float]):
        """Set coil discretisations"""
        self.__setter("discretisation", values)
        self._pad_discretisation(self.__list_getter("_quad_x"))

    @n_turns.setter
    def n_turns(self, values: float | Iterable[float]):
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
        self,
        *coils: Coil | CoilGroup,
        current: float | None = None,
        psi_analytic: bool = False,
        Bx_analytic: bool = True,
        Bz_analytic: bool = True,
    ):
        super().__init__(
            *coils,
            psi_analytic=psi_analytic,
            Bx_analytic=Bx_analytic,
            Bz_analytic=Bz_analytic,
        )
        self.current = self._get_current() if current is None else current

    def _get_current(self):
        current = self._coils[0].current
        if isinstance(current, Iterable):
            current = current[0]
        return current

    @property
    def primary_group(self) -> Coil | CoilGroup:
        """Get the first coil or group in the circuit as the 'primary_group'"""
        return self._coils[0]

    def add_coil(self, *coils: Coil | CoilGroup):
        """
        Add coil to circuit forcing the same current
        """
        super().add_coil(coils)
        self.current = self._get_current()

    @CoilGroup.current.setter
    def current(self, values: float | Iterable[float]):
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
        *coils: Coil | CoilGroup,
        psi_analytic: bool = False,
        Bx_analytic: bool = True,
        Bz_analytic: bool = True,
    ):
        if len(coils) == 1:
            coils = (coils[0], deepcopy(coils[0]))
        if len(coils) != 2:  # noqa: PLR2004
            raise EquilibriaError(
                f"Wrong number of coils to create a {type(self).__name__}"
            )

        super().__init__(
            *coils,
            psi_analytic=psi_analytic,
            Bx_analytic=Bx_analytic,
            Bz_analytic=Bz_analytic,
        )

        symmetry_line = np.array([[0, 0], [1, 0]])

        self.modify_symmetry(symmetry_line)
        diff = self._symmetrise()
        self.symmetric_group.x -= diff[0]
        self.symmetric_group.z -= diff[1]

    @property
    def symmetric_group(self) -> Coil | CoilGroup:
        """Get the second coil or group as the 'symmetric_group'"""
        return self._coils[1]

    def modify_symmetry(self, symmetry_line: np.ndarray):
        """
        Create a unit vector for the symmetry of the circuit

        Parameters
        ----------
        symmetry_line:
            two points making a symmetry line [[float, float], [float, float]]
        """
        self._symmetry_line = symmetry_line
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

        Returns
        -------
        :
            The change in position
        """
        cp = self._get_group_centre()
        cp[1] -= self._shift

        mirror = np.dot(self.sym_mat, cp.T)
        mirror[1] += self._shift

        return (
            np.array([
                self._get_symmetric_group_x_centre(),
                self._get_symmetric_group_z_centre(),
            ])
            - mirror
        )

    @Circuit.position.setter
    def position(self, values: np.ndarray):
        """Set coil positions"""
        self.x = values[0]
        self.z = values[1]

    @Circuit.x.setter
    def x(self, new_x: float):
        """
        Set x coordinate of each coil
        """
        if isinstance(new_x, np.ndarray):
            new_x = np.mean(new_x[0])
        self.primary_group.x += new_x - self._get_primary_group_x_centre()
        self.symmetric_group.x -= self._symmetrise()[0]

    @Circuit.z.setter
    def z(self, new_z: float):
        """
        Set z coordinate of each coil
        """
        if isinstance(new_z, np.ndarray):
            new_z = np.mean(new_z[0])
        self.primary_group.z += new_z - self._get_primary_group_z_centre()
        self.symmetric_group.z -= self._symmetrise()[1]

    def remove_coil(self, *coil_name: str, _top_level: bool = True) -> list[str] | None:
        """
        Remove coil(s) from CoilSet

        Parameters
        ----------
        coil_name:
            coil(s) to remove
        _top_level:
            FOR INTERNAL USE, flags if at top level of nested coilgroup stack

        Returns
        -------
        :
            Removed coil names, if not at top level of nested stack

        Raises
        ------
        ValueError
            Cannot remove coils from a SymmetricCircuit with nested CoilGroups
        """
        if not all(isinstance(coil, Coil) for coil in self._coils):
            raise ValueError(
                "Cannot remove coils from a SymmetricCircuit with nested CoilGroups"
            ) from None

        if all(name in coil_name for name in self.name):
            # needed for short circuiting
            pass
        elif any(name in coil_name for name in self.name):
            bluemira_warn(
                "Removing one coil from a SymmetricCircuit implies removing both"
            )
            coil_name = self.name

        return super().remove_coil(*coil_name, _top_level=_top_level)

    def _get_primary_group_x_centre(self) -> np.float64:
        """Get the x centre of the first coil group"""  # noqa: DOC201
        return np.mean(self.primary_group.x)

    def _get_primary_group_z_centre(self) -> np.float64:
        """Get the z centre of the first coil group"""  # noqa: DOC201
        return np.mean(self.primary_group.z)

    def _get_symmetric_group_x_centre(self) -> np.float64:
        """Get the x centre of the first coil group"""  # noqa: DOC201
        return np.mean(self.symmetric_group.x)

    def _get_symmetric_group_z_centre(self) -> np.float64:
        """Get the z centre of the first coil group"""  # noqa: DOC201
        return np.mean(self.symmetric_group.z)

    def _get_group_centre(self) -> np.ndarray:
        """Get the centre of the first coil group"""  # noqa: DOC201
        return np.array([
            self._get_primary_group_x_centre(),
            self._get_primary_group_z_centre(),
        ])


@dataclass
class CoilSetOptimisationState:
    """
    State of the optimisation of a CoilSet

    Parameters
    ----------
    currents:
        The current values of the coils
    xs:
        The x positions of the coils
    zs:
        The z positions of the coils
    """

    currents: np.ndarray
    xs: np.ndarray
    zs: np.ndarray

    def __post_init__(self):
        if self.xs.size != self.zs.size:
            raise ValueError("Number of xs must match the number of zs in the CoilSet")

    @property
    def positions(self) -> np.ndarray:
        """Get the positions as a (2,N) array"""
        return np.array([self.xs, self.zs])

    @property
    def positions_flat(self) -> np.ndarray:
        """Get the positions as an array with xs and zs concatenated (xs then zs)"""
        return np.concatenate([self.xs, self.zs])


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
        *coils: Coil | CoilGroup,
        control_names: list | bool | None = None,
        psi_analytic: bool = False,
        Bx_analytic: bool = True,
        Bz_analytic: bool = True,
    ):
        super().__init__(
            *coils,
            psi_analytic=psi_analytic,
            Bx_analytic=Bx_analytic,
            Bz_analytic=Bz_analytic,
        )
        self.control = control_names

    def remove_coil(self, *coil_name: str, _top_level: bool = True) -> list[str] | None:
        """
        Remove coil(s) from CoilSet

        Parameters
        ----------
        coil_name:
            coil(s) to remove
        _top_level:
            FOR INTERNAL USE, flags if at top level of nested coilgroup stack

        Returns
        -------
        :
            Removed coil names, if not at top level of nested stack
        """
        removed_coils = super().remove_coil(*coil_name, _top_level=_top_level)
        self.control = list(set(self.control) & set(self.name))
        return removed_coils

    @property
    def control(self) -> list[str]:
        """Get control coil names"""
        return self._control

    @control.setter
    def control(self, control_names: list[str] | bool | None = None):
        """Set which coils are actively controlled

        Parameters
        ----------
        control_names:
                    - list of str, each one being the name of each control coil.
                    - None, for when ALL coils are control coils.
                    - a boolean, which denotes all controlled vs none controlled.
        """
        names = self.name
        if isinstance(control_names, list):
            self._control_ind = [names.index(c) for c in control_names]
        elif control_names or control_names is None:
            self._control_ind = np.arange(len(names)).tolist()
        else:
            self._control_ind = []
        self._control = [names[c] for c in self._control_ind]

    def get_control_coils(self) -> CoilSet:
        """Get control coils"""  # noqa: DOC201
        coils = []
        for c in self._coils:
            if isinstance(c, CoilSet):
                coils.extend(c.get_control_coils()._coils)
            elif (isinstance(c, Coil) and c.name in self.control) or (
                isinstance(c, CoilGroup) and any(n in self.control for n in c.name)
            ):
                coils.append(c)
        return CoilSet(*coils)

    def get_uncontrolled_coils(self) -> CoilSet:
        """Get uncontrolled coils"""
        coils = []
        for c in self._coils:
            if isinstance(c, CoilSet):
                coils.extend(c.get_uncontrolled_coils()._coils)
            elif (isinstance(c, Coil) and c.name not in self.control) or (
                isinstance(c, CoilGroup) and not any(n in self.control for n in c.name)
            ):
                coils.append(c)
        return CoilSet(*coils)

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
        self, output: np.ndarray, *, sum_coils: bool = False, control: bool = False
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
        :
            Summed response output
        """
        inds = self._control_ind if control else slice(None)

        return np.sum(output[..., inds], axis=-1) if sum_coils else output[..., inds]

    def get_coiltype(self, *ctype: str | CoilType) -> CoilSet | None:
        """Get coils by coils type"""  # noqa: DOC201
        if coiltype := self._get_coiltype(*ctype):
            return CoilSet(*coiltype)  # noqa: DOC201
        return None

    @classmethod
    def from_group_vecs(
        cls, eqdsk: EQDSKInterface, control_coiltypes=(CoilType.PF, CoilType.CS)
    ) -> CoilGroup:
        """Create CoilSet from eqdsk group vectors.

        Automatically sets all coils that are not implicitly passive to control coils
        """  # noqa: DOC201
        self = super().from_group_vecs(eqdsk)

        self.control = [coil.name for coil in self._get_coiltype(*control_coiltypes)]
        return self

    def get_optimisation_state(
        self, position_coil_names: list[str] | None = None, current_scale: float = 1.0
    ) -> CoilSetOptimisationState:
        """
        Get the state of the CoilSet for optimisation.

        Parameters
        ----------
        position_coil_names:
            The names of the coils to get the positions of
        current_scale:
            The scale of the currents

        Returns
        -------
        CoilSetOptimisationState
            The state of the CoilSet for optimisation
        """
        cc = self.get_control_coils()
        xs, zs = cc._get_opt_positions(position_coil_names)
        currents = cc._opt_currents / current_scale
        return CoilSetOptimisationState(
            currents=currents,
            xs=xs,
            zs=zs,
        )

    def set_optimisation_state(
        self,
        opt_currents: np.ndarray | None = None,
        coil_position_map: dict[str, np.ndarray] | None = None,
        current_scale: float = 1.0,
    ):
        """
        Set the state of the CoilSet, post optimisation.

        Used in conjunction with `get_optimisation_state`.

        Parameters
        ----------
        opt_currents:
            The optimisation currents
        coil_position_map:
            The map of coil names to positions: [x, z]
        current_scale:
            The scale of the currents
        """
        cc = self.get_control_coils()
        if opt_currents is not None:
            cc._opt_currents = opt_currents * current_scale
        if coil_position_map is not None:
            cc._set_opt_positions(coil_position_map)

    @property
    def n_current_optimisable_coils(self) -> int:
        """
        Get the number of all current optimisable coils.
        """
        return len(self.current_optimisable_coil_names)

    @property
    def current_optimisable_coil_names(self) -> list[str]:
        """
        Get the names of all current optimisable coils.
        """
        optimisable_coil_names = [
            c.primary_coil.name if isinstance(c, Circuit) else c.name
            for c in self._coils
        ]
        return [*flatten_iterable(optimisable_coil_names)]

    @property
    def all_current_optimisable_coils(self) -> list[Coil]:
        """
        Get a list of all coils that can be current optimised.
        """
        return [self[cn] for cn in self.current_optimisable_coil_names]

    def get_current_optimisable_coils(
        self, coil_names: list[str] | None = None
    ) -> list[Coil]:
        """
        Get a list of coils that can be current optimised.

        If coil_names is given, only the coils with those names are returned.
        Names in coil_names must be a subset of current optimisable coils.

        Parameters
        ----------
        coil_names:
            The names of the coils to get.
            If None, all current optimisable coils are returned.

        Raises
        ------
        ValueError
            If a name in `coil_names` in not in `all_current_optimisable_coils`.
        """  # noqa: DOC201
        if coil_names is None:
            return self.all_current_optimisable_coils

        opt_coils_map = {c.name: c for c in self.all_current_optimisable_coils}
        rtn = []
        for cn in coil_names:
            c = opt_coils_map.get(cn)
            if c is not None:
                rtn.append(c)
            else:
                raise ValueError(f"Coil {cn} is not a current optimisable coil")
        return rtn

    @property
    def _opt_currents_inds(self) -> list[int]:
        """
        Get the indices of the coils that can be optimised.

        These indices are used to extract the optimisable currents from the CoilSet
        and are based on the index of the coils in the name array.
        """
        return [self.name.index(cn) for cn in self.current_optimisable_coil_names]

    @property
    def _contains_circuits(self) -> bool:
        """
        A simple check to see if the CoilSet contains any Circuits.
        """
        return any(isinstance(c, Circuit) for c in self._coils)

    @property
    def _opt_currents_expand_mat(self) -> np.ndarray:
        """
        Get the optimisation currents expansion matrix.

        This matrix is used to convert the optimisable currents to the full set of
        currents in the CoilSet.

        Raises
        ------
        ValueError
            Number of optimisable coils not equal number of groups/coils
        """
        cc = self.get_control_coils()

        n_all_coils = cc.n_coils()
        n_opt_coils = cc.n_current_optimisable_coils
        n_distinct_coils_and_groupings = len(cc._coils)

        if not cc._contains_circuits:
            return np.eye(n_all_coils)

        # this should be true as, at the top level, the number
        # of coil or group objects should be the same as the no
        # of optimisable coils
        if n_opt_coils != n_distinct_coils_and_groupings:
            raise ValueError(
                "The number of optimisable coils does not match the number "
                "of distinct coils and groupings. Something's gone wrong."
            )

        # you are putting 1's in the col. corresponding
        # to all coils in the same Circuit
        mat = np.zeros((n_all_coils, n_opt_coils))
        i_row = 0
        for i_col, c in enumerate(cc._coils):
            if isinstance(c, Circuit):
                n_coils_in_group = c.n_coils()
                for n in range(n_coils_in_group):
                    mat[i_row + n, i_col] = 1
                i_row += n
            else:
                mat[i_row, i_col] = 1
            i_row += 1
        return mat

    @property
    def _opt_currents(self) -> np.ndarray:
        """
        Get the currents for the optimisable coils.
        """
        return self.current[self._opt_currents_inds]

    @_opt_currents.setter
    def _opt_currents(self, values: np.ndarray):
        """
        Set the currents for the optimisable coils.

        Raises
        ------
        ValueError
            Number of values not equal to number of optimisable currents
        """
        n_all_coils = self.n_coils()

        n_vals = values.shape[0]
        n_curr_opt_coils = self.n_current_optimisable_coils

        if n_vals == 1:
            c = values[0]
            self.current = np.ones(n_all_coils) * c
            return

        if n_vals != n_curr_opt_coils:
            raise ValueError(
                f"The number of current elements {n_vals} "
                "does not match the number of "
                f"optimisable currents: {n_curr_opt_coils}"
            )

        self.current = self._opt_currents_expand_mat @ values

    @property
    def n_position_optimisable_coils(self) -> int:
        """
        Get the number of coils that can be position optimised.
        """
        return len(self.position_optimisable_coil_names)

    @property
    def position_optimisable_coil_names(self) -> list[str]:
        """
        Get the names of the coils that can be position optimised.
        """
        optimisable_coil_names = (
            c.primary_coil.name if isinstance(c, SymmetricCircuit) else c.name
            for c in self._coils
        )
        return [*flatten_iterable(optimisable_coil_names)]

    @property
    def all_position_optimisable_coils(self) -> list[Coil]:
        """
        Get the names of all coils that can be position optimised.
        """
        return [self[cn] for cn in self.position_optimisable_coil_names]

    def get_position_optimisable_coils(
        self, coil_names: list[str] | None = None
    ) -> list[Coil]:
        """
        Get the coils that can be position optimised.

        Parameters
        ----------
        coil_names:
            The names of the coils to get the positions of.
            If None, all position optimisable coils are returned.

        Raises
        ------
        ValueError
            Coil's position not optimisable
        """  # noqa: DOC201
        if coil_names is None:
            return self.all_position_optimisable_coils

        opt_coils_map = {c.name: c for c in self.all_position_optimisable_coils}
        rtn = []
        for cn in coil_names:
            c = opt_coils_map.get(cn)
            if c is not None:
                rtn.append(c)
            else:
                raise ValueError(f"Coil {cn} is not a position optimisable coil")
        return rtn

    def _get_opt_positions(
        self, position_coil_names: list[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the positions of the position optimisable coils.

        Parameters
        ----------
        position_coil_names:
            The names of the coils to get the positions of.
            If None, all position optimisable coils are returned.

        Returns
        -------
        :
            The x coordinates
        :
            The z coordinates
        """
        coils = self.get_position_optimisable_coils(position_coil_names)
        x, z = [c.x for c in coils], [c.z for c in coils]
        return np.asarray(x), np.asarray(z)

    def _set_opt_positions(self, coil_position_map: dict[str, np.ndarray]) -> None:
        """
        Set the positions of the position optimisable coils.
        """
        pos_opt_coil_names = self.get_control_coils().position_optimisable_coil_names
        for coil_name, position in coil_position_map.items():
            if coil_name in pos_opt_coil_names:
                c = self.get_coil_or_group_with_coil_name(coil_name)
                c.position = position
