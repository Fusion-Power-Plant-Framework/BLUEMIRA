# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Coil and coil grouping objects
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bluemira.base.constants import MU_0
from bluemira.equilibria.constants import X_TOLERANCE
from bluemira.magnetostatics.greens import (
    greens_Bx,
    greens_Bz,
    greens_dbz_dx,
    greens_psi,
)
from bluemira.magnetostatics.semianalytic_2d import (
    semianalytic_Bx,
    semianalytic_Bz,
    semianalytic_psi,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from bluemira.equilibria.coils._grouping import CoilGroup


class CoilGroupFieldsMixin:
    """
    CoilGroup magnetic fields mixin.

    Add field calculation mechanics to coilgroups
    """

    __slots__ = (
        "_Bx_analytic",
        "_Bz_analytic",
        "_einsum_str",
        "_psi_analytic",
        "_quad_dx",
        "_quad_dz",
        "_quad_weighting",
        "_quad_x",
        "_quad_z",
    )

    def __init__(
        self,
        *,
        psi_analytic: bool = False,
        Bx_analytic: bool = True,
        Bz_analytic: bool = True,
    ):
        self._psi_analytic = psi_analytic
        self._Bx_analytic = Bx_analytic
        self._Bz_analytic = Bz_analytic

    def dB_d(self, x: float | np.ndarray, z: float | np.ndarray):
        """
        dB_d of Coilset

        Parameters
        ----------
        x:
            The x values at which to calculate the dB_d response
        z:
            The z values at which to calculate the dB_d response
        sum_coils:
            sum over coils
        control:
            operations on control coils only

        Returns
        -------
        :
            Differential of magnetic field
        """
        return self.dB_d_response(x, z) * self.current

    def dB_d_response(self, x: float | np.ndarray, z: float | np.ndarray):
        """
        Unit dB_d of Coilset

        Parameters
        ----------
        x:
            The x values at which to calculate the dB_d response
        z:
            The z values at which to calculate the dB_d response
        sum_coils:
            sum over coils
        control:
            operations on control coils only

        Returns
        -------
        :
            Differential of magnetic field response
        """
        return self._mix_control_method(x, z, greens_dbz_dx, None, disable_analytic=True)

    def psi(self, x: float | np.ndarray, z: float | np.ndarray):
        """
        Calculate poloidal flux at (x, z)
        """  # noqa: DOC201
        return self.psi_response(x, z) * self.current

    def psi_response(self, x, z):
        return self._mix_control_method(
            x, z, greens_psi, semianalytic_psi, disable_analytic=not self._psi_analytic
        )

    def Bx(self, x: float | np.ndarray, z: float | np.ndarray):
        """
        Calculate radial magnetic field Bx at (x, z)
        """  # noqa: DOC201
        return self.Bx_response(x, z) * self.current

    def Bx_response(
        self, x: float | np.ndarray, z: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Calculate the radial magnetic field response at (x, z) due to a unit
        current. Green's functions are used outside the coil, and a semianalytic
        method is used for the field inside the coil.

        Parameters
        ----------
        x:
            The x values at which to calculate the Bx response
        z:
            The z values at which to calculate the Bx response

        Returns
        -------
        :
            The radial magnetic field response at the x, z coordinates.
        """
        return self._mix_control_method(
            x, z, greens_Bx, semianalytic_Bx, disable_analytic=not self._Bx_analytic
        )

    def Bz(self, x: float | np.ndarray, z: float | np.ndarray) -> float | np.ndarray:
        """
        Calculate vertical magnetic field Bz at (x, z)
        """  # noqa: DOC201
        return self.Bz_response(x, z) * self.current

    def Bz_response(
        self, x: float | np.ndarray, z: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Calculate the vertical magnetic field response at (x, z) due to a unit
        current. Green's functions are used outside the coil, and a semianalytic
        method is used for the field inside the coil.

        Parameters
        ----------
        x:
            The x values at which to calculate the Bz response
        z:
            The z values at which to calculate the Bz response

        Returns
        -------
        :
            The vertical magnetic field response at the x, z coordinates.
        """
        return self._mix_control_method(
            x, z, greens_Bz, semianalytic_Bz, disable_analytic=not self._Bz_analytic
        )

    def Bp(self, x: float | np.ndarray, z: float | np.ndarray):
        """
        Calculate poloidal magnetic field Bp at (x, z)

        Notes
        -----
        .. math::
            B_p = \\sqrt{B_x^2 + B_z^2}

        where `B_x` and `B_z` are the x and z components of the
        magnetic field.
        """  # noqa: DOC201
        return np.hypot(self.Bx(x, z), self.Bz(x, z))

    def F(self, eqcoil: CoilGroup) -> np.ndarray:
        """
        Calculate the force response at the coil centre including the coil
        self-force.

        .. math::

             \\mathbf{F} = \\mathbf{j}\\times \\mathbf{B}
            F_x = IB_z+\\dfrac{\\mu_0I^2}{4\\pi X}\\textrm{ln}\\bigg(\\dfrac{8X}{r_c}-1+\\xi/2\\bigg)
            F_z = -IBx

        Returns
        -------
        :
            The force response
        """  # noqa: W505, E501
        multiplier = self.current * 2 * np.pi * self.x
        cr = self._current_radius
        if any(cr != 0):
            # true divide errors for zero current coils
            cr_ind = np.nonzero(cr)
            fx = np.zeros_like(cr)
            fx[cr_ind] = (
                MU_0
                * self.current[cr_ind] ** 2
                / (4 * np.pi * self.x[cr_ind])
                * (np.log(8 * self.x[cr_ind] / cr[cr_ind]) - 1 + 0.25)
            )
        else:
            fx = 0

        return np.array([
            multiplier * (eqcoil.Bz(self.x, self.z) + fx),
            -multiplier * eqcoil.Bx(self.x, self.z),
        ]).T

    def control_F(self, coil_grp: CoilGroup) -> np.ndarray:
        """
        Calculates the coil mutual force

        \t:math:`Fz_{i,j}=-2\\pi X_i\\mathcal{G}(X_j,Z_j,X_i,Z_i)`

        Returns
        -------
        :
            The Green's matrix element for the coil mutual force.

        """
        # TODO @je-cook: Vectorise
        # 3575
        x, z = np.atleast_1d(self.x), np.atleast_1d(self.z)  # single coil
        pos = np.array([x, z])
        response = np.zeros((x.size, coil_grp.x.size, 2))
        for j, coil in enumerate(coil_grp.all_coils()):
            xw = np.nonzero(x == coil.x)[0]
            zw = np.nonzero(z == coil.z)[0]
            same_pos = np.nonzero(xw == zw)[0]
            if same_pos.size > 0:
                # self inductance
                # same_pos could be an array that is indexed from zw.
                # This loops over zw and creates an index in xw where xw == zw
                # better ways welcome!
                xxw = []
                for _z in zw:
                    if (_pos := np.nonzero(_z == xw)[0]).size > 0:
                        xxw.extend(_pos)
                cr = self._current_radius[np.array(xxw)]
                Bz = np.zeros((x.size, 1))
                Bx = Bz.copy()  # Should be 0 anyway
                mask = np.zeros_like(Bz, dtype=bool)
                mask[same_pos] = True
                if any(cr != 0):
                    cr_ind = np.nonzero(cr)
                    Bz[mask][cr_ind] = (
                        MU_0
                        / (4 * np.pi * x[cr_ind])
                        * (np.log(8 * x[cr_ind] / cr[cr_ind]) - 1 + 0.25)
                    )
                if False in mask:
                    Bz[~mask] = coil.Bz_response(*pos[:, ~mask[:, 0]])
                    Bx[~mask] = coil.Bx_response(*pos[:, ~mask[:, 0]])

            else:
                Bz = coil.Bz_response(x, z)
                Bx = coil.Bx_response(x, z)

            # 1 cross B
            response[:, j, :] = (
                2 * np.pi * x[:, np.newaxis] * np.squeeze(np.array([Bz, -Bx]).T)
            )
        return response

    def _stored_greens(self, green: float | np.ndarray):
        """
        Quickly calculate plasma psi, Bx or Bz from Greens functions and current
        """  # noqa: DOC201
        return self.current * green

    def _mix_control_method(
        self,
        x: float | np.ndarray,
        z: float | np.ndarray,
        greens_func: Callable,
        semianalytic_func: Callable,
        *,
        disable_analytic: bool = False,
    ) -> float | np.ndarray:
        """
        Boiler-plate helper function to mixed the Green's function responses
        with the semi-analytic function responses, as a function of position
        outside/inside the coil boundary.

        Parameters
        ----------
        x:
            The x values at which to calculate the response at
        z:
            The z values at which to calculate the response at
        greens_func:
            greens function
        semianalytic_func:
            semianalytic function

        Returns
        -------
        Mixed control response
        """
        x, z = np.ascontiguousarray(x), np.ascontiguousarray(z)

        # if not wrapped in np.array the following if won't work for `Coil`
        zero_coil_size = np.array(
            np.logical_or(np.isclose(self.dx, 0), np.isclose(self.dz, 0))
        )

        if False in zero_coil_size:
            # if dx or dz is not 0 and x,z inside coil
            inside = np.logical_and(
                self._points_inside_coil(x, z), ~zero_coil_size[np.newaxis]
            )
            if np.all(~inside) or disable_analytic:
                return self._response_greens(greens_func, x, z)
            if np.all(inside):
                # Not called for circuits as they will always be a mixture
                return self._response_analytical(semianalytic_func, x, z)
            return self._combined_control(inside, x, z, greens_func, semianalytic_func)
        return self._response_greens(greens_func, x, z)

    def _combined_control(
        self,
        inside: np.ndarray,
        x: float | np.ndarray,
        z: float | np.ndarray,
        greens_func: Callable,
        semianalytic_func: Callable,
    ) -> float | np.ndarray:
        """
        Combine semianalytic and greens function calculation of [psi, Bx, Bz]

        Used for situation where there are calculation points both inside and
        outside the coil boundaries.

        Parameters
        ----------
        inside:
            array of if the point is inside a coil
        x:
            The x values at which to calculate the response at
        z:
            The z values at which to calculate the response at
        greens_func:
            greens function
        semianalytic_func:
            semianalytic function

        Returns
        -------
        Combined control response
        """
        response = np.zeros_like(inside, dtype=float)
        for coil, (points, qx, qz, qw, cx, cz, cdx, cdz) in enumerate(
            zip(
                np.moveaxis(inside, -1, 0),
                self._quad_x,
                self._quad_z,
                self._quad_weighting,
                self.x,
                self.z,
                self.dx,
                self.dz,
                strict=False,
            )
        ):
            if np.any(~points):
                response[~points, coil] = self._response_greens(
                    greens_func,
                    x[~points],
                    z[~points],
                    split=True,
                    _quad_x=qx,
                    _quad_z=qz,
                    _quad_weight=qw,
                )

            if np.any(points):
                response[points, coil] = self._response_analytical(
                    semianalytic_func,
                    x[points],
                    z[points],
                    split=True,
                    coil_x=cx,
                    coil_z=cz,
                    coil_dx=cdx,
                    coil_dz=cdz,
                )

        return np.squeeze(response)

    def _points_inside_coil(
        self,
        x: float | np.array,
        z: float | np.array,
        *,
        atol: float = X_TOLERANCE,
    ) -> np.ndarray:
        """
        Determine which points lie inside or on the coil boundary.

        Parameters
        ----------
        x:
            The x coordinates to check
        z:
            The z coordinates to check
        atol:
            Add an offset, to ensure points very near the edge are counted as
            being on the edge of a coil

        Returns
        -------
        The Boolean array of point indices inside/outside the coil boundary
        """
        x, z = (
            np.ascontiguousarray(x)[..., np.newaxis],
            np.ascontiguousarray(z)[..., np.newaxis],
        )

        x_min, x_max = (
            self.x - self.dx - atol,
            self.x + self.dx + atol,
        )
        z_min, z_max = (
            self.z - self.dz - atol,
            self.z + self.dz + atol,
        )
        return (
            (x >= x_min[np.newaxis])
            & (x <= x_max[np.newaxis])
            & (z >= z_min[np.newaxis])
            & (z <= z_max[np.newaxis])
        )

    def _response_greens(
        self,
        greens: Callable,
        x: float | np.ndarray,
        z: float | np.ndarray,
        *,
        split: bool = False,
        _quad_x: np.ndarray | None = None,
        _quad_z: np.ndarray | None = None,
        _quad_weight: np.ndarray | None = None,
    ) -> float | np.ndarray:
        """
        Calculate [psi, Bx, Bz] response at (x, z) due to a unit
        current using Green's functions.

        Parameters
        ----------
        greens:
            greens function
        x:
            The x values at which to calculate the response at
        z:
            The z values at which to calculate the response at
        split:
            Flag for if :meth:_combined_control is used
        _quad_x:
            :meth:_combined_control x positions
        _quad_z:
            :meth:_combined_control z positions
        _quad_weight:
            :meth:_combined_control weighting

        Returns
        -------
        Greens response
        """
        if not split:
            _quad_x = self._quad_x
            _quad_z = self._quad_z
            _quad_weight = self._quad_weighting

        ind = np.nonzero(_quad_weight)
        out = np.zeros((*x.shape, *_quad_x.shape))

        out[(*(slice(None) for _ in x.shape), *ind)] = greens(
            _quad_x[ind][np.newaxis],
            _quad_z[ind][np.newaxis],
            x[..., np.newaxis],
            z[..., np.newaxis],
        )

        return np.squeeze(
            np.einsum(
                self._einsum_str,
                out,
                _quad_weight,
            )
        )

    def _response_analytical(
        self,
        semianalytic: Callable,
        x: np.ndarray,
        z: np.ndarray,
        *,
        split: bool = False,
        coil_x: np.ndarray | None = None,
        coil_z: np.ndarray | None = None,
        coil_dx: np.ndarray | None = None,
        coil_dz: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Calculate [psi, Bx, Bz] response at (x, z) due to a unit
        current using semi-analytic method.

        Parameters
        ----------
        semianalytic:
            semianalytic function
        x:
            The x values at which to calculate the response at
        z:
            The z values at which to calculate the response at
        split:
            Flag for if :meth:_combined_control is used
        coil_x:
            :meth:_combined_control x positions
        coil_z:
            :meth:_combined_control z positions
        coil_dx:
            :meth:_combined_control x positions
        coil_dz:
            :meth:_combined_control z positions

        Returns
        -------
        Analytical response
        """
        if not split:
            coil_x = self.x
            coil_z = self.z
            coil_dx = self.dx
            coil_dz = self.dz

        return np.squeeze(
            semianalytic(
                coil_x[np.newaxis],
                coil_z[np.newaxis],
                x[..., np.newaxis],
                z[..., np.newaxis],
                d_xc=coil_dx[np.newaxis],
                d_zc=coil_dz[np.newaxis],
            )
        )


class CoilSetFieldsMixin(CoilGroupFieldsMixin):
    """
    CoilSet magnetic fields mixin.

    Adjust output of coilgroup field calculations dealing with control coils
    or summing over coils
    """

    __slots__ = ()

    def psi(
        self,
        x: np.ndarray,
        z: np.ndarray,
        *,
        sum_coils: bool = True,
        control: bool = False,
    ) -> np.ndarray:
        """
        Psi of Coilset

        Parameters
        ----------
        x:
            The x values at which to calculate the psi response
        z:
            The z values at which to calculate the psi response
        sum_coils:
            sum over coils
        control:
            operations on control coils only

        Returns
        -------
        Poloidal magnetic flux density
        """
        return self._sum(super().psi(x, z), sum_coils=sum_coils, control=control)

    def Bx(
        self,
        x: np.ndarray,
        z: np.ndarray,
        *,
        sum_coils: bool = True,
        control: bool = False,
    ) -> np.ndarray:
        """
        Bx of Coilset

        Parameters
        ----------
        x:
            The x values at which to calculate the Bx response
        z:
            The z values at which to calculate the Bx response
        sum_coils:
            sum over coils
        control:
            operations on control coils only

        Returns
        -------
        Radial magnetic field
        """
        return self._sum(super().Bx(x, z), sum_coils=sum_coils, control=control)

    def Bz(
        self,
        x: np.ndarray,
        z: np.ndarray,
        *,
        sum_coils: bool = True,
        control: bool = False,
    ) -> np.ndarray:
        """
        Bz of Coilset

        Parameters
        ----------
        x:
            The x values at which to calculate the Bz response
        z:
            The z values at which to calculate the Bz response
        sum_coils:
            sum over coils
        control:
            operations on control coils only

        Returns
        -------
        Vertical magnetic field
        """
        return self._sum(super().Bz(x, z), sum_coils=sum_coils, control=control)

    def dB_d(
        self,
        x: np.ndarray,
        z: np.ndarray,
        *,
        sum_coils: bool = True,
        control: bool = False,
    ) -> np.ndarray:
        """
        dB_d of Coilset

        Parameters
        ----------
        x:
            The x values at which to calculate the dB_d response
        z:
            The z values at which to calculate the dB_d response
        sum_coils:
            sum over coils
        control:
            operations on control coils only

        Returns
        -------
        Differential of magnetic field
        """
        return self._sum(super().dB_d(x, z), sum_coils=sum_coils, control=control)

    def psi_response(
        self,
        x: np.ndarray,
        z: np.ndarray,
        *,
        sum_coils: bool = False,
        control: bool = False,
    ) -> np.ndarray:
        """
        Unit psi of Coilset

        Parameters
        ----------
        x:
            The x values at which to calculate the psi response
        z:
            The z values at which to calculate the psi response
        sum_coils:
            sum over coils
        control:
            operations on control coils only

        Returns
        -------
        Psi response
        """
        return self._sum(
            super().psi_response(x, z), sum_coils=sum_coils, control=control
        )

    def Bx_response(
        self,
        x: np.ndarray,
        z: np.ndarray,
        *,
        sum_coils: bool = False,
        control: bool = False,
    ) -> np.ndarray:
        """
        Unit Bx of Coilset

        Parameters
        ----------
        x:
            The x values at which to calculate the Bx response
        z:
            The z values at which to calculate the Bx response
        sum_coils:
            sum over coils
        control:
            operations on control coils only

        Returns
        -------
        Bx response
        """
        return self._sum(super().Bx_response(x, z), sum_coils=sum_coils, control=control)

    def Bz_response(
        self,
        x: np.ndarray,
        z: np.ndarray,
        *,
        sum_coils: bool = False,
        control: bool = False,
    ) -> np.ndarray:
        """
        Unit Bz of Coilset

        Parameters
        ----------
        x:
            The x values at which to calculate the Bz response
        z:
            The z values at which to calculate the Bz response
        sum_coils:
            sum over coils
        control:
            operations on control coils only

        Returns
        -------
        Bz response
        """
        return self._sum(super().Bz_response(x, z), sum_coils=sum_coils, control=control)

    def dB_d_response(
        self,
        x: np.ndarray,
        z: np.ndarray,
        *,
        sum_coils: bool = False,
        control: bool = False,
    ) -> np.ndarray:
        """
        Unit dB_d of Coilset

        Parameters
        ----------
        x:
            The x values at which to calculate the dB_d response
        z:
            The z values at which to calculate the dB_d response
        sum_coils:
            sum over coils
        control:
            operations on control coils only

        Returns
        -------
        :
            Differential of magnetic field response
        """
        return self._sum(
            super().dB_d_response(x, z), sum_coils=sum_coils, control=control
        )

    def control_F(self, coil_grp: CoilGroup, *, control: bool = False) -> np.ndarray:
        """
        Calculate the coil mutual force

        \t:math:`Fz_{i,j}=-2\\pi X_i\\mathcal{G}(X_j,Z_j,X_i,Z_i)`

        Parameters
        ----------
        coil_grp`:
            the coil group to calculate against
        control:
            operations on control coils only

        Returns
        -------
        :
            the Green's matrix element for the coil mutual force.
        """
        if control:
            inds = self._control_ind
            inds2 = coil_grp._control_ind
        else:
            inds = inds2 = slice(None)

        return super().control_F(coil_grp)[inds][:, inds2]

    def _stored_greens(
        self, bgreen: np.ndarray, *, sum_coils: bool = True, control: bool = False
    ) -> np.ndarray:
        """
        Uses the Greens mapped dict to quickly compute the Bx,Bz or psi

        Parameters
        ----------
        bgreen:
            The unit Bx response
        sum_coils:
            sum over coils
        control:
            operations on control coils only

        Returns
        -------
        :
            Cached Greens response
        """
        return self._sum(
            super()._stored_greens(bgreen), sum_coils=sum_coils, control=control
        )


class CoilFieldsMixin(CoilGroupFieldsMixin):
    """
    Coil magnetic fields mixin.

    Add field calculation mechanics to Coils
    """

    __slots__ = ()

    def _points_inside_coil(
        self,
        x: float | np.array,
        z: float | np.array,
        *,
        atol: float = X_TOLERANCE,
    ) -> np.ndarray:
        """
        Determine which points lie inside or on the coil boundary.

        Parameters
        ----------
        x:
            The x values to check
        z:
            The z values to check
        atol:
            Add an offset, to ensure points very near the edge are counted as
            being on the edge of a coil

        Returns
        -------
        The Boolean array of point indices inside/outside the coil boundary
        """
        x, z = (
            np.ascontiguousarray(x)[..., np.newaxis],
            np.ascontiguousarray(z)[..., np.newaxis],
        )

        x_min, x_max = (
            self.x - self.dx - atol,
            self.x + self.dx + atol,
        )
        z_min, z_max = (
            self.z - self.dz - atol,
            self.z + self.dz + atol,
        )
        return (x >= x_min) & (x <= x_max) & (z >= z_min) & (z <= z_max)

    def _combined_control(
        self,
        inside: np.ndarray,
        x: np.ndarray,
        z: np.ndarray,
        greens_func: Callable,
        semianalytic_func: Callable,
    ):
        """
        Combine semianalytic and greens function calculation of magnetic field

        Used for situation where there are calculation points both inside and
        outside the coil boundaries.

        Parameters
        ----------
        inside:
            array of if the point is inside a coil
        x:
            The x values at which to calculate the response at
        z:
            The z values at which to calculate the response at
        greens_func:
            greens function
        semianalytic_func:
            semianalytic function

        Returns
        -------
        Combined response
        """
        response = np.zeros(inside.shape[:-1])
        points = inside[..., 0]

        if np.any(~points):
            response[~points] = self._response_greens(
                greens_func, x[~points], z[~points]
            )

        if np.any(points):
            response[points] = self._response_analytical(
                semianalytic_func, x[points], z[points]
            )

        return response

    def _response_analytical(
        self,
        semianalytic: Callable,
        x: np.ndarray,
        z: np.ndarray,
        *_args,
        **_kwargs,
    ) -> np.ndarray:
        """
        Calculate [psi, Bx, Bz] response at (x, z) due to a unit
        current using semi-analytic method.

        Parameters
        ----------
        semianalytic:
            semianalytic function
        x:
            The x values at which to calculate the response at
        z:
            The z values at which to calculate the response at

        Returns
        -------
        Analytical response
        """
        return super()._response_analytical(
            semianalytic,
            x,
            z,
            split=True,
            coil_x=np.array([self.x]),
            coil_z=np.array([self.z]),
            coil_dx=np.array([self.dx]),
            coil_dz=np.array([self.dz]),
        )
