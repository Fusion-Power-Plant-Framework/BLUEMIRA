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

from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np

from bluemira.base.constants import MU_0
from bluemira.equilibria.constants import X_TOLERANCE
from bluemira.magnetostatics.greens import greens_Bx, greens_Bz, greens_psi
from bluemira.magnetostatics.semianalytic_2d import semianalytic_Bx, semianalytic_Bz

if TYPE_CHECKING:
    from bluemira.equilibria.coils._grouping import CoilGroup


class CoilGroupFieldsMixin:
    """
    CoilGroup magnetic fields mixin.

    Add field calculation mechanics to coilgroups
    """

    __slots__ = (
        "_quad_dx",
        "_quad_dz",
        "_quad_x",
        "_quad_z",
        "_quad_weighting",
        "_einsum_str",
    )

    def psi(self, x: Union[float, np.ndarray], z: Union[float, np.ndarray]):
        """
        Calculate poloidal flux at (x, z)
        """
        return self.psi_response(x, z) * self.current

    def _psi_greens(self, pgreen: Union[float, np.ndarray]):
        """
        Calculate plasma psi from Greens functions and current
        """
        return self.current * pgreen

    def psi_response(self, x: Union[float, np.ndarray], z: Union[float, np.ndarray]):
        """
        Calculate poloidal flux at (x, z) due to a unit current
        """
        x, z = np.ascontiguousarray(x), np.ascontiguousarray(z)

        ind = np.nonzero(self._quad_weighting)
        out = np.zeros((*x.shape, *self._quad_x.shape))

        out[(*(slice(None) for _ in x.shape), *ind)] = greens_psi(
            self._quad_x[ind][np.newaxis],
            self._quad_z[ind][np.newaxis],
            x[..., np.newaxis],
            z[..., np.newaxis],
            self._quad_dx[ind][np.newaxis],
            self._quad_dz[ind][np.newaxis],
        )

        return np.squeeze(
            np.einsum(
                self._einsum_str,
                out,
                self._quad_weighting[np.newaxis],
            )
        )

    def Bx(self, x: Union[float, np.ndarray], z: Union[float, np.ndarray]):
        """
        Calculate radial magnetic field Bx at (x, z)
        """
        return self.Bx_response(x, z) * self.current

    def _Bx_greens(self, bgreen: Union[float, np.ndarray]):
        """
        Uses the Greens mapped dict to quickly compute the Bx
        """
        return self.current * bgreen

    def Bx_response(
        self, x: Union[float, np.ndarray], z: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
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
        The radial magnetic field response at the x, z coordinates.
        """
        return self._mix_control_method(
            x, z, self._Bx_response_greens, self._Bx_response_analytical
        )

    def Bz(
        self, x: Union[float, np.ndarray], z: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Calculate vertical magnetic field Bz at (x, z)
        """
        return self.Bz_response(x, z) * self.current

    def _Bz_greens(self, bgreen: Union[float, np.ndarray]):
        """
        Uses the Greens mapped dict to quickly compute the Bx
        """
        return self.current * bgreen

    def Bz_response(
        self, x: Union[float, np.ndarray], z: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
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
        The vertical magnetic field response at the x, z coordinates.
        """
        return self._mix_control_method(
            x, z, self._Bz_response_greens, self._Bz_response_analytical
        )

    def Bp(self, x: Union[float, np.ndarray], z: Union[float, np.ndarray]):
        """
        Calculate poloidal magnetic field Bp at (x, z)
        """
        return np.hypot(self.Bx(x, z), self.Bz(x, z))

    def _mix_control_method(
        self,
        x: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        greens_func: Callable,
        semianalytic_func: Callable,
    ) -> Union[float, np.ndarray]:
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
            if np.all(~inside):
                return greens_func(x, z)
            if np.all(inside):
                # Not called for circuits as they will always be a mixture
                return semianalytic_func(x, z)
            return self._combined_control(inside, x, z, greens_func, semianalytic_func)
        return greens_func(x, z)

    def _combined_control(
        self,
        inside: np.ndarray,
        x: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        greens_func: Callable,
        semianalytic_func: Callable,
    ) -> Union[float, np.ndarray]:
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
            )
        ):
            if np.any(~points):
                response[~points, coil] = greens_func(
                    x[~points], z[~points], True, qx, qz, qw
                )

            if np.any(points):
                response[points, coil] = semianalytic_func(
                    x[points], z[points], True, cx, cz, cdx, cdz
                )

        return np.squeeze(response)

    def _points_inside_coil(
        self,
        x: Union[float, np.array],
        z: Union[float, np.array],
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

    def _B_response_greens(
        self,
        greens: Callable,
        x: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        split: bool = False,
        _quad_x: Optional[np.ndarray] = None,
        _quad_z: Optional[np.ndarray] = None,
        _quad_weight: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        """
        Calculate magnetic field B response at (x, z) due to a unit
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
            Flag for if :func:_combined_control is used
        _quad_x:
            :func:_combined_control x positions
        _quad_z:
            :func:_combined_control z positions
        _quad_weight:
            :func:_combined_control weighting

        Returns
        -------
        Magnetic field response
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

    def _Bx_response_greens(
        self,
        x: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        split: bool = False,
        _quad_x: Optional[np.ndarray] = None,
        _quad_z: Optional[np.ndarray] = None,
        _quad_weight: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        """
        Calculate radial magnetic field Bx response at (x, z) due to a unit
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
            Flag for if :func:_combined_control is used
        _quad_x:
            :func:_combined_control x positions
        _quad_z:
            :func:_combined_control z positions
        _quad_weight:
            :func:_combined_control weighting

        Returns
        -------
        Radial magnetic field response
        """
        return self._B_response_greens(
            greens_Bx, x, z, split, _quad_x, _quad_z, _quad_weight
        )

    def _Bz_response_greens(
        self,
        x: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        split: bool = False,
        _quad_x: Optional[np.ndarray] = None,
        _quad_z: Optional[np.ndarray] = None,
        _quad_weight: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        """
        Calculate vertical magnetic field Bz at (x, z) due to a unit current

        Parameters
        ----------
        greens:
            greens function
        x:
            The x values at which to calculate the response at
        z:
            The z values at which to calculate the response at
        split:
            Flag for if :func:_combined_control is used
        _quad_x:
            :func:_combined_control x positions
        _quad_z:
            :func:_combined_control z positions
        _quad_weight:
            :func:_combined_control weighting

        Returns
        -------
        Vertical magnetic field response
        """
        return self._B_response_greens(
            greens_Bz, x, z, split, _quad_x, _quad_z, _quad_weight
        )

    def _B_response_analytical(
        self,
        semianalytic: Callable,
        x: np.ndarray,
        z: np.ndarray,
        split: bool = False,
        coil_x: Optional[np.ndarray] = None,
        coil_z: Optional[np.ndarray] = None,
        coil_dx: Optional[np.ndarray] = None,
        coil_dz: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calculate magnetic field Bx response at (x, z) due to a unit
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
            Flag for if :func:_combined_control is used
        coil_x:
            :func:_combined_control x positions
        coil_z:
            :func:_combined_control z positions
        coil_dx:
            :func:_combined_control x positions
        coil_dz:
            :func:_combined_control z positions

        Returns
        -------
        Magnetic field response
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

    def _Bx_response_analytical(
        self,
        x: np.ndarray,
        z: np.ndarray,
        split: bool = False,
        coil_x: Optional[np.ndarray] = None,
        coil_z: Optional[np.ndarray] = None,
        coil_dx: Optional[np.ndarray] = None,
        coil_dz: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calculate radial magnetic field Bx response at (x, z) due to a unit
        current using semi-analytic method.

        Parameters
        ----------
        x:
            The x values at which to calculate the response at
        z:
            The z values at which to calculate the response at
        split:
            Flag for if :func:_combined_control is used
        coil_x:
            :func:_combined_control x positions
        coil_z:
            :func:_combined_control z positions
        coil_dx:
            :func:_combined_control x positions
        coil_dz:
            :func:_combined_control z positions

        Returns
        -------
        Radial magnetic field response
        """
        return self._B_response_analytical(
            semianalytic_Bx, x, z, split, coil_x, coil_z, coil_dx, coil_dz
        )

    def _Bz_response_analytical(
        self,
        x: np.ndarray,
        z: np.ndarray,
        split: bool = False,
        coil_x: Optional[np.ndarray] = None,
        coil_z: Optional[np.ndarray] = None,
        coil_dx: Optional[np.ndarray] = None,
        coil_dz: Optional[np.ndarray] = None,
    ):
        """
        Calculate vertical magnetic field Bz response at (x, z) due to a unit
        current using semi-analytic method.

        Parameters
        ----------
        x:
            The x values at which to calculate the response at
        z:
            The z values at which to calculate the response at
        split:
            Flag for if :func:_combined_control is used
        coil_x:
            :func:_combined_control x positions
        coil_z:
            :func:_combined_control z positions
        coil_dx:
            :func:_combined_control x positions
        coil_dz:
            :func:_combined_control z positions

        Returns
        -------
        Vertical magnetic field response
        """
        return self._B_response_analytical(
            semianalytic_Bz, x, z, split, coil_x, coil_z, coil_dx, coil_dz
        )

    def F(self, eqcoil: CoilGroup) -> np.ndarray:
        """
        Calculate the force response at the coil centre including the coil
        self-force.

        .... math::

             \\mathbf{F} = \\mathbf{j}\\times \\mathbf{B}
            F_x = IB_z+\\dfrac{\\mu_0I^2}{4\\pi X}\\textrm{ln}\\bigg(\\dfrac{8X}{r_c}-1+\\xi/2\\bigg)
            F_z = -IBx
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

        return np.array(
            [
                multiplier * (eqcoil.Bz(self.x, self.z) + fx),
                -multiplier * eqcoil.Bx(self.x, self.z),
            ]
        ).T

    def control_F(self, coil: CoilGroup) -> np.ndarray:
        """
        Returns the Green's matrix element for the coil mutual force.

        \t:math:`Fz_{i,j}=-2\\pi X_i\\mathcal{G}(X_j,Z_j,X_i,Z_i)`
        """
        # TODO Vectorise
        x, z = np.atleast_1d(self.x), np.atleast_1d(self.z)  # single coil
        pos = np.array([x, z])
        response = np.zeros((x.size, coil.x.size, 2))
        coils = coil._coils
        for j, coil2 in enumerate(coils):
            xw = np.nonzero(x == coil2.x)[0]
            zw = np.nonzero(z == coil2.z)[0]
            same_pos = np.nonzero(xw == zw)[0]
            if same_pos.size > 0:
                # self inductance
                xxw = xw[same_pos]
                cr = self._current_radius[xxw]
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
                    Bz[~mask] = coil2.Bz_response(*pos[:, ~mask[:, 0]])
                    Bx[~mask] = coil2.Bx_response(*pos[:, ~mask[:, 0]])

            else:
                Bz = coil2.Bz_response(x, z)
                Bx = coil2.Bx_response(x, z)
            # 1 cross B
            response[:, j, :] = (
                2 * np.pi * x[:, np.newaxis] * np.squeeze(np.array([Bz, -Bx]).T)
            )
        return response


class CoilSetFieldsMixin(CoilGroupFieldsMixin):
    """
    CoilSet magnetic fields mixin.

    Adjust output of coilgroup field calculations dealing with control coils
    or summing over coils
    """

    __slots__ = ()

    def psi(
        self, x: np.ndarray, z: np.ndarray, sum_coils: bool = True, control: bool = False
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
        self, x: np.ndarray, z: np.ndarray, sum_coils: bool = True, control: bool = False
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
        self, x: np.ndarray, z: np.ndarray, sum_coils: bool = True, control: bool = False
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

    def psi_response(
        self,
        x: np.ndarray,
        z: np.ndarray,
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
        sum_coils: bool = False,
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
        Bz response
        """
        return self._sum(super().Bz_response(x, z), sum_coils=sum_coils, control=control)

    def _psi_greens(
        self, psigreens: np.ndarray, sum_coils: bool = True, control: bool = False
    ) -> np.ndarray:
        """
        Uses the Greens mapped dict to quickly compute the psi

        Parameters
        ----------
        psigreens:
            The unit psi response
        sum_coils:
            sum over coils
        control:
            operations on control coils only

        Returns
        -------
        Cached Greens psi response
        """
        return self._sum(
            super()._psi_greens(psigreens), sum_coils=sum_coils, control=control
        )

    def _Bx_greens(
        self, bgreen: np.ndarray, sum_coils: bool = True, control: bool = False
    ) -> np.ndarray:
        """
        Uses the Greens mapped dict to quickly compute the Bx

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
        Cached Greens Bx response
        """
        return self._sum(
            super()._Bx_greens(bgreen), sum_coils=sum_coils, control=control
        )

    def _Bz_greens(
        self, bgreen: np.ndarray, sum_coils: bool = True, control: bool = False
    ) -> np.ndarray:
        """
        Uses the Greens mapped dict to quickly compute the Bz

        Parameters
        ----------
        bgreen:
            The unit Bz response
        sum_coils:
            sum over coils
        control:
            operations on control coils only

        Returns
        -------
        Cached Greens Bs response
        """
        return self._sum(
            super()._Bz_greens(bgreen), sum_coils=sum_coils, control=control
        )


class CoilFieldsMixin(CoilGroupFieldsMixin):
    """
    Coil magnetic fields mixin.

    Add field calculation mechanics to Coils
    """

    __slots__ = ()

    def _points_inside_coil(
        self,
        x: Union[float, np.array],
        z: Union[float, np.array],
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

    @staticmethod
    def _combined_control(
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
            response[~points] = greens_func(x[~points], z[~points])

        if np.any(points):
            response[points] = semianalytic_func(x[points], z[points])

        return response

    def _B_response_analytical(
        self,
        semianalytic: Callable,
        x: np.ndarray,
        z: np.ndarray,
        *_args,
        **_kwargs,
    ):
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
        return super()._B_response_analytical(
            semianalytic,
            x,
            z,
            split=True,
            coil_x=np.array([self.x]),
            coil_z=np.array([self.z]),
            coil_dx=np.array([self.dx]),
            coil_dz=np.array([self.dz]),
        )
