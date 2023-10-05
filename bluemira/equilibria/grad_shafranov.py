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
Grad-Shafranov operator classes
"""
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import factorized

from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.grid import Grid

__all__ = ["GSSolver"]


class GSOperator:
    """
    Calculates sparse matrix for the Grad-Shafranov operator

    Parameters
    ----------
    x_min:
        Minimum X value on which to calculate the G-S operator
    x_max:
        Maximum X value on which to calculate the G-S operator
    z_min:
        Minimum Z value on which to calculate the G-S operator
    z_max:
        Maximum Z value on which to calculate the G-S operator
    force_symmetry:
        If true, the G-S operator will be constructed for the
        lower half space Z<=0 with symmetry conditions imposed
        at Z=0.

    \t:math:`{\\Delta}^{*} = X^{2}{\\nabla}{\\cdot}\\dfrac{1}{X^2}{\\nabla}`

    \t:math:`\\dfrac{1}{(\\Delta Z)^2}\\psi_{i-1, j}+\\bigg(\\dfrac{1}{(\\Delta X)^2}+\\dfrac{1}{2X_j(\\Delta X)}\\bigg)\\psi_{i, j-1}`
    \t:math:`+\\Bigg[2\\Bigg(\\dfrac{1}{(\\Delta X)^2}+\\dfrac{1}{(\\Delta Z)^2}\\Bigg)\\Bigg]\\psi_{i, j}`
    \t:math:`+\\bigg(\\dfrac{1}{(\\Delta X)^2}-\\dfrac{1}{2X_j(\\Delta X)}\\bigg)\\psi_{i, j+1}`
    \t:math:`+\\dfrac{1}{(\\Delta Z)^2}\\psi_{i+1, j}=-\\mu_0 X_j J_{\\phi_{i, j}}`
    """  # noqa: W505, E501

    def __init__(
        self,
        x_min: float,
        x_max: float,
        z_min: float,
        z_max: float,
        force_symmetry: bool = False,
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.z_min = z_min
        self.z_max = z_max
        self.force_symmetry = force_symmetry

    def __call__(self, nx: int, nz: int) -> csr_matrix:
        """
        Create a sparse matrix with given resolution

        Parameters
        ----------
        nx:
            The discretisation of the 2-D field in X
        nz:
            The discretisation of the 2-D field in Z
        """
        d_x = (self.x_max - self.x_min) / (nx - 1)
        d_z = (self.z_max - self.z_min) / (nz - 1)
        half_xdx = 0.5 / (np.linspace(self.x_min, self.x_max, nx) * d_x)
        inv_dx_2, inv_dz_2 = 1 / d_x**2, 1 / d_z**2

        if self.force_symmetry:
            # Check if applied grid is symmetric
            if not np.isclose(self.z_min, -self.z_max):
                raise EquilibriaError(
                    "Symmetry is being forced, but underlying limits are not symmetric"
                    " about z=0."
                )
            # Replace nz with number of rows in lower half space of system
            # (including midplane)
            nz = np.floor_divide(nz + 1, 2)

        A = lil_matrix((nx * nz, nx * nz))
        A.setdiag(1)

        i, j = np.meshgrid(np.arange(1, nx - 1), np.arange(1, nz - 1), indexing="ij")
        i = i.ravel()
        j = j.ravel()
        ind = i * nz + j

        A[ind, ind] = -2 * (inv_dx_2 + inv_dz_2)  # j, l
        A[ind, ind + nz] = inv_dx_2 - half_xdx[i]  # j, l-1
        A[ind, ind - nz] = inv_dx_2 + half_xdx[i]  # j, l+1
        A[ind, ind + 1] = inv_dz_2  # j-1, l
        A[ind, ind - 1] = inv_dz_2  # j+1, l

        # Apply symmetry boundary if desired
        if self.force_symmetry:
            # Apply ghost point method to apply symmetry to (d/dz^2) operator
            # contributions close to symmetry plane.
            # If symmetry boundary is centred halfway between cells,
            # d(psi)/dz = 0 across midplane,
            # else if symmetry boundary is centred on cells, d(psi)/dz
            # is equal either side of midplane but reverses sign.
            ind = i * nz + nz - 1
            ghost_factor = 1 + nz % 2

            A[ind, ind] = -2 * inv_dx_2 - ghost_factor * inv_dz_2
            A[ind, ind - 1] = ghost_factor * inv_dz_2
            A[ind, ind + nz] = inv_dx_2 - half_xdx[i]  # j, l-1
            A[ind, ind - nz] = inv_dx_2 + half_xdx[i]  # j, l+1
        return A.tocsr()  # Compressed sparse row format


class DirectSolver:
    """
    Direct solver applying lower-upper decomposition to solver
    the linear system A x = b.

    Parameters
    ----------
    A:
        Linear operator on LHS of equation system, in csr format.
    """

    def __init__(self, A: csr_matrix):
        self.solve = factorized(A.tocsc())

    def __call__(self, b: np.ndarray) -> np.ndarray:
        """
        Solve the matrix problem by LU decomposition.

        Parameters
        ----------
        b:
            Numpy array containing RHS of equation system.
            Converted to 1D before linear solve applied..
        """
        b1d = np.reshape(b, -1)
        x = self.solve(b1d)
        if np.any(np.isnan(x)):
            raise EquilibriaError("Matrix inversion problem in GS solver.")
        return np.reshape(x, b.shape)


class GSSolver(DirectSolver):
    """
    Solver for the Grad-Shafranov system.
    Uses lower-upper decomposition during linear solve.

    Parameters
    ----------
    grid:
        The grid upon which to solve the G-S equation.
    force_symmetry:
        If true, the G-S operator will be constructed for the
        lower half space Z<=0 with symmetry conditions imposed
        at Z=0.
    """

    def __init__(self, grid: Grid, force_symmetry: bool = False):
        self.grid = grid
        self.force_symmetry = force_symmetry

        gsoperator = GSOperator(
            self.grid.x_min,
            self.grid.x_max,
            self.grid.z_min,
            self.grid.z_max,
            force_symmetry=self.force_symmetry,
        )

        super().__init__(gsoperator(self.grid.nx, self.grid.nz))

    def __call__(self, b: np.ndarray) -> np.ndarray:
        """
        Solves the linear system Ax=b using LU decomposition,
        If the G-S operator is in symmetric form, problem symmetry
        is explicitly enforced.

        Parameters
        ----------
        b: np.array(nx, nz)
            2-D X, Z map of the RHS of the G-S equation.
        """
        if self.force_symmetry:
            nz = self.grid.nz

            # Trim RHS vector to half vector containing unique values
            half_nz = np.floor_divide(nz + 1, 2)
            b_half = b[:, 0:half_nz]

            # Solve linear system
            x_half = super().__call__(b_half)

            # Create full-length vector by symmetry
            x = np.zeros(np.shape(b))
            x[:, 0:half_nz] = x_half
            x[:, half_nz:] = np.flip(x_half, axis=1)[:, nz % 2 :]

        else:
            # Solve linear system with no symmetry assumptions.
            x = super().__call__(b)

        return x
