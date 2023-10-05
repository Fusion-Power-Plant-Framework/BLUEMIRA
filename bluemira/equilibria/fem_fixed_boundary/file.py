# %%
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
File saving for fixed boundary equilibrium
"""
from typing import Dict, Optional
from warnings import warn

import numpy as np
from scipy.integrate import quad, quadrature

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FixedBoundaryEquilibrium,
)
from bluemira.equilibria.fem_fixed_boundary.utilities import (
    _interpolate_profile,
    find_magnetic_axis,
    get_mesh_boundary,
)
from bluemira.equilibria.file import EQDSKInterface
from bluemira.equilibria.grid import Grid


def _pressure_profile(pprime, psi_norm, psi_mag):
    pressure = np.zeros(len(psi_norm))
    for i in range(len(psi_norm)):
        pressure[i] = quad(pprime, psi_norm[i], 1.0, limit=500)[0] * psi_mag
    return pressure


def _fpol_profile(ffprime, psi_norm, psi_mag, fvac):
    fpol = np.zeros(len(psi_norm))
    for i in range(len(psi_norm)):
        fpol[i] = np.sqrt(
            2
            * quadrature(ffprime, psi_norm[i], 1.0, maxiter=500, rtol=1e-6, tol=1e-6)[0]
            * psi_mag
            + fvac**2
        )
    return fpol


def save_fixed_boundary_to_file(
    file_path: str,
    file_header_name: str,
    equilibrium: FixedBoundaryEquilibrium,
    nx: int,
    nz: int,
    file_format: str = "json",
    json_kwargs: Optional[Dict] = None,
    **kwargs,
):
    """
    Save a fixed boundary equilibrium to a file.

    Parameters
    ----------
    file_path:
        File path to save the file to
    file_header_name:
        File header name to use in the the file
    equilibrium:
        Equilibrium object to save to file
    nx:
        Number of radial points to use in the psi map
    nz:
        Number of vertical points to use in the psi map
    file_format:
        Format of the file
    json_kwargs:
        kwargs to use if saving to JSON
    """
    if kw_formatt := kwargs.pop("formatt", None):
        warn(
            "Using kwarg 'formatt' is no longer supported. Use file_format instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        file_format = kw_formatt

    xbdry, zbdry = get_mesh_boundary(equilibrium.mesh)
    xbdry = np.append(xbdry, xbdry[0])
    zbdry = np.append(zbdry, zbdry[0])
    nbdry = len(xbdry)

    x_mag, z_mag = find_magnetic_axis(equilibrium.psi, equilibrium.mesh)
    psi_mag = equilibrium.psi(x_mag, z_mag)

    # Make a minimum grid
    x_min = np.min(xbdry)
    x_max = np.max(xbdry)
    z_min = np.min(zbdry)
    z_max = np.max(zbdry)
    grid = Grid(x_min, x_max, z_min, z_max, nx=nx, nz=nz)

    psi = np.zeros((nx, nz))
    for i, xi in enumerate(grid.x_1d):
        for j, zj in enumerate(grid.z_1d):
            psi[i, j] = equilibrium.psi([xi, zj])

    p_prime = equilibrium.p_prime
    ff_prime = equilibrium.ff_prime
    psi_norm = np.linspace(0, 1, len(ff_prime))

    p_prime_func = _interpolate_profile(psi_norm, p_prime)
    ff_prime_func = _interpolate_profile(psi_norm, ff_prime)

    if equilibrium.R_0 is None:
        bluemira_warn(
            "No explicit R_0 information provided when saving fixed boundary equilibrium"
            " to file. Taking the average of the boundary radial coordinate extrema."
        )
        R_0 = grid.x_mid
    else:
        R_0 = equilibrium.R_0
    if equilibrium.B_0 is None:
        bluemira_warn(
            "No explicit B_0 information provided when saving fixed boundary equilibrium"
            " to file. Setting to 0!"
        )
        B_0 = 0.0
    else:
        B_0 = equilibrium.B_0

    fvac = R_0 * B_0
    psi_vector = psi_norm * psi_mag
    pressure = _pressure_profile(p_prime_func, psi_vector, psi_mag)
    fpol = _fpol_profile(ff_prime_func, psi_norm, psi_mag, fvac)

    data = EQDSKInterface(
        bcentre=B_0,
        cplasma=equilibrium.I_p,
        dxc=np.array([]),
        dzc=np.array([]),
        ffprime=ff_prime,
        fpol=fpol,
        Ic=np.array([]),
        name=file_header_name,
        nbdry=nbdry,
        ncoil=0,
        nlim=0,
        nx=nx,
        nz=nz,
        pressure=pressure,
        pprime=p_prime,
        psi=psi,
        psibdry=np.zeros(nbdry),
        psimag=psi_mag,
        xbdry=xbdry,
        xc=np.array([]),
        xcentre=grid.x_mid,
        xdim=grid.x_size,
        xgrid1=grid.x_min,
        xlim=np.array([]),
        xmag=x_mag,
        zbdry=zbdry,
        zc=np.array([]),
        zdim=grid.z_size,
        zlim=np.array([]),
        zmag=z_mag,
        zmid=grid.z_mid,
        x=grid.x_1d,
        z=grid.z_1d,
        psinorm=psi_norm,
        qpsi=np.array([]),
    )
    data.write(file_path, file_format=file_format, json_kwargs=json_kwargs)
    return data
