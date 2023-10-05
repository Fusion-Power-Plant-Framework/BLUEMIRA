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

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from bluemira.base.constants import EPS
from bluemira.base.file import get_bluemira_path
from bluemira.equilibria import Equilibrium
from bluemira.equilibria.error import EquilibriaError, FluxSurfaceError
from bluemira.equilibria.find import find_flux_surface_through_point
from bluemira.equilibria.flux_surfaces import (
    ClosedFluxSurface,
    FieldLineTracer,
    OpenFluxSurface,
    PartialOpenFluxSurface,
    poloidal_angle,
)
from bluemira.equilibria.shapes import flux_surface_cunningham, flux_surface_johner
from bluemira.geometry.coordinates import (
    Coordinates,
    coords_plane_intersect,
    interpolate_points,
)
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import _signed_distance_2D

TEST_PATH = get_bluemira_path("equilibria/test_data", subfolder="tests")


class TestOpenFluxSurfaceStuff:
    @classmethod
    def setup_class(cls):
        eq_name = "eqref_OOB.json"
        cls.eq = Equilibrium.from_eqdsk(Path(TEST_PATH, eq_name))

    def test_bad_geometry(self):
        closed_coords = Coordinates({"x": [0, 4, 5, 8, 0], "z": [1, 2, 3, 4, 1]})
        with pytest.raises(FluxSurfaceError):
            _ = OpenFluxSurface(closed_coords)
        with pytest.raises(FluxSurfaceError):
            _ = PartialOpenFluxSurface(closed_coords)

    def test_connection_length(self):
        """
        Use both a flux surface and field line tracing approach to calculate connection
        length and check they are the same or similar.
        """
        x_start, z_start = 12, 0
        x_loop, z_loop = find_flux_surface_through_point(
            self.eq.x,
            self.eq.z,
            self.eq.psi(),
            x_start,
            z_start,
            self.eq.psi(x_start, z_start),
        )
        fs = OpenFluxSurface(Coordinates({"x": x_loop, "z": z_loop}))
        lfs, hfs = fs.split(self.eq.get_OX_points()[0][0])
        l_lfs = lfs.connection_length(self.eq)
        l_hfs = hfs.connection_length(self.eq)

        # test discretisation sensitivity
        lfs_loop = deepcopy(lfs.coords)
        lfs_loop = Coordinates(interpolate_points(*lfs_loop.xyz, 3 * len(lfs_loop)))
        lfs_interp = PartialOpenFluxSurface(lfs_loop)
        l_lfs_interp = lfs_interp.connection_length(self.eq)
        assert np.isclose(l_lfs, l_lfs_interp, rtol=5e-3)

        hfs_loop = deepcopy(hfs.coords)
        hfs_loop = Coordinates(interpolate_points(*hfs_loop.xyz, 3 * len(hfs_loop)))
        hfs_interp = PartialOpenFluxSurface(hfs_loop)
        l_hfs_interp = hfs_interp.connection_length(self.eq)
        assert np.isclose(l_hfs, l_hfs_interp, rtol=5e-3)

        # compare with field line tracer
        flt = FieldLineTracer(self.eq)
        l_flt_lfs = flt.trace_field_line(x_start, z_start, n_turns_max=20, forward=True)
        l_flt_hfs = flt.trace_field_line(
            x_start, z_start, n_turns_max=20, forward=False
        ).connection_length
        print(len(l_flt_lfs.coords))
        assert np.isclose(l_flt_lfs.connection_length, l_lfs, rtol=2e-2)
        assert np.isclose(l_flt_hfs, l_hfs, rtol=2e-2)


class TestClosedFluxSurface:
    def test_bad_geometry(self):
        open_loop = Coordinates({"x": [0, 4, 5, 8], "z": [1, 2, 3, 4]})
        with pytest.raises(FluxSurfaceError):
            _ = ClosedFluxSurface(open_loop)

    def test_symmetric(self):
        kappa = 1.5
        delta = 0.4
        # Note that n=1000%4 == 0
        fs = flux_surface_cunningham(7, 0, 1, kappa, delta, n=1000)
        fs.close()
        fs = ClosedFluxSurface(fs)
        assert np.isclose(fs.kappa, kappa)
        assert np.isclose(fs.kappa_lower, kappa)
        assert np.isclose(fs.kappa_upper, kappa)
        assert np.isclose(fs.delta_lower, fs.delta_upper)
        assert np.isclose(fs.zeta_lower, fs.zeta_upper)

    def test_johner(self):
        R_0, z_0, a, kappa_u, kappa_l, delta_u, delta_l, a1, a2, a3, a4 = (
            7,
            0,
            2,
            1.9,
            1.6,
            0.4,
            0.33,
            0,
            0,
            45,
            30,
        )
        fs = flux_surface_johner(
            R_0, z_0, a, kappa_u, kappa_l, delta_u, delta_l, a1, a2, a3, a4, n=1000
        )
        fs.close()
        fs = ClosedFluxSurface(fs)

        assert np.isclose(fs.major_radius, R_0)
        assert np.isclose(fs._z_centre, z_0)
        assert np.isclose(fs.minor_radius, a)
        assert np.isclose(fs.kappa, np.average([kappa_l, kappa_u]))
        assert np.isclose(fs.kappa_upper, kappa_u)
        assert np.isclose(fs.kappa_lower, kappa_l)
        assert np.isclose(fs.delta, np.average([delta_l, delta_u]))
        assert np.isclose(fs.delta_upper, delta_u)
        assert np.isclose(fs.delta_lower, delta_l)
        assert not np.isclose(fs.zeta_upper, fs.zeta_lower)


class TestFieldLine:
    @classmethod
    def setup_class(cls):
        eq_name = "eqref_OOB.json"
        cls.eq = Equilibrium.from_eqdsk(Path(TEST_PATH, eq_name))
        cls.flt = FieldLineTracer(cls.eq)
        cls.field_line = cls.flt.trace_field_line(13, 0, n_points=1000)

    def test_non_planar_coodinates_raises_error(self):
        with pytest.raises(EquilibriaError):
            FieldLineTracer(
                self.eq,
                Coordinates(
                    {"x": [6, 3, 3, 4, 5], "y": [0, 2, 0, 4, 0], "z": [1, 2, 3, 4, 5]}
                ),
            )

    def test_connection_length(self):
        assert np.isclose(
            self.field_line.connection_length, self.field_line.coords.length, rtol=5e-2
        )

    def test_connection_length_coordinates_grid(self):
        """
        Check to see behaviour is the same with Coordinates and Grid
        """
        xmin, xmax = self.eq.grid.x_min, self.eq.grid.x_max
        zmin, zmax = self.eq.grid.z_min, self.eq.grid.z_max
        coords = Coordinates(
            {
                "x": [xmin, xmax, xmax, xmin, xmin],
                "y": 0,
                "z": [zmin, zmin, zmax, zmax, zmin],
            }
        )
        flt = FieldLineTracer(self.eq, coords)
        field_line = flt.trace_field_line(13, 0, n_points=1000)
        assert np.isclose(
            field_line.connection_length, field_line.coords.length, rtol=5e-2
        )
        assert np.isclose(
            self.field_line.connection_length, field_line.connection_length
        )
        self._check_endpoint(field_line, coords)

    def test_connection_length_coordinates(self):
        coords = Coordinates(
            {
                "x": [self.eq.grid.x_min, 9, 12, 13, 13, 12.5, 4, self.eq.grid.x_min],
                "y": 0,
                "z": [self.eq.grid.z_min, -7, -7, -6, 6, 5, 7, self.eq.grid.z_min],
            }
        )
        flt = FieldLineTracer(self.eq, coords)
        field_line = flt.trace_field_line(12.5, 0, n_points=1000, forward=False)
        self._check_endpoint(field_line, coords)

    def _check_endpoint(self, field_line, coords, tol=1e-8):
        """
        Check that the end point of a field line lies near enough to the boundary
        """
        end_point = self._extract_endpoint(field_line)
        ep_xz = np.array([end_point[0], end_point[2]])
        assert abs(_signed_distance_2D(ep_xz, coords.xz.T)) < tol

    def _extract_endpoint(self, field_line):
        """
        Get the end point of a field line in 3-D and map it to 2-D
        """
        end_point = field_line.coords.xyz.T[-1]
        r = np.hypot(*end_point[:2])
        z = end_point[2]
        angle = np.linspace(0, 2 * np.pi, 1000)
        circle = Coordinates({"x": r * np.cos(angle), "y": r * np.sin(angle), "z": z})
        inters = coords_plane_intersect(circle, BluemiraPlane(axis=(0, 1, 0)))
        return next(i for i in inters if i[0] > 0)


def test_poloidal_angle():
    eq_name = "DN-DEMO_eqref.json"
    eq = Equilibrium.from_eqdsk(Path(TEST_PATH, eq_name))
    # Building inputs
    x_strike = 10.0
    z_strike = -7.5
    Bp_strike = eq.Bp(x_strike, z_strike)
    Bt_strike = eq.Bt(x_strike)
    # Glancing angle
    gamma = 5.0
    # Poloidal angle
    theta = poloidal_angle(Bp_strike, Bt_strike, gamma)
    assert theta > gamma
    # By hand, from a different calculation
    assert round(theta, 1) == pytest.approx(20.6, rel=0, abs=EPS)
