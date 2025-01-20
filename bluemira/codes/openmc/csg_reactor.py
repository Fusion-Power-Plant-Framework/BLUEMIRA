# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Putting together csg components to form a csg reactor model in the OpenMC universe."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np
import openmc
import openmc.region

from bluemira.codes.openmc.make_cell import BlanketCellArray, DivertorCellArray
from bluemira.codes.openmc.material import CellType
from bluemira.codes.openmc.tools import (
    OpenMCEnvironment,
    flat_intersection,
    flat_union,
    round_up_next_openmc_ids,
)
from bluemira.geometry.constants import D_TOLERANCE
from bluemira.geometry.error import GeometryError
from bluemira.geometry.tools import (
    is_convex,
    polygon_revolve_signed_volume,
)
from bluemira.radiation_transport.neutronics.constants import to_cm, to_cm3

if TYPE_CHECKING:
    import numpy.typing as npt

    from bluemira.codes.openmc.material import MaterialsLibrary
    from bluemira.radiation_transport.neutronics.neutronics_axisymmetric import (
        NeutronicsReactor,
    )


@dataclass
class CSGComponents:
    """A container for each component represented in CSG geometry."""

    blanket: BlanketCellArray
    divertor: DivertorCellArray
    tf_coils: list[openmc.Cell]
    cs_coil: openmc.Cell
    plasma: openmc.Cell
    radiation_shield: openmc.Cell
    ext_void: openmc.Cell
    universe: openmc.region.Intersection

    @property
    def cells(self):
        """Get the list of all cells."""
        return (
            *chain.from_iterable((*self.blanket, *self.divertor)),
            *self.tf_coils,
            self.cs_coil,
            self.plasma,
            self.radiation_shield,
            self.ext_void,
        )

    def get_all_hollow_merged_cells(self):
        """Blanket and divertor cells"""
        return [
            *[openmc.Cell(region=stack.get_overall_region()) for stack in self.blanket],
            *[openmc.Cell(region=stack.get_overall_region()) for stack in self.divertor],
        ]

    def set_volumes(self):
        """
        Sets the volume of the voids. Not necessary/ used anywhere yet.
        """
        ext_vertices = self.get_exterior_vertices()
        total_universe_volume = (
            #  top - bottom
            (self.universe[0].surface.z0 - self.universe[1].surface.z0)
            * np.pi
            * self.universe[2].surface.r ** 2  # cylinder
        )  # cm^3

        # is this needed?
        # self.universe.volume = total_universe_volume

        outer_boundary_volume = to_cm3(
            polygon_revolve_signed_volume(ext_vertices[:, ::2])
        )
        ext_void_volume = total_universe_volume - outer_boundary_volume
        if self.tf_coils:
            for coil in self.tf_coils:
                ext_void_volume -= coil.volume
        if self.cs_coil:
            ext_void_volume -= self.cs_coil.volume
        self.ext_void.volume = ext_void_volume
        blanket_volumes = sum(cell.volume for cell in chain.from_iterable(self.blanket))
        divertor_volumes = sum(
            cell.volume for cell in chain.from_iterable(self.divertor)
        )
        self.plasma.volume = outer_boundary_volume - blanket_volumes - divertor_volumes

    def get_exterior_vertices(self) -> npt.NDArray:
        """
        Get the 3D coordinates of every point at the outer boundary of the tokamak's
        poloidal cross-section.

        Returns
        -------
        coordinates
            array of shape (N+1+n*M, 3), where N = number of blanket pre-cells,
            M = number of divertor pre-cells, n = discretisation_level used when chopping
            up the divertor in
            :meth:`bluemira.radiation_transport.neutronics.DivertorWireAndExteriorCurve.make_divertor_pre_cell_array`
        """
        return np.concatenate([
            self.blanket.exterior_vertices(),
            self.divertor.exterior_vertices()[::-1],
        ])

    def get_interior_vertices(self) -> npt.NDArray:
        """
        Get the 3D coordinates of every point at the interior boundary of the tokamak's
        poloidal cross-section

        Returns
        -------
        coordinates
            array of shape ((N+1)+sum(number of interior points of the divertor), 3),
            where N = number of blanket pre-cells, M = number of divertor pre-cells.
            Runs clockwise, beginning at the inboard blanket-divertor joining point.
        """
        return np.concatenate([
            self.blanket.interior_vertices(),
            self.divertor.interior_vertices()[::-1],
        ])


class OpenMCReactor(OpenMCEnvironment):
    """
    A reactor in OpenMC's environment, containing the major components in
    self.csg_components.
    """

    def __init__(self, csg_components: CSGComponents):
        self.csg_components = csg_components

    def create_universe_box(
        self, r_max: float, z_minmax: tuple[float, float], *, control_id: bool = False
    ) -> openmc.region.Intersection:
        """
        Contain all components of interest in the csg model with finite-length cylinder.
        """
        z_min, z_max = sorted(z_minmax)
        bottom = self.find_suitable_z_plane(
            z_min,
            boundary_type="vacuum",
            surface_id=998 if control_id else None,
            name="Universe bottom",
        )
        top = self.find_suitable_z_plane(
            z_max,
            boundary_type="vacuum",
            surface_id=999 if control_id else None,
            name="Universe top",
        )
        universe_cylinder = openmc.ZCylinder(
            r=to_cm(r_max),
            surface_id=1000 if control_id else None,
            boundary_type="vacuum",
            name="Max radius of Universe",
        )
        self.csg_components.universe = -top & +bottom & -universe_cylinder
        return self.csg_components.universe

    def populate_radiation_shield_box(
        self,
        r_max: float,
        z_minmax: tuple[float, float],
        universe: openmc.region.Intersection,
        materials: MaterialsLibrary,
    ) -> openmc.Cell:
        """Define the radiation shield wall as a hollow of the universe box."""
        z_min, z_max = sorted(z_minmax)
        bottom_inner = self.find_suitable_z_plane(
            z_min,
            name="Radiation shield bottom inner wall",
        )

        top_inner = self.find_suitable_z_plane(
            z_max,
            name="Radiation shield top inner wall",
        )

        radial_cylinder_inboard = openmc.ZCylinder(
            r=to_cm(r_max),
            name="Radiation shield wall radial inboard",
        )

        rad_shield_inner = -top_inner & +bottom_inner & -radial_cylinder_inboard

        self.csg_components.radiation_shield = openmc.Cell(
            name="Radiation shield wall",
            fill=materials.match_material(CellType.RadiationShield),
            region=universe & ~rad_shield_inner,
        )
        return self.csg_components.radiation_shield

    def populate_coils(
        self,
        solenoid_radius: float,
        tf_coil_thick: float,
        z_minmax: tuple[float, float],
        materials: MaterialsLibrary,
    ) -> tuple[openmc.Cell, list[openmc.Cell]]:
        """
        Make tf coil and the central solenoid. The former wraps around the latter.

        Parameters
        ----------
        solenoid_radius:
            Central solenoid radius [m]
        tf_coil_thick:
            Thickness of the tf-coil, wrapped around the central solenoid [m]
        z_minmax:
            z-coordinate of the bottom and top z-planes shared by both cylinders
            (cs and tf coil)

        Raises
        ------
        GeometryError
            Thickness of TF coil and solenoid must be positive
        """
        z_min, z_max = sorted(z_minmax)
        if tf_coil_thick <= 0 or solenoid_radius <= 0:
            raise GeometryError(
                "Centrol column TF Coils and solenoid must have positive thicknesses!"
            )
        solenoid = openmc.ZCylinder(r=to_cm(solenoid_radius))
        central_tf_coil = openmc.ZCylinder(r=to_cm(tf_coil_thick + solenoid_radius))
        top = self.find_suitable_z_plane(
            z_max,
            [z_max - D_TOLERANCE, z_max + D_TOLERANCE],
            name="Top of central solenoid",
        )
        bottom = self.find_suitable_z_plane(
            z_min,
            [z_min - D_TOLERANCE, z_min + D_TOLERANCE],
            name="Bottom of central solenoid",
        )
        central_solenoid = openmc.Cell(
            name="Central solenoid",
            fill=materials.match_material(CellType.CSCoil),
            region=+bottom & -top & -solenoid,
        )
        tf_coils = [
            openmc.Cell(
                name="TF coil (sheath around central solenoid)",
                fill=materials.match_material(CellType.TFCoil),
                region=+bottom & -top & +solenoid & -central_tf_coil,
            )
        ]
        central_solenoid.volume = (top.z0 - bottom.z0) * np.pi * solenoid.r**2
        tf_coils[0].volume = (
            (top.z0 - bottom.z0) * np.pi * (central_tf_coil.r**2 - solenoid.r**2)
        )
        return central_solenoid, tf_coils

    def get_plasma_void(self, *, control_id: bool = False) -> openmc.Region:
        """
        Get the plasma chamber (as an openmc.Region object).

        Raises
        ------
        GeometryError
            Geometry must be convex
        """
        blanket, divertor = self.csg_components.blanket, self.csg_components.divertor
        blanket_interior_pts = blanket.interior_vertices()
        dividing_surface = self.make_dividing_surface(blanket)
        if not is_convex(blanket_interior_pts):
            raise GeometryError(
                f"{blanket} interior (blanket_outline's inner_curve) needs to be convex!"
            )
        plasma = self.region_from_surface_series(
            [*blanket.interior_surfaces(), dividing_surface],
            blanket_interior_pts,
            control_id=control_id,
        )

        divertor_exterior_vertices = divertor.exterior_vertices()
        if not is_convex(divertor_exterior_vertices):
            raise GeometryError(f"{divertor} exterior needs to be convex!")
        exhaust_including_divertor = self.region_from_surface_series(
            [*divertor.exterior_surfaces(), dividing_surface],
            divertor_exterior_vertices,
            control_id=control_id,
        )

        return flat_union([
            plasma,
            exhaust_including_divertor,
        ]) & ~divertor.exclusion_zone(control_id=control_id)

    def populate_void_cells(
        self, *, control_id: bool = False
    ) -> tuple[openmc.Cell, openmc.Cell]:
        """Make the plasma chamber and the outside ext_void. This should be called AFTER
        the blanket and divertor cells are created.
        """
        blanket_silhouette, divertor_silhouette = (
            self.get_blanket_and_divertor_outer_regions(
                self.csg_components.blanket,
                self.csg_components.divertor,
                control_id=control_id,
            )
        )
        void_region = (
            self.csg_components.universe & ~blanket_silhouette & ~divertor_silhouette
        )
        if self.csg_components.tf_coils:
            void_region &= ~self.csg_components.tf_coils[0].region
        if self.csg_components.cs_coil:
            void_region &= ~self.csg_components.cs_coil.region
        if self.csg_components.rad_shield:
            void_region &= ~self.csg_components.rad_shield.region

        self.csg_components.plasma = openmc.Cell(
            region=self.get_plasma_void(control_id=control_id),
            fill=None,
            name="Plasma void",
        )
        self.csg_components.ext_void = openmc.Cell(
            region=flat_intersection(void_region),
            fill=None,
            name="Exterior void",
        )
        return self.csg_components.plasma, self.csg_components.ext_void

    def make_dividing_surface(
        self, component
    ) -> openmc.Surface | openmc.model.ZConeOneSided:
        """Surface that marks the end of the divertor/blanket's exterior."""
        exterior_pts = component.exterior_vertices()[:, ::2]
        return self.surface_from_2points(exterior_pts[0], exterior_pts[-1])

    def get_blanket_and_divertor_outer_regions(
        self, *, control_id: bool = False
    ) -> openmc.Region:
        """
        Get the entire tokamak's poloidal cross-section (everything inside
        self.geom.boundary) as an openmc.Region.
        """
        blanket = self.universe.blanket
        divertor = self.universe.divertor
        dividing_surface = self.make_dividing_surface(blanket)
        blanket_outer = self.region_from_surface_series(
            [*blanket.exterior_surfaces(), dividing_surface],
            blanket.exterior_vertices(),
            control_id=control_id,
        )
        divertor_outer = self.region_from_surface_series(
            # TODO @OceanNuclear: this is wrong if the exterior surface of the divertor is not convex.  # noqa: TD003, W505, E501
            [*divertor.exterior_surfaces(), dividing_surface],
            divertor.exterior_vertices(),
            control_id=control_id,
        )
        return blanket_outer, divertor_outer

    @classmethod
    def from_pre_cell_reactor(
        cls,
        pre_cell_reactor: NeutronicsReactor,
        materials: MaterialsLibrary,
        *,
        control_id: bool = False,
    ) -> CSGComponents:
        """Make pre-cell arrays for the blanket and the divertor.

        Parameters
        ----------
        materials:
            library containing information about the materials
        tokamak_dimensions:
            A parameter
            :class:`bluemira.radiation_transport.neutronics.params.TokamakDimensions`,
            Specifying the dimensions of various layers in the blanket, divertor, and
            central solenoid.
        control_id: bool
            Whether to set the blanket Cells and surface IDs by force or not.
            With this set to True, it will be easier to understand where each cell came
            from. However, it will lead to warnings and errors if a cell/surface is
            generated to use a cell/surface ID that has already been used respectively.
            Keep this as False if you're running openmc simulations multiple times in one
            session.
        """
        # determine universe_box

        z_max, z_min, r_max, r_min = pre_cell_reactor.half_bounding_box

        z_min_adj = z_min - D_TOLERANCE
        z_max_adj = z_max + D_TOLERANCE
        r_max_adj = r_max + D_TOLERANCE

        rad_shield_wall_tk = pre_cell_reactor.tokamak_dimensions.rad_shield.wall

        # make the universe box, incorporates the radiation shield wall
        self = cls(CSGComponents(*[None for _ in CSGComponents.__dataclass_fields__]))

        self.csg_components.universe = self.create_universe_box(
            r_max_adj + rad_shield_wall_tk,
            (z_min_adj - rad_shield_wall_tk, z_max_adj + rad_shield_wall_tk),
            control_id=control_id,
        )

        self.csg_components.blanket = BlanketCellArray.from_pre_cell_array(
            pre_cell_reactor.blanket,
            materials,
            pre_cell_reactor.tokamak_dimensions,
            csg=self,
            control_id=control_id,
        )

        # change the cell and surface id register before making the divertor.
        # (ids will only count up from here.)
        if control_id:
            round_up_next_openmc_ids()

        self.csg_components.divertor = DivertorCellArray.from_pre_cell_array(
            pre_cell_reactor.divertor,
            materials,
            pre_cell_reactor.tokamak_dimensions.divertor,
            csg=self,
            override_start_end_surfaces=(
                self.csg_components.blanket[0].ccw_surface,
                self.csg_components.blanket[-1].cw_surface,
            ),
            # ID cannot be controlled at this point.
        )

        # make the plasma cell and the exterior void.
        if control_id:
            round_up_next_openmc_ids()

        self.populate_coils(
            r_min - pre_cell_reactor.tokamak_dimensions.cs_coil.thickness,
            pre_cell_reactor.tokamak_dimensions.cs_coil.thickness,
            (z_min_adj, z_max_adj),
            materials,
        )
        # make the radiation shield wall
        # which is a hollow of the universe box
        self.populate_radiation_shield_box(
            r_max_adj,
            (z_min_adj, z_max_adj),
            self.csg_components.universe,
            self.csg_components.materials,
        )
        self.populate_void_cells(control_id=control_id)

        self.csg_components.calculate_void_volumes()

        return self
