# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Plasma Face Designer
"""

from dataclasses import dataclass

import numpy as np

from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import boolean_cut, make_polygon
from bluemira.geometry.wire import BluemiraWire


@dataclass
class PlasmaFaceDesignerParams(ParameterFrame):
    """Parameters for running the :class:`.PlasmaFaceDesigner`."""

    div_type: Parameter[str]
    c_rm: Parameter[float]
    lower_port_angle: Parameter[float]


class PlasmaFaceDesigner(
    Designer[tuple[BluemiraFace, BluemiraFace, tuple[float, float]]]
):
    """
    Designs the Plasma facing keep out zones

    Parameters
    ----------
    params:
        Plasma face designer parameters
    ivc_boundary:
        IVC boundary keep out zone
    wall_boundary:
        wall boundary keep out zone (cut at divertor)
    divertor_silhouette:
        divertor keep out zone
    """

    param_cls: type[PlasmaFaceDesignerParams] = PlasmaFaceDesignerParams
    params: PlasmaFaceDesignerParams

    def __init__(
        self,
        params: ParameterFrame | dict,
        ivc_boundary: BluemiraWire,
        wall_boundary: BluemiraWire,
        divertor_silhouette: tuple[BluemiraWire, ...],
    ):
        super().__init__(params)
        if self.params.div_type.value == "DN":
            raise NotImplementedError("Double Null divertor not implemented")
        self.ivc_boundary = ivc_boundary
        self.wall_boundary = wall_boundary
        self.divertor_silhouette = divertor_silhouette

    def run(self) -> tuple[BluemiraFace, BluemiraFace, tuple[float, float]]:
        """
        Run method for PlasmaFaceDesigner

        Returns
        -------
        blanket_face:
            The blanket face
        divertor_face:
            The divertor face
        div_wall_join_pt:
            The divertor blanket join point
        """
        # For double null this and self.divertor_silhouette need a structure
        # to accommodate two divertors
        plasma_facing_wire = BluemiraWire([
            self.wall_boundary,
            *self.divertor_silhouette,
        ])

        in_vessel_face = BluemiraFace([self.ivc_boundary, plasma_facing_wire])

        # Cut a clearance between the blankets and divertor - getting two
        # new faces
        vessel_bbox = in_vessel_face.bounding_box
        # The minimum z-value of the wall boundary. The boundary should
        # be open at the lower end and the start and end points of the
        # wire should be at the same z. But take the minimum z value of
        # the start and end points.
        # Note we do not use bounding_box here due to a bug: 34228d3
        baffel_z = np.min([
            self.wall_boundary.start_point().z,
            self.wall_boundary.end_point().z,
        ])
        # This is the x the outer baffle connects to the wall
        baffel_ib_start_x = np.min([
            self.wall_boundary.start_point().x,
            self.wall_boundary.end_point().x,
        ])
        baffel_ob_end_x = np.max([
            self.wall_boundary.start_point().x,
            self.wall_boundary.end_point().x,
        ])

        rm_clearance_face = _make_clearance_face(
            vessel_bbox.x_min,
            (baffel_ib_start_x + baffel_ob_end_x) / 2,
            baffel_z,
            self.params.c_rm.value,
        )
        rm_outer_clearance_face = _angled_wall_cutter(
            baffel_ob_end_x,
            baffel_z,
            self.params.c_rm.value,
            self.params.lower_port_angle.value,
        )

        blanket_face, divertor_face = _cut_vessel_shape(
            in_vessel_face, [rm_clearance_face, rm_outer_clearance_face]
        )

        div_wall_join_pt = (baffel_ob_end_x, baffel_z)

        return blanket_face, divertor_face, div_wall_join_pt


def _angled_wall_cutter(
    join_x: float, join_z: float, thickness: float, duct_angle_degree: float
) -> BluemiraFace:
    x_coords = np.zeros(4)
    x_coords[:2] = -2  # arb x len, must be larger than wall thickness at the angle
    x_coords[2:] = 2

    y_coords = np.zeros(4)

    z_coords = np.zeros(4)
    z_coords[[0, 3]] = thickness / 2
    z_coords[[1, 2]] = -thickness / 2

    cutter_face = BluemiraFace(make_polygon([x_coords, y_coords, z_coords], closed=True))
    cutter_face.translate((join_x, 0, join_z))
    cutter_face.rotate(
        degree=duct_angle_degree, base=(join_x, 0, join_z), direction=(0, -1, 0)
    )

    return cutter_face


def _make_clearance_face(
    x_start: float, x_end: float, z: float, thickness: float
) -> BluemiraFace:
    """
    Makes a rectangular face in xz with the given thickness in z.

    The face is intended to be used to cut a remote maintenance
    clearance between blankets and divertor.

    Returns
    -------
    :
        Clearance face
    """
    x_coords = np.zeros(4)
    x_coords[:2] = x_start
    x_coords[2:] = x_end

    y_coords = np.zeros(4)

    z_coords = np.zeros(4)
    z_coords[[0, 3]] = z + thickness / 2
    z_coords[[1, 2]] = z - thickness / 2

    return BluemiraFace(make_polygon([x_coords, y_coords, z_coords], closed=True))


def _cut_vessel_shape(
    in_vessel_face: BluemiraFace, cutter_faces: list[BluemiraFace]
) -> tuple[BluemiraFace, BluemiraFace]:
    """
    Cut a remote maintenance clearance into the given vessel shape.

    Returns
    -------
    blanket_face:
        Cut blanket face
    divertor_face:
        Cut divertor face
    """
    pieces = boolean_cut(in_vessel_face, cutter_faces)
    blanket_face = pieces[np.argmax([p.center_of_mass[2] for p in pieces])]
    divertor_face = pieces[np.argmin([p.center_of_mass[2] for p in pieces])]
    return blanket_face, divertor_face
