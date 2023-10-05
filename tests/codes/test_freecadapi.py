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

from pathlib import Path
from unittest.mock import patch

import freecad  # noqa: F401
import Part
import numpy as np
import pytest
from FreeCAD import Base, newDocument

import bluemira.codes._freecadapi as cadapi
from bluemira.base.constants import EPS
from bluemira.codes.error import FreeCADError
from bluemira.geometry.constants import D_TOLERANCE
from tests._helpers import skipif_import_error


class TestFreecadapi:
    @classmethod
    def setup_class(cls):
        cls.square_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
        cls.closed_square_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
        ]

    @staticmethod
    def offsetter(wire):
        return cadapi.offset_wire(
            wire,
            0.05,
            join="intersect",
            open_wire=False,
        )

    def test_multi_offset_wire(self):
        circ = cadapi.make_circle(10)
        wire1 = self.offsetter(circ)
        wire2 = self.offsetter(wire1)

        assert circ.Length < wire1.Length < wire2.Length

    def test_multi_offset_wire_without_arranged_edges(self):
        """
        FreeCAD Topological naming bug

        As of 08/2022 FreeCAD has a ordering/naming bug as detailed in #1347.
        The result is that some operations do not work as expected (such as offset_wire)
        Some operations when repeated also raise errors such as DisjointedFace errors.
        arrange_edges tries to reorder the internal edges of a wire to attempt to side
        step the issue.

        FreeCAD aims to fix this for v1 which is due for release in 2023.
        When that happens our work arounds can be removed (including this test)
        """

        circ = cadapi.make_circle(10)
        with patch(
            "bluemira.codes._freecadapi.arrange_edges",
            new=lambda a, b: b,  # noqa: ARG005
        ):
            wire1 = self.offsetter(circ)
            wire2 = self.offsetter(wire1)

        assert circ.Length < wire1.Length
        # these two should break in future, this mean the topo naming may be fixed
        assert circ.Length == wire2.Length
        assert wire1.Length > wire2.Length

    def test_fail_vector_to_numpy(self):
        with pytest.raises(TypeError):
            arr = cadapi.vector_to_numpy(self.square_points)

    def test_fail_points_to_numpy(self):
        with pytest.raises(TypeError):
            arr = cadapi.point_to_numpy(self.square_points)

    def test_single_vector_to_numpy(self):
        inp = np.array((1.0, 0.5, 2.0))
        vector = Base.Vector(inp)
        arr = cadapi.vector_to_numpy(vector)
        comparison = arr == inp
        assert comparison.all()

    def test_vector_to_numpy(self):
        vectors = [Base.Vector(v) for v in self.square_points]
        arr = cadapi.vector_to_numpy(vectors)
        comparison = arr == np.array(self.square_points)
        assert comparison.all()

    def test_point_to_numpy(self):
        vectors = [Part.Point(Base.Vector(v)) for v in self.square_points]
        arr = cadapi.point_to_numpy(vectors)
        comparison = arr == np.array(self.square_points)
        assert comparison.all()

    def test_vertex_to_numpy(self):
        vertexes = [Part.Vertex(Base.Vector(v)) for v in self.square_points]
        arr = cadapi.vertex_to_numpy(vertexes)
        comparison = arr == np.array(self.square_points)
        assert comparison.all()

    def test_make_polygon(self):
        # open wire
        open_wire: Part.Wire = cadapi.make_polygon(self.square_points)
        vertexes = open_wire.Vertexes
        assert len(vertexes) == 4
        assert len(open_wire.Edges) == 3
        arr = cadapi.vertex_to_numpy(vertexes)
        comparison = arr == np.array(self.square_points)
        assert comparison.all()
        assert not open_wire.isClosed()
        # closed wire
        closed_wire: Part.Wire = cadapi.make_polygon(self.closed_square_points)
        vertexes = closed_wire.Vertexes
        assert len(vertexes) == 4
        assert len(closed_wire.Edges) == 4
        arr = cadapi.vertex_to_numpy(vertexes)
        comparison = arr == np.array(self.square_points)
        assert comparison.all()
        assert closed_wire.isClosed()

    def test_make_bezier(self):
        bezier: Part.Wire = cadapi.make_bezier(self.square_points)
        curve = bezier.Edges[0].Curve
        assert type(curve) is Part.BezierCurve

    def test_interpolate_bspline(self):
        pntslist = self.square_points
        bspline: Part.Wire = cadapi.interpolate_bspline(pntslist)
        curve = bspline.Edges[0].Curve
        assert type(curve) is Part.BSplineCurve
        # assert that the bspline pass through the points
        # get the points parameter
        params = [curve.parameter(Base.Vector(p)) for p in pntslist]
        # get the points on the curve at the calculated parameters
        test_points = cadapi.vector_to_list([curve.value(par) for par in params])
        # assert that the points on the curve are equal (within a tolerance) to the
        # points used to generate the bspline
        assert np.allclose(
            np.array(test_points) - np.array(pntslist), 0, atol=D_TOLERANCE
        )

    def test_length(self):
        open_wire: Part.Wire = cadapi.make_polygon(self.square_points)
        assert (
            cadapi.length(open_wire)
            == open_wire.Length
            == pytest.approx(3.0, rel=0, abs=EPS)
        )
        closed_wire: Part.Wire = cadapi.make_polygon(self.closed_square_points)
        assert (
            cadapi.length(closed_wire)
            == closed_wire.Length
            == pytest.approx(4.0, rel=0, abs=EPS)
        )

    def test_area(self):
        wire: Part.Wire = cadapi.make_polygon(self.closed_square_points)
        assert cadapi.area(wire) == wire.Area == pytest.approx(0.0, rel=0, abs=EPS)
        face: Part.Face = Part.Face(wire)
        assert cadapi.area(face) == face.Area == pytest.approx(1.0, rel=0, abs=EPS)

    def test_center_of_mass(self):
        wire: Part.Wire = cadapi.make_polygon(self.closed_square_points)
        face: Part.Face = Part.Face(wire)
        com = cadapi.center_of_mass(wire)
        comparison = com == np.array((0.5, 0.5, 0.0))
        assert isinstance(com, np.ndarray)
        assert comparison.all()

    def test_scale_shape(self):
        factor = 2.0
        wire: Part.Wire = cadapi.make_polygon(self.closed_square_points)
        scaled_wire = cadapi.scale_shape(wire.copy(), factor)
        face: Part.Face = Part.Face(scaled_wire)
        assert cadapi.area(face) == 1.0 * factor**2
        assert (
            cadapi.length(face)
            == cadapi.length(scaled_wire)
            == pytest.approx(4.0 * factor, rel=0, abs=EPS)
        )
        face_from_wire = Part.Face(wire)
        scaled_face = cadapi.scale_shape(face_from_wire.copy(), factor)
        assert cadapi.length(scaled_face) == cadapi.length(face)
        assert cadapi.area(scaled_face) == cadapi.area(face)

    def test_discretize(self):
        wire: Part.Wire = cadapi.make_polygon(self.closed_square_points)
        ndiscr = 10
        points = cadapi.discretize(wire, ndiscr)
        assert len(points) == ndiscr
        length_w = wire.Length
        dl = length_w / float(ndiscr - 1)
        points = cadapi.discretize(wire, dl=dl)
        assert len(points) == ndiscr

    def test_discretize_by_edges(self):
        wire: Part.Wire = cadapi.make_polygon(self.closed_square_points)
        ndiscr = 10
        points = cadapi.discretize_by_edges(wire, ndiscr)

        dl = 0.4
        points1 = cadapi.discretize_by_edges(wire, dl=dl)

        dl = 0.4
        points2 = cadapi.discretize_by_edges(wire, ndiscr=100, dl=dl)
        assert np.allclose(points1 - points2, 0, atol=D_TOLERANCE)

    def test_discretize_vs_discretize_by_edges(self):
        wire1 = cadapi.make_polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        wire2 = cadapi.make_polygon([[0, 0, 0], [0, 1, 0], [1, 1, 0]])
        wire2.reverse()
        wire = Part.Wire([wire1, wire2])

        # ndiscr is chosen in such a way that both discretize and discretize_by_edges
        # give the same points (so that a direct comparison is possible).
        points1 = cadapi.discretize(wire, ndiscr=5)
        points2 = cadapi.discretize_by_edges(wire, ndiscr=4)

        # assert that points1 and points2 are the same
        assert np.allclose(points1 - points2, 0, atol=D_TOLERANCE)

    def test_start_point_given_polygon(self):
        wire = cadapi.make_polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0]])

        start_point = cadapi.start_point(wire)

        assert isinstance(start_point, np.ndarray)
        np.testing.assert_equal(start_point, np.array([0, 0, 0]))

    def test_end_point_given_polygon(self):
        wire = cadapi.make_polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0]])

        end_point = cadapi.end_point(wire)

        assert isinstance(end_point, np.ndarray)
        np.testing.assert_equal(end_point, np.array([1, 1, 0]))

    def test_catcherror(self):
        @cadapi.catch_caderr(ValueError)
        def func():
            raise FreeCADError("Error")

        with pytest.raises(ValueError):  # noqa: PT011
            func()

    def test_save_cad(self, tmp_path):
        shape = cadapi.extrude_shape(cadapi.make_circle(), (0, 0, 1))
        filename = f"{tmp_path}/tst.stp"

        cadapi.save_cad([shape], filename)
        assert Path(filename).exists
        with open(filename) as fh:
            stp_content = fh.read()
        assert "myshape" not in stp_content
        assert "Bluemira" in stp_content
        assert (
            "AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF. {1 0 10303 442 1 1 4"
            in stp_content
        )

        cadapi.save_cad(
            [shape],
            filename,
            labels=["myshape"],
            author="myfile",
            stp_file_scheme="AP214IS",
        )
        assert Path(filename).exists()
        with open(filename) as fh:
            stp_content = fh.read()
        assert "myshape" in stp_content  # shape label in file
        assert "Bluemira" in stp_content  # bluemira still in file if author changed
        assert "myfile" in stp_content  # author change
        assert (
            "AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }" in stp_content
        )  # scheme change


class TestCADFiletype:
    @classmethod
    def setup_class(cls):
        doc = newDocument()
        cls.shape = doc.addObject("Part::FeaturePython")
        cls.shape.Shape = cadapi.extrude_shape(cadapi.make_circle(), (0, 0, 1))
        doc.recompute()

    def setup_method(self):
        import FreeCADGui

        if not hasattr(FreeCADGui, "subgraphFromObject"):
            FreeCADGui.setupWithoutGUI()

    @pytest.mark.parametrize(("name", "ftype"), cadapi.CADFileType.__members__.items())
    def test_init(self, name, ftype):
        assert cadapi.CADFileType[name] == ftype
        assert cadapi.CADFileType(ftype.value) == ftype

    # Commented out CADFileTypes dont work with basic shapes tested or needed more
    # FreeCAD imported, should be reviewed in future
    @pytest.mark.parametrize(
        "name",
        [
            "ASCII_STEREO_MESH",
            "ADDITIVE_MANUFACTURING",
            "AUTOCAD_DXF",
            "BINMESH",
            "BREP",
            "BREP_2",
            "CSG",
            "FREECAD",
            "GLTRANSMISSION",
            "GLTRANSMISSION_2",
            "IGES",
            "IGES_2",
            "INVENTOR_V2_1",
            "JSON",
            "OBJ",
            "OBJ_WAVE",
            "OFF",
            "OPENSCAD",
            "PLY_STANFORD",
            "SIMPLE_MODEL",
            "STEP",
            "STEP_2",
            "STEP_ZIP",  # Case sensitive extension
            "STL",
            "THREED_MANUFACTURING",
            pytest.param("IFC_BIM", marks=[skipif_import_error("ifcopenshell")]),
            pytest.param(
                "IFC_BIM_JSON",  # github.com/buildingSMART/ifcJSON
                marks=[skipif_import_error("ifcopenshell", "ifcjson")],
            ),
            pytest.param("DAE", marks=[skipif_import_error("collada")]),
            pytest.param("AUTOCAD", marks=[pytest.mark.xfail]),  # LibreDWG required
            # # Part.Feature has no compatible object type, find compatible object type
            # "ASC", "BDF", "DAT", "FENICS_FEM", "FENICS_FEM_XML", "INP", "MED",
            # "MESHJSON", "MESHPY", "MESHYAML", "PCD", "PLY", "TETGEN_FEM", "UNV",
            # "VTK", "VTU", "YAML", "Z88_FEM_MESH", "Z88_FEM_MESH_2",
            # # More FreeCAD than we import, fails differently on each import
            # "WEBGL",
            # # No file output
            # "SVG, "SVG_FLAT",
            # # Requires TechDrawGui import which requires a GUI
            # "PDF", "VRML", "VRML_2", "VRML_ZIP", "VRML_ZIP_2",
            # "WEBGL_X3D", "X3D", "X3DZ"
        ],
    )
    def test_exporter_function_exists_and_creates_a_file(self, name, tmp_path):
        filetype = cadapi.CADFileType[name]
        filename = f"{tmp_path/'tst'}.{filetype.value}"
        if name != "FREECAD":  # custom function in this case
            assert filetype.exporter.__name__ == "export"

        cadapi.CADFileType[name].exporter([self.shape], filename)
        assert Path(filename).exists()
