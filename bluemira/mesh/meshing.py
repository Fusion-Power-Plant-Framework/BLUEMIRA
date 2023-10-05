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
Core functionality for the bluemira mesh module.
"""

from __future__ import annotations

import copy
import inspect
import pprint
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Union

# import mesher lib (gmsh)
import gmsh

from bluemira.base.look_and_feel import bluemira_print
from bluemira.mesh.error import MeshOptionsError

if TYPE_CHECKING:
    from bluemira.base.components import Component
# import bluemira modules


# Mesh options for the moment are limited to definition of mesh size for each point (
# quantity called lcar to be consistent with gmsh) and the definition of physical
# groups.
DEFAULT_MESH_OPTIONS = {
    "lcar": None,
    "physical_group": None,
}

MESH_DATA = {"points_tag": 0, "cntrpoints_tag": 0, "curve_tag": 1, "surface_tag": 2}
SUPPORTED_GEOS = ["BluemiraWire", "BluemiraFace", "BluemiraShell", "BluemiraCompound"]


def get_default_options() -> dict:
    """
    Returns the default display options.
    """
    return copy.deepcopy(DEFAULT_MESH_OPTIONS)


class MeshOptions:
    """
    The options that are available for meshing objects.
    """

    def __init__(self, **kwargs):
        self._options = get_default_options()
        self.modify(**kwargs)

    @property
    def lcar(self) -> float:
        """
        Mesh size of points.
        """
        return self._options["lcar"]

    @lcar.setter
    def lcar(self, val: float):
        self._options["lcar"] = val

    @property
    def physical_group(self):
        """
        Definition of physical groups.
        """
        return self._options["physical_group"]

    @physical_group.setter
    def physical_group(self, val: float):
        self._options["physical_group"] = val

    def as_dict(self) -> dict:
        """
        Returns the instance as a dictionary.
        """
        return copy.deepcopy(self._options)

    def modify(self, **kwargs):
        """
        Function to override meshing options.
        """
        if kwargs:
            for k in kwargs:
                if k in self._options:
                    self._options[k] = kwargs[k]

    def __repr__(self):
        """
        Representation string of the DisplayOptions.
        """
        return f"{self.__class__.__name__}({pprint.pformat(self._options)}" + "\n)"


class Meshable:
    """Mixin class to make a class meshable"""

    def __init__(self):
        super().__init__()
        self._mesh_options = MeshOptions()

    @property
    def mesh_options(self) -> MeshOptions:
        """
        The options that will be used to mesh the object.
        """
        return self._mesh_options

    @mesh_options.setter
    def mesh_options(self, value: Union[MeshOptions, Dict]):
        if isinstance(value, MeshOptions):
            self._mesh_options = value
        elif isinstance(value, Dict):
            if self.mesh_options is None:
                self.mesh_options = MeshOptions()
            self.mesh_options.modify(**value)
        else:
            raise MeshOptionsError("Mesh options must be set to a MeshOptions instance.")


class Mesh:
    """
    A class for supporting the creation of meshes and writing out those meshes to files.
    """

    def __init__(
        self,
        modelname: str = "Mesh",
        terminal: int = 0,
        meshfile: Optional[Union[str, List[str]]] = None,
        logfile: str = "gmsh.log",
    ):
        self.modelname = modelname
        self.terminal = terminal
        self.meshfile = (
            ["Mesh.geo_unrolled", "Mesh.msh"] if meshfile is None else meshfile
        )
        self.logfile = logfile

    @staticmethod
    def _check_meshfile(meshfile: Union[str, list]) -> List[str]:
        """
        Check the mesh file input.
        """
        # todo: should be implemented also a check on the file extension. Only a
        # limited type of file extensions is allowed by gmsh.
        if isinstance(meshfile, str):
            meshfile = [meshfile]
        elif isinstance(meshfile, list):
            if len(meshfile) < 1:
                raise ValueError("meshfile is an empty list")
        else:
            raise TypeError("meshfile must be a string or a list of strings")
        return meshfile

    @property
    def meshfile(self) -> List[str]:
        """
        The path(s) to the file(s) containing the meshes.
        """
        return self._meshfile

    @meshfile.setter
    def meshfile(self, meshfile: Union[str, List[str]]):
        self._meshfile = self._check_meshfile(meshfile)

    def __call__(self, obj: Union[Component, Meshable], dim: int = 2):
        """
        Generate the mesh and save it to file.
        """
        bluemira_print("Starting mesh process...")

        if "Component" in [c.__name__ for c in inspect.getmro(type(obj))]:
            from bluemira.base.tools import create_compound_from_component

            obj = create_compound_from_component(obj)

        if isinstance(obj, Meshable):
            # gmsh is initialized
            _FreeCADGmsh._initialize_mesh(self.terminal, self.modelname)
            # Mesh the object. A dictionary with the geometrical and internal
            # information that are used by gmsh is returned. In particular,
            # a gmsh key is added to any meshed entity.
            buffer = self.__mesh_obj(obj, dim=dim)
            # Check for possible intersection (only allowed at the boundary to adjust
            # the gmsh_dictionary
            Mesh.__iterate_gmsh_dict(buffer, Mesh._check_intersections)

            # Create the physical groups
            self._apply_physical_group(buffer)

            # apply the mesh size
            self._apply_mesh_size(buffer)

            # generate the mesh
            _FreeCADGmsh._generate_mesh()

            # save the mesh file
            for file in self.meshfile:
                _FreeCADGmsh._save_mesh(file)

            # close gmsh
            _FreeCADGmsh._finalize_mesh(self.logfile)
        else:
            raise TypeError("Only Meshable objects can be meshed")

        bluemira_print("Mesh process completed.")

        return buffer

    def __mesh_obj(self, obj, dim: int):
        """
        Function to mesh the object.
        """
        from bluemira.geometry.tools import serialize_shape

        if not hasattr(obj, "ismeshed") or not obj.ismeshed:
            # object is serialized into a dictionary
            if obj.__class__.__name__ in SUPPORTED_GEOS:
                buffer = serialize_shape(obj)
            else:
                raise ValueError(
                    f"Mesh procedure not implemented for {obj.__class__.__name__} type."
                )
            # Each object is recreated into gmsh. Here there is a trick: in order to
            # allow the correct mesh in case of intersection, the procedure
            # is made meshing the objects with increasing dimension.
            for d in range(1, dim + 1, 1):
                self.__convert_item_to_gmsh(buffer, d)
            obj.ismeshed = True
        else:
            bluemira_print("Object already meshed")
        return buffer

    def __convert_item_to_gmsh(self, buffer: dict, dim: int):
        for k in buffer:
            if k == "BluemiraWire":
                self.__convert_wire_to_gmsh(buffer, dim)
            if k == "BluemiraFace":
                self.__convert_face_to_gmsh(buffer, dim)
            if k == "BluemiraShell":
                self.__convert_shell_to_gmsh(buffer, dim)
            if k == "BluemiraCompound":
                self.__convert_compound_to_gmsh(buffer, dim)

    def _apply_physical_group(self, buffer: dict):
        """
        Function to apply physical groups
        """
        dict_dim = {
            "BluemiraWire": 1,
            "BluemiraFace": 2,
            "BluemiraShell": 2,
            "BluemiraCompound": 2,
        }
        other_dict = {0: "points_tag", 1: "curve_tag", 2: "surface_tag"}
        for k, v in buffer.items():
            if k in dict_dim:
                if "physical_group" in v:
                    _FreeCADGmsh.add_physical_group(
                        dict_dim[k],
                        self.get_gmsh_dict(buffer, "default")[other_dict[dict_dim[k]]],
                        v["physical_group"],
                    )
                for o in v["boundary"]:
                    self._apply_physical_group(o)

    def _apply_mesh_size(self, buffer: dict):
        """
        Function to apply mesh size.
        """
        # mesh size is applied not only to the vertexes of the defined geometry,
        # but also to the intersection points (new vertexes). For this reason,
        # it is important to do this operation after the completion of the mesh
        # procedure.
        points_lcar2 = self.__create_dict_for_mesh_size(buffer)
        if len(points_lcar2) > 0:
            for p in points_lcar2:
                _FreeCADGmsh._set_mesh_size([(0, p[0])], p[1])

    def __create_dict_for_mesh_size(self, buffer: dict):
        """
        Function to create the correct dictionary format for the
        application of the mesh size.
        """
        dict_dim = {
            "BluemiraWire": 1,
            "BluemiraFace": 2,
            "BluemiraShell": 2,
            "BluemiraCompound": 2,
        }
        other_dict = {0: "points_tag", 1: "curve_tag", 2: "surface_tag"}
        points_lcar = []
        for k, v in buffer.items():
            if k in dict_dim:
                if "lcar" in v and v["lcar"] is not None:
                    points_tags = self.get_gmsh_dict(buffer, "gmsh")[other_dict[0]]
                    if len(points_tags) > 0:
                        points_lcar += [(p[1], v["lcar"]) for p in points_tags]
                for o in v["boundary"]:
                    points_lcar += self.__create_dict_for_mesh_size(o)
        points_lcar = sorted(points_lcar, key=lambda element: (element[0], element[1]))
        points_lcar.reverse()
        points_lcar = dict(points_lcar)
        return list(points_lcar.items())

    @staticmethod
    def __apply_fragment(
        buffer: dict,
        dim: Iterable[int] = (2, 1, 0),
        all_ent=None,
        tools: Optional[list] = None,
        remove_object: bool = True,
        remove_tool: bool = True,
    ):
        """
        Apply the boolean fragment operation.
        """
        all_ent, oo, oov = _FreeCADGmsh._fragment(
            dim, all_ent, [] if tools is None else tools, remove_object, remove_tool
        )
        Mesh.__iterate_gmsh_dict(buffer, _FreeCADGmsh._map_mesh_dict, all_ent, oov)

    @staticmethod
    def _check_intersections(gmsh_dict: dict):
        """
        Check intersection and add the necessary vertexes to the gmsh dict.
        """
        if len(gmsh_dict["curve_tag"]) > 0:
            gmsh_curve_tag = [(1, tag) for tag in gmsh_dict["curve_tag"]]
            new_points = _FreeCADGmsh._get_boundary(gmsh_curve_tag)
            new_points = list({tag[1] for tag in new_points})
            gmsh_dict["points_tag"] = new_points

    @staticmethod
    def __iterate_gmsh_dict(buffer: dict, function: Callable, *args):
        """
        Supporting function to iterate over a gmsh dict.
        """
        if "BluemiraWire" in buffer:
            boundary = buffer["BluemiraWire"]["boundary"]
            if "gmsh" in buffer["BluemiraWire"]:
                function(buffer["BluemiraWire"]["gmsh"], *args)
            for item in boundary:
                for k in item:
                    if k == "BluemiraWire":
                        Mesh.__iterate_gmsh_dict(item, function, *args)

        if "BluemiraFace" in buffer:
            boundary = buffer["BluemiraFace"]["boundary"]
            if "gmsh" in buffer["BluemiraFace"]:
                function(buffer["BluemiraFace"]["gmsh"], *args)
            for item in boundary:
                Mesh.__iterate_gmsh_dict(item, function, *args)

        if "BluemiraShell" in buffer:
            boundary = buffer["BluemiraShell"]["boundary"]
            if "gmsh" in buffer["BluemiraShell"]:
                function(buffer["BluemiraShell"]["gmsh"], *args)
            for item in boundary:
                Mesh.__iterate_gmsh_dict(item, function, *args)

        if "BluemiraCompound" in buffer:
            boundary = buffer["BluemiraCompound"]["boundary"]
            if "gmsh" in buffer["BluemiraCompound"]:
                function(buffer["BluemiraCompound"]["gmsh"], *args)
            for item in boundary:
                Mesh.__iterate_gmsh_dict(item, function, *args)

    def __convert_wire_to_gmsh(self, buffer: dict, dim: int):
        """
        Converts a wire to gmsh. If dim is not equal to 1, wire is not meshed.
        """
        for type_, value in buffer.items():
            if type_ == "BluemiraWire":
                boundary = value["boundary"]
                if dim == 1:
                    value["gmsh"] = {
                        "points_tag": [],
                        "cntrpoints_tag": [],
                        "curve_tag": [],
                        "curveloop_tag": [],
                        "surface_tag": [],
                    }
                    for item in boundary:
                        for btype_, bvalue in item.items():
                            if btype_ == "BluemiraWire":
                                self.__convert_wire_to_gmsh(item, dim)
                            else:
                                for curve in bvalue:
                                    curve_gmsh_dict = _FreeCADGmsh.create_gmsh_curve(
                                        curve
                                    )
                                    value["gmsh"]["points_tag"] += curve_gmsh_dict[
                                        "points_tag"
                                    ]
                                    value["gmsh"]["cntrpoints_tag"] += curve_gmsh_dict[
                                        "cntrpoints_tag"
                                    ]
                                    value["gmsh"]["curve_tag"] += curve_gmsh_dict[
                                        "curve_tag"
                                    ]

                    # get the dictionary of the BluemiraWire defined in buffer
                    # as default and gmsh format
                    dict_gmsh = self.get_gmsh_dict(buffer, "gmsh")

                    # fragment points_tag and curves
                    all_ent = dict_gmsh["points_tag"] + dict_gmsh["curve_tag"]
                    self.__apply_fragment(buffer, all_ent, [], False, False)
            else:
                raise NotImplementedError(f"Serialization non implemented for {type_}")

    def __convert_face_to_gmsh(self, buffer: dict, dim: int):
        """
        Converts a face to gmsh.
        """
        for type_, value in buffer.items():
            if type_ == "BluemiraFace":
                boundary = value["boundary"]
                if dim == 1:
                    value["gmsh"] = {}
                    for item in boundary:
                        for btype_ in item:
                            if btype_ == "BluemiraWire":
                                self.__convert_wire_to_gmsh(item, dim)

                    # get the dictionary of the BluemiraWire defined in buffer
                    # as default and gmsh format
                    dict_gmsh = self.get_gmsh_dict(buffer, "gmsh")

                    # fragment points_tag and curves
                    all_ent = dict_gmsh["points_tag"] + dict_gmsh["curve_tag"]
                    self.__apply_fragment(buffer, all_ent=all_ent)
                elif dim == 2:  # noqa: PLR2004
                    value["gmsh"]["curveloop_tag"] = []
                    for item in boundary:
                        dict_curve = self.get_gmsh_dict(item)
                        value["gmsh"]["curveloop_tag"].append(
                            gmsh.model.occ.addCurveLoop(dict_curve["curve_tag"])
                        )
                    gmsh.model.occ.synchronize()
                    value["gmsh"]["surface_tag"] = [
                        gmsh.model.occ.addPlaneSurface(value["gmsh"]["curveloop_tag"])
                    ]
                    gmsh.model.occ.synchronize()

    def __convert_shell_to_gmsh(self, buffer: dict, dim: int):
        """
        Converts a shell to gmsh.
        """
        for type_, value in buffer.items():
            if type_ == "BluemiraShell":
                boundary = value["boundary"]
                if dim == 1:
                    value["gmsh"] = {}
                    for item in boundary:
                        self.__convert_face_to_gmsh(item, dim)
                        # get the dictionary of the BluemiraShell defined in buffer
                        # as default and gmsh format
                        dict_gmsh = self.get_gmsh_dict(buffer, "gmsh")

                        # fragment points_tag and curves
                        all_ent = dict_gmsh["points_tag"] + dict_gmsh["curve_tag"]
                        self.__apply_fragment(buffer, all_ent=all_ent)
                elif dim == 2:  # noqa: PLR2004
                    for item in boundary:
                        self.__convert_face_to_gmsh(item, dim)

    def __convert_compound_to_gmsh(self, buffer: dict, dim: int):
        """
        Converts a compound to gmsh.
        """
        for type_, value in buffer.items():
            if type_ == "BluemiraCompound":
                boundary = value["boundary"]
                if dim == 1:
                    value["gmsh"] = {}
                    for item in boundary:
                        self.__convert_item_to_gmsh(item, dim)
                        # get the dictionary of the Component defined in buffer
                        # as default and gmsh format
                        dict_gmsh = self.get_gmsh_dict(buffer, "gmsh")

                        # fragment points_tag and curves
                        all_ent = dict_gmsh["points_tag"] + dict_gmsh["curve_tag"]
                        self.__apply_fragment(buffer, all_ent=all_ent)
                elif dim == 2:  # noqa: PLR2004
                    for item in boundary:
                        self.__convert_item_to_gmsh(item, dim)

    def get_gmsh_dict(self, buffer: dict, file_format: str = "default"):
        """
        Returns the gmsh dict in a default (only tags) or gmsh (tuple(dim,
        tag)) format.
        """
        gmsh_dict = {}
        output = None

        for d in MESH_DATA:
            gmsh_dict[d] = []

        def _extract_mesh_from_buffer(buffer, obj_name):
            if obj_name not in buffer:
                raise ValueError(f"No {obj_name} to mesh.")

            boundary = buffer[obj_name]["boundary"]
            if "gmsh" in buffer[obj_name]:
                for d in MESH_DATA:
                    if d in buffer[obj_name]["gmsh"]:
                        gmsh_dict[d] += buffer[obj_name]["gmsh"][d]

            for item in boundary:
                if obj_name == "BluemiraWire":
                    for k in item:
                        if k == obj_name:
                            temp_dict = self.get_gmsh_dict(item)
                            for d in MESH_DATA:
                                gmsh_dict[d] += temp_dict[d]
                else:
                    temp_dict = self.get_gmsh_dict(item)
                    for d in MESH_DATA:
                        gmsh_dict[d] += temp_dict[d]

        for geo_name in SUPPORTED_GEOS:
            if geo_name in buffer:
                _extract_mesh_from_buffer(buffer, geo_name)

        for d in MESH_DATA:
            gmsh_dict[d] = list(dict.fromkeys(gmsh_dict[d]))

        if file_format == "default":
            output = gmsh_dict
        elif file_format == "gmsh":
            output = {}
            for d in MESH_DATA:
                output[d] = [(MESH_DATA[d], tag) for tag in gmsh_dict[d]]

        return output


class _FreeCADGmsh:
    @staticmethod
    def _initialize_mesh(terminal: int = 1, modelname: str = "Mesh"):
        # GMSH file generation #######################
        # Before using any functions in the Python API,
        # Gmsh must be initialized:
        gmsh.initialize()

        # By default Gmsh will not print out any messages:
        # in order to output messages
        # on the terminal, just set the "General.Terminal" option to 1:
        gmsh.option.setNumber("General.Terminal", terminal)

        gmsh.logger.start()

        # gmsh.option.setNumber("Mesh.MshFileVersion", 2.0)

        # Next we add a new model named "t1" (if gmsh.model.add() is
        # not called a new
        # unnamed model will be created on the fly, if necessary):
        gmsh.model.add(modelname)

    @staticmethod
    def _save_mesh(meshfile: str = "Mesh.geo_unrolled"):
        # ... and save it to disk
        gmsh.write(meshfile)

    @staticmethod
    def _finalize_mesh(logfile: str = "gmsh.log"):
        with open(logfile, "w") as file_handler:
            file_handler.write("\n".join(str(item) for item in gmsh.logger.get()))

        gmsh.logger.stop()

        # This should be called when you are done using the Gmsh Python API:
        gmsh.finalize()

    @staticmethod
    def _generate_mesh(mesh_dim: int = 3):
        # Before it can be meshed, the internal CAD representation must
        # be synchronized with the Gmsh model, which will create the
        # relevant Gmsh data structures. This is achieved by the
        # gmsh.model.occ.synchronize() API call for the built-in
        # geometry kernel. Synchronizations can be called at any time,
        # but they involve a non trivial amount of processing;
        # so while you could synchronize the internal CAD data after
        # every CAD command, it is usually better to minimize
        # the number of synchronization points.
        gmsh.model.occ.synchronize()

        # We can then generate a mesh...
        gmsh.model.mesh.generate(mesh_dim)

    @staticmethod
    def create_gmsh_curve(buffer: dict):
        """
        Function to create gmsh curve from a dictionary (buffer).
        """
        gmsh_dict = {}

        points_tag = []
        cntrpoints_tag = []
        curve_tag = []
        for type_ in buffer:
            if type_ == "LineSegment":
                points_tag.extend(
                    _add_points(buffer[type_]["StartPoint"], buffer[type_]["EndPoint"])
                )
                curve_tag.append(gmsh.model.occ.addLine(points_tag[0], points_tag[1]))
            elif type_ == "BezierCurve":
                cntrpoints_tag.extend(_add_points(*buffer[type_]["Poles"]))
                curve_tag.append(gmsh.model.occ.addBezier(cntrpoints_tag))
                points_tag.extend((cntrpoints_tag[0], cntrpoints_tag[-1]))
            elif type_ == "BSplineCurve":
                cntrpoints_tag.extend(_add_points(*buffer[type_]["Poles"]))
                curve_tag.append(gmsh.model.occ.addBSpline(cntrpoints_tag))
                points_tag.extend((cntrpoints_tag[0], cntrpoints_tag[-1]))
            elif type_ == "ArcOfCircle":
                start_tag, end_tag, centre_tag = _add_points(
                    buffer[type_]["StartPoint"],
                    buffer[type_]["EndPoint"],
                    buffer[type_]["Center"],
                )
                points_tag.extend((start_tag, end_tag))
                curve_tag.append(
                    gmsh.model.occ.addCircleArc(start_tag, centre_tag, end_tag)
                )
                cntrpoints_tag.append(centre_tag)
            elif type_ == "ArcOfEllipse":
                start_tag, end_tag, focus_tag, centre_tag = _add_points(
                    buffer[type_]["StartPoint"],
                    buffer[type_]["EndPoint"],
                    buffer[type_]["Focus1"],
                    buffer[type_]["Center"],
                )
                points_tag.extend((start_tag, end_tag))
                curve_tag.append(
                    gmsh.model.occ.addEllipseArc(
                        start_tag, centre_tag, focus_tag, end_tag
                    )
                )
                cntrpoints_tag.extend((centre_tag, focus_tag))
            else:
                raise NotImplementedError(
                    f"Gmsh curve creation non implemented for {type_}"
                )

        gmsh_dict["points_tag"] = points_tag
        gmsh_dict["cntrpoints_tag"] = cntrpoints_tag
        gmsh_dict["curve_tag"] = curve_tag
        gmsh.model.occ.synchronize()
        return gmsh_dict

    @staticmethod
    def _fragment(
        dim: Iterable[int] = (2, 1, 0),
        all_ent: Optional[List[int]] = None,
        tools: Optional[list] = None,
        remove_object: bool = True,
        remove_tool: bool = True,
    ):
        if not hasattr(dim, "__len__"):
            dim = [dim]
        if all_ent is None:
            all_ent = []
            for d in dim:
                all_ent += gmsh.model.getEntities(d)
        oo = []
        oov = []
        if len(all_ent) > 1:
            oo, oov = gmsh.model.occ.fragment(
                objectDimTags=all_ent,
                toolDimTags=[] if tools is None else tools,
                removeObject=remove_object,
                removeTool=remove_tool,
            )
            gmsh.model.occ.synchronize()

        return all_ent, oo, oov

    @staticmethod
    def _map_mesh_dict(mesh_dict: dict, all_ent, oov: Optional[list] = None):
        if oov is None:
            oov = []
        dim_dict = {
            "points_tag": 0,
            "cntrpoints_tag": 0,
            "curve_tag": 1,
            "surface_tag": 2,
        }
        new_gmsh_dict = {}

        for key in dim_dict:
            new_gmsh_dict[key] = []

        for type_, values in mesh_dict.items():
            if type_ != "curveloop_tag":
                for v in values:
                    dim = dim_dict[type_]
                    if (dim, v) in all_ent:
                        if len(oov) > 0:
                            new_gmsh_dict[type_].extend(
                                [o[1] for o in oov[all_ent.index((dim, v))]]
                            )
                    else:
                        new_gmsh_dict[type_].append(v)

        for key in dim_dict:
            mesh_dict[key] = list(dict.fromkeys(new_gmsh_dict[key]))

        return new_gmsh_dict

    @staticmethod
    def set_mesh_size(dim_tags, size):
        gmsh.model.occ.mesh.setSize(dim_tags, size)
        gmsh.model.occ.synchronize()

    @staticmethod
    def add_physical_group(dim, tags, name: Optional[str] = None):
        tag = gmsh.model.addPhysicalGroup(dim, tags)
        if name is not None:
            gmsh.model.setPhysicalName(dim, tag, name)

    @staticmethod
    def _set_mesh_size(dim_tags, size):
        gmsh.model.mesh.setSize(dim_tags, size)

    @staticmethod
    def _get_boundary(dimtags, combined=False, recursive=False):
        return gmsh.model.getBoundary(dimtags, combined, recursive)


def _add_points(*point: Iterable) -> List:
    """
    Add gmsh model points
    """
    return [gmsh.model.occ.addPoint(p[0], p[1], p[2]) for p in point]
