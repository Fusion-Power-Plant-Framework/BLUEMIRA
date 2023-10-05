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
Structural module plotting tools
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes

    from bluemira.structural.element import Element
    from bluemira.structural.geometry import DeformedGeometry, Geometry
    from bluemira.structural.node import Node

from copy import deepcopy

import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

from bluemira.display import plot_3d
from bluemira.display.plotter import PlotOptions
from bluemira.geometry.placement import BluemiraPlacement
from bluemira.structural.constants import (
    DEFLECT_COLOR,
    LOAD_INT_VECTORS,
    LOAD_STR_VECTORS,
    STRESS_COLOR,
)
from bluemira.utilities.plot_tools import Plot3D

DEFAULT_STRUCT_PLOT_OPTIONS = {
    "bound_scale": 1.1,
    "show_all_nodes": True,
    "show_stress": False,
    "show_deflection": False,
    "interpolate": False,
    "show_cross_sections": True,
    "annotate_nodes": True,
    "annotate_elements": True,
    "node_options": {"marker": "o", "ms": 12, "color": "k", "alpha": 1},
    "symmetry_node_color": "g",
    "support_node_color": "r",
    "element_options": {"linewidth": 3, "color": "k", "linestyle": "-", "alpha": 1},
    "show_as_grey": False,
    "cross_section_options": {"color": "b"},
}


def annotate_node(ax: Axes, node: Node, text_size: int, color: str):
    """
    Annotate a node.
    """
    name = f"N{node.id_number}"
    ax.text(
        node.x,
        node.y,
        node.z,
        name,
        fontsize=text_size,
        color=color,
    )


def annotate_element(ax: Axes, element: Element, text_size: int, color: str):
    """
    Annotate an element.
    """
    name = f"E{element.id_number}"
    ax.text(
        *element.mid_point,
        name,
        size=text_size,
        color=color,
    )


def arrow_scale(vector: np.ndarray, max_length: float, max_force: float) -> np.ndarray:
    """
    Scales an arrow such that, regardless of direction, it has a reasonable
    size

    Parameters
    ----------
    vector:
        3-D vector of the arrow (3)
    max_length
        The maximum length of the arrow
    max_force:
        The maximum force value in the model (absolute)

    Returns
    -------
    The scaled force arrow (3)
    """
    v_norm = np.linalg.norm(vector)
    if v_norm == 0:
        return vector  # who cares? No numpy warning

    scale = (max_length * np.abs(vector)) / max_force

    return scale * vector / v_norm


def _plot_force(ax: Axes, node: Node, vector: np.ndarray, color: str = "r"):
    """
    Plots a single force arrow in 3-D to indicate a linear load

    Parameters
    ----------
    ax:
        The ax on which to plot
    node:
        The node or location at which the force occurs
    vector:
        The force direction vector
    color:
        The color to plot the force as
    """
    ax.quiver(
        node.x - vector[0], node.y - vector[1], node.z - vector[2], *vector, color=color
    )


def _plot_moment(
    ax: Axes, node: Node, vector: np.ndarray, color: str = "r", support: bool = False
):
    """
    Plots a double "moment" arrow in 3-D to indicate a moment load. Offset the
    moment arrows off from the nodes a little, to enable overlaps with forces.

    Parameters
    ----------
    ax:
        The ax on which to plot
    node:
        The node or location at which the force occurs
    vector:
        The force direction vector
    color:
        The color to plot the force as
    """
    if support:
        # Offsets the moment arrows a little so we can see overlaps with forces
        vector *= 2
        f1 = 0.5
        f2 = 0.25
    else:
        f1 = 1
        f2 = 0.5
    ax.quiver(
        node.x - vector[0],
        node.y - vector[1],
        node.z - vector[2],
        *f1 * vector,
        color=color,
    )
    ax.quiver(
        node.x - vector[0],
        node.y - vector[1],
        node.z - vector[2],
        *f2 * vector,
        color=color,
        arrow_length_ratio=0.6,
    )


class BasePlotter:
    """
    Base utility plotting class for structural models
    """

    def __init__(self, geometry: Geometry, ax: Optional[Axes] = None, **kwargs):
        self.geometry = geometry
        if ax is None:
            self.ax = Plot3D()
        else:
            self.ax = ax

        self.options = {**DEFAULT_STRUCT_PLOT_OPTIONS, **kwargs}

        # Cached size and plot hints
        self._unit_length = None
        self._force_size = None
        self._size = None

        self.color_normer = None

    @property
    def unit_length(self) -> float:
        """
        Calculates a characteristic unit length for the model: the minimum
        element size
        """
        if self._unit_length is None:
            lengths = np.zeros(self.geometry.n_elements)
            for i, element in enumerate(self.geometry.elements):
                lengths[i] = element.length
            self._unit_length = np.min(lengths)

        return self._unit_length

    @property
    def force_size(self) -> float:
        """
        Calculates a characteristic force vector length for plotting purposes

        Returns
        -------
        The minimum and maximum forces
        """
        if self._force_size is None:
            loads = []
            for element in self.geometry.elements:
                for load in element.loads:
                    if load["type"] == "Element Load":
                        loads.append(load["Q"])
                    elif load["type"] == "Distributed Load":
                        loads.append(load["w"] / element.length)

            for node in self.geometry.nodes:
                loads.extend(load["Q"] for load in node.loads)

            self._force_size = np.max(np.abs(loads))

        return self._force_size

    @property
    def size(self) -> float:
        """
        Calculates the size of the model bounding box
        """
        if self._size is None:
            xmax, xmin, ymax, ymin, zmax, zmin = self.geometry.bounds()

            self._size = max([xmax - xmin, ymax - ymin, zmax - zmin])

        return self._size

    @property
    def text_size(self) -> int:
        """
        Get a reasonable guess of the font size to use in plotting.

        Returns
        -------
        size: float
            The font size to use in plotting
        """
        return max(10, self.size // 30)

    def plot_nodes(self):
        """
        Plots all the Nodes in the Geometry.
        """
        kwargs = deepcopy(self.options["node_options"])
        default_color = kwargs.pop(
            "color", DEFAULT_STRUCT_PLOT_OPTIONS["node_options"]["color"]
        )

        for node in self.geometry.nodes:
            if node.supports.any():
                color = self.options["support_node_color"]
            elif node.symmetry:
                color = self.options["symmetry_node_color"]
            else:
                color = default_color

            self.ax.plot([node.x], [node.y], [node.z], color=color, **kwargs)

            if self.options["annotate_nodes"]:
                annotate_node(self.ax, node, self.text_size, color)

    def plot_supports(self):
        """
        Plots all supports in the Geometry.
        """
        lengths = np.array([e.length for e in self.geometry.elements])
        length = lengths.min() / 5
        for node in self.geometry.nodes:
            if node.supports.any():
                for i, support in enumerate(node.supports):
                    vector = length * LOAD_INT_VECTORS[i]
                    if support and i < 3:  # noqa: PLR2004
                        # Linear support (single black arrow)
                        _plot_force(self.ax, node, vector, color="k")
                    elif support and i >= 3:  # noqa: PLR2004
                        # Moment support (double red arrow, offset to enable overlap)
                        _plot_moment(self.ax, node, vector, support=True, color="g")

    def plot_elements(self):
        """
        Plots all of the Elements in the Geometry.
        """
        kwargs = deepcopy(self.options["element_options"])
        default_color = kwargs.pop(
            "color", DEFAULT_STRUCT_PLOT_OPTIONS["element_options"]["color"]
        )

        for element in self.geometry.elements:
            x = [element.node_1.x, element.node_2.x]
            y = [element.node_1.y, element.node_2.y]
            z = [element.node_1.z, element.node_2.z]

            if self.options["show_stress"] and self.color_normer:
                color = STRESS_COLOR(self.color_normer(element.max_stress))
            elif self.options["show_deflection"] and self.color_normer:
                color = DEFLECT_COLOR(self.color_normer(element.max_displacement))
            else:
                color = default_color

            self.ax.plot(x, y, z, marker=None, color=color, **kwargs)

            if self.options["annotate_elements"]:
                annotate_element(self.ax, element, self.text_size, color="k")

            if self.options["interpolate"]:
                ls = kwargs.pop(
                    "linestyle",
                    DEFAULT_STRUCT_PLOT_OPTIONS["element_options"]["linestyle"],
                )
                self.ax.plot(
                    *element.shapes, marker=None, linestyle="--", color=color, **kwargs
                )
                kwargs["linestyle"] = ls

    def plot_cross_sections(self):
        """
        Plots the cross-sections for each Element in the Geometry, rotated to
        the mid-point of the Element.
        """
        xss = []
        options = []
        for element in self.geometry.elements:
            matrix = np.zeros((4, 4))
            matrix[:3, :3] = element.lambda_matrix[:3, :3].T
            matrix[:3, -1] = element.mid_point
            matrix[-1, :] = [0, 0, 0, 1]
            placement = BluemiraPlacement.from_matrix(matrix)
            plot_options = PlotOptions(
                show_wires=False,
                show_faces=True,
                face_options=self.options["cross_section_options"],
            )
            options.append(plot_options)
            xs = element._cross_section.geometry.deepcopy()
            xs.change_placement(placement)
            xss.append(xs)

        plot_3d(xss, ax=self.ax, show=False, options=options)

    def plot_loads(self):
        """
        Plots all of the loads applied to the geometry
        """
        for node in self.geometry.nodes:
            if node.loads:
                for load in node.loads:
                    self._plot_node_load(node, load)

        for element in self.geometry.elements:
            for load in element.loads:
                if load["type"] == "Element Load":
                    self._plot_element_load(element, load)
                elif load["type"] == "Distributed Load":
                    self._plot_distributed_load(element, load)

    def _plot_node_load(self, node, load):
        load_value = load["Q"] * LOAD_STR_VECTORS[load["sub_type"]]

        load_value = arrow_scale(load_value, 10 * self.unit_length, self.force_size)

        if "F" in load["sub_type"]:
            _plot_force(self.ax, node, load_value, color="r")

        elif "M" in load["sub_type"]:
            _plot_moment(self.ax, node, load_value, color="r")

    def _plot_element_load(self, element, load):
        load = load["Q"] * LOAD_STR_VECTORS[load["sub_type"]]

        load = arrow_scale(load, 10 * self.unit_length, self.force_size)

        dcm = element.lambda_matrix[0:3, 0:3]
        load = load @ dcm
        point = np.array(
            [element.node_1.x, element.node_1.y, element.node_1.z], dtype=float
        )
        point += (np.array([1.0, 0.0, 0.0]) * float(load["x"])) @ dcm
        self.ax.quiver(*point - load, *load, color="r")

    def _plot_distributed_load(self, element, load):
        length = element.length
        n = int(length * 10)
        dcm = element.lambda_matrix[0:3, 0:3]
        load = load["w"] * LOAD_STR_VECTORS[load["sub_type"]] / length

        load = arrow_scale(load, 10 * self.unit_length, self.force_size)

        load = load @ dcm
        load = load * np.ones((3, n)).T
        load = load.T
        point = np.array(
            [element.node_1.x, element.node_1.y, element.node_1.z], dtype=float
        )
        point = point * np.ones((3, n)).T
        point += (
            np.array([x * np.array([1.0, 0.0, 0.0]) for x in np.linspace(0, length, n)])
            @ dcm
        )
        point = point.T
        self.ax.quiver(*point - load, *load, color="r")

    def _set_aspect_equal(self):
        """
        Hack to make matplotlib 3D look good. Draw a white bounding box around
        the nodes
        """
        x_bb, y_bb, z_bb = self.geometry.bounding_box()

        x_bb *= self.options["bound_scale"]
        y_bb *= self.options["bound_scale"]
        z_bb *= self.options["bound_scale"]

        for x, y, z in zip(x_bb, y_bb, z_bb):
            self.ax.plot([x], [y], [z], color="w")


class GeometryPlotter(BasePlotter):
    """
    Utility class for the plotting of structural geometry models
    """

    def __init__(self, geometry: Geometry, ax: Optional[Axes] = None, **kwargs):
        super().__init__(geometry, ax, **kwargs)
        self.options = deepcopy(DEFAULT_STRUCT_PLOT_OPTIONS)
        self.options["show_stress"] = False
        self.options["show_deflection"] = False

        self.plot_nodes()
        self.plot_elements()
        self.plot_supports()
        self.plot_loads()
        if self.options["show_cross_sections"]:
            self.plot_cross_sections()
        self._set_aspect_equal()


class DeformedGeometryPlotter(BasePlotter):
    """
    Utility class for the plotting of structural deformed geometry models and
    overlaying with GeometryPlotters
    """

    def __init__(self, geometry: DeformedGeometry, ax: Optional[Axes] = None, **kwargs):
        super().__init__(geometry, ax, **kwargs)
        self.options = deepcopy(DEFAULT_STRUCT_PLOT_OPTIONS)
        self.options["node_options"]["color"] = "b"
        self.options["element_options"]["color"] = "b"
        self.options["show_stress"] = False
        self.options["show_deflection"] = True
        self.options["annotate_nodes"] = True
        self.options["interpolate"] = True
        self.options["show_all_nodes"] = False

        self.plot_nodes()
        self.plot_elements()
        self._set_aspect_equal()


class StressDeformedGeometryPlotter(BasePlotter):
    """
    Utility class for the plotting of structural deformed geometry models and
    overlaying with GeometryPlotters
    """

    def __init__(
        self,
        geometry: DeformedGeometry,
        ax: Optional[Axes] = None,
        stress: Optional[np.ndarray] = None,
        deflection: bool = False,
        **kwargs,
    ):
        super().__init__(geometry, ax, **kwargs)
        self.options = deepcopy(DEFAULT_STRUCT_PLOT_OPTIONS)
        self.options["node_options"]["color"] = "b"
        self.options["element_options"]["color"] = None
        self.options["show_stress"] = True
        self.options["annotate_nodes"] = False
        self.options["interpolate"] = True
        self.options["show_all_nodes"] = False
        self.options["show_as_grey"] = False

        self.color_normer = self.make_color_normer(stress, deflection)

        self.plot_nodes()
        self.plot_elements()
        self._set_aspect_equal()

    @staticmethod
    def make_color_normer(stress: np.ndarray, deflection: bool = False):
        """
        Make a ColorNorm object for the plot based on the stress values.
        """
        smin, smax = min(stress), max(stress)
        if smin == smax:

            class SameColour:
                def __call__(self, _):
                    return 0.5

            return SameColour()

        centre = 0

        if deflection:
            # deflections positive when plotting
            deflections = np.abs(stress)
            return Normalize(vmin=0, vmax=max(deflections))

        if not smin < 0 < smax:
            centre = (smin + smax) / 2

        return TwoSlopeNorm(centre, vmin=min(stress), vmax=max(stress))
