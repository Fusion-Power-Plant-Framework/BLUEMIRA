# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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
import numpy as np
from matplotlib.colors import DivergingNorm, Normalize

from bluemira.display import plot_3d
from bluemira.display.plotter import PlotOptions
from bluemira.geometry.plane import BluemiraPlane
from bluemira.structural.constants import (
    DEFLECT_COLOR,
    FLOAT_TYPE,
    LOAD_INT_VECTORS,
    LOAD_STR_VECTORS,
    STRESS_COLOR,
)
from bluemira.utilities.plot_tools import Plot3D

DEFAULT_PLOT_OPTIONS = {
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


def annotate_node(ax, node, text_size, color):
    """
    Annotate a node.
    """
    name = f"N{node.id_number}"
    ax.text(
        node.x,
        node.y,
        node.z,
        name,
        size=text_size,
        color=color,
    )


def annotate_element(ax, element, text_size, color):
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


def arrow_scale(vector, max_length, max_force):
    """
    Scales an arrow such that, regardless of direction, it has a reasonable
    size

    Parameters
    ----------
    vector: np.array(3)
        3-D vector of the arrow
    max_length: float
        The maximum length of the arrow
    max_force: float
        The maximum force value in the model (absolute)

    Returns
    -------
    vector: np.array(3)
        The scaled force arrow
    """
    v_norm = np.linalg.norm(vector)
    if v_norm == 0:
        return vector  # who cares? No numpy warning

    scale = (max_length * np.abs(vector)) / max_force

    return scale * vector / v_norm


def _plot_force(ax, node, vector, color="r"):
    """
    Plots a single force arrow in 3-D to indicate a linear load

    Parameters
    ----------
    ax: matplotlib Axes3D object
        The ax on which to plot
    node: Node object
        The node or location at which the force occurs
    vector: np.array(3)
        The force direction vector
    color: str
        The color to plot the force as
    """
    ax.quiver(
        node.x - vector[0], node.y - vector[1], node.z - vector[2], *vector, color=color
    )


def _plot_moment(ax, node, vector, color="r", support=False):
    """
    Plots a double "moment" arrow in 3-D to indicate a moment load. Offset the
    moment arrows off from the nodes a little, to enable overlaps with forces.

    Parameters
    ----------
    ax: matplotlib Axes3D object
        The ax on which to plot
    node: Node object
        The node or location at which the force occurs
    vector: np.array(3)
        The force direction vector
    color: str
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

    def __init__(self, geometry, ax=None, **kwargs):
        self.geometry = geometry
        if ax is None:
            self.ax = Plot3D()
        else:
            self.ax = ax

        self.options = {**DEFAULT_PLOT_OPTIONS, **kwargs}

        # Cached size and plot hints
        self._unit_length = None
        self._force_size = None
        self._size = None

        self.color_normer = None

    @property
    def unit_length(self):
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
    def force_size(self):
        """
        Calculates a characteristic force vector length for plotting purposes

        Returns
        -------
        f_length: float
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
                for load in node.loads:
                    loads.append(load["Q"])

            self._force_size = np.max(np.abs(loads))

        return self._force_size

    @property
    def size(self):
        """
        Calculates the size of the model bounding box
        """
        if self._size is None:
            xmax, xmin, ymax, ymin, zmax, zmin = self.geometry.bounds()

            self._size = max([xmax - xmin, ymax - ymin, zmax - zmin])

        return self._size

    @property
    def text_size(self):
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
        kwargs = self.options["node_options"].copy()
        default_color = kwargs.pop(
            "color", DEFAULT_PLOT_OPTIONS["node_options"]["color"]
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
                    if support and i < 3:
                        # Linear support (single black arrow)
                        _plot_force(self.ax, node, vector, color="k")
                    elif support and i >= 3:
                        # Moment support (double red arrow, offset to enable overlap)
                        _plot_moment(self.ax, node, vector, support=True, color="g")

    def plot_elements(self):
        """
        Plots all of the Elements in the Geometry.
        """
        kwargs = self.options["node_options"].copy()
        default_color = kwargs.pop(
            "color", DEFAULT_PLOT_OPTIONS["element_options"]["color"]
        )

        for element in self.geometry.elements:
            x = [element.node_1.x, element.node_2.x]
            y = [element.node_1.y, element.node_2.y]
            z = [element.node_1.z, element.node_2.z]

            if self.options["show_stress"]:
                color = STRESS_COLOR(self.color_normer(element.max_stress))
            elif self.options["show_deflection"]:
                color = DEFLECT_COLOR(self.color_normer(element.max_deflection))
            else:
                color = default_color

            self.ax.plot(x, y, z, color=color, **kwargs)

            if self.options["annotate_elements"]:
                annotate_element(self.ax, element, self.text_size, color)

            if self.options["interpolate"]:
                ls = kwargs.pop("linestyle")
                self.ax.plot(*element.shapes, linestyle="--", **kwargs)
                kwargs["linestyle"] = ls

    def plot_cross_sections(self):
        """
        Plots the cross-sections for each Element in the Geometry, rotated to
        the mid-point of the Element.
        """
        xss = []
        for element in self.geometry.elements:
            plane = BluemiraPlane(base=element.mid_point, axis=element.space_vector)
            plot_options = PlotOptions(
                show_wires=False,
                show_faces=True,
                face_options=self.options["cross_section_options"],
                # plane="yz",
            )
            xs = element._cross_section.geometry.deepcopy()
            xs.change_plane(plane)
            xss.append(xs)
            # xs.rotate()
            # xs.translate(*element.mid_point)
        plot_3d(xss, ax=self.ax, show=False)  # , options=[plot_options])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from bluemira.structural.crosssection import IBeam
    from bluemira.structural.geometry import Geometry
    from bluemira.structural.material import SS316

    xs = IBeam(0.5, 0.25, 0.1, 0.1)
    mat = SS316()

    geometry = Geometry()
    geometry.add_node(0, 0, 0)
    geometry.add_node(1, 1, 1)
    geometry.add_node(2, 1, 1)
    geometry.add_element(0, 1, xs, mat)
    geometry.add_element(1, 2, xs, mat)

    plotter = BasePlotter(geometry)
    # plotter.plot_nodes()
    # plotter.plot_supports()
    # plotter.plot_elements()
    plotter.plot_cross_sections()
    plt.show()
