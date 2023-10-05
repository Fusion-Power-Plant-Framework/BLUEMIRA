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
Finite element geometry
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes

    from bluemira.geometry.coordinates import Coordinates
    from bluemira.structural.crosssection import CrossSection
    from bluemira.structural.material import StructuralMaterial

from copy import deepcopy

import numpy as np
from scipy.sparse import lil_matrix

from bluemira.geometry.bound_box import BoundingBox
from bluemira.structural.constants import D_TOLERANCE
from bluemira.structural.element import Element
from bluemira.structural.error import StructuralError
from bluemira.structural.node import Node
from bluemira.structural.plotting import (
    DeformedGeometryPlotter,
    GeometryPlotter,
    StressDeformedGeometryPlotter,
)


class Geometry:
    """
    Abstract object for the collection of nodes and elements in the finite
    element model
    """

    def __init__(self):
        self.nodes = []
        self.node_xyz = []
        self.elements = []

    @property
    def n_nodes(self) -> int:
        """
        The number of Nodes in the Geometry. Used to index Nodes.

        Returns
        -------
        The number of Nodes in the Geometry.
        """
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        """
        The number of Elements in the Geometry. Used to index Elements.

        Returns
        -------
        The number of Elements in the Geometry.
        """
        return len(self.elements)

    def add_node(self, x: float, y: float, z: float) -> int:
        """
        Add a Node to the Geometry object. Will check if an identical Node is
        already present.

        Parameters
        ----------
        x:
            The node global x coordinate
        y:
            The node global y coordinate
        z:
            The node global z coordinate

        Returns
        -------
        The ID number of the node that was added
        """
        node = Node(x, y, z, self.n_nodes)

        # Check that the node isn't already in the geometry
        for other in self.nodes:  # This could be slow, look to hash and set for speed
            if node == other:
                # Do not add new node, instead returning existing node id
                return other.id_number

        self.nodes.append(node)
        self.node_xyz.append([x, y, z])
        return node.id_number

    def find_node(self, x: float, y: float, z: float) -> int:
        """
        Return the node ID if the node coordinates are in the geometry.

        Parameters
        ----------
        x:
            The x coordinate of the node to find
        y:
            The y coordinate of the node to find
        z:
            The z coordinate of the node to find

        Returns
        -------
        The node ID
        """
        a = np.array(self.node_xyz)
        b = np.array([x, y, z])
        d_array = np.sqrt(np.sum((a - b) ** 2, axis=1))

        arg = np.argmin(d_array)

        if d_array[arg] > D_TOLERANCE:
            closest = self.nodes[arg].id_number
            proximity = d_array[arg]
            raise StructuralError(
                f"The node: [{x:.2f}, {y:.2f}, {z:.2f}] was not "
                "found in the model.\n"
                f"Closest node: {closest} at {proximity:.2f} m away"
            )

        return self.nodes[arg].id_number

    def move_node(self, node_id: int, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
        """
        Move a Node in the Geometry. If the Node is moved to the position
        of another Node, its connections are transferred, and the Node is
        removed.

        Parameters
        ----------
        node_id:
            The id_number of the Node to move
        dx:
            The x distance to move the Node
        dy:
            The y distance to move the Node
        dz:
            The z distance to move the Node
        """
        if np.sqrt(dx**2 + dy**2 + dz**2) <= D_TOLERANCE:
            return

        moved_node = deepcopy(self.nodes[node_id])
        moved_node.x += dx
        moved_node.y += dy
        moved_node.z += dz
        # Check that the new Node is not equal to any other nodes
        for other in self.nodes:
            if other == moved_node:
                # Check if there is an Element between the two Nodes
                for elem_id in moved_node.connections:
                    if elem_id in other.connections:
                        # If there is an Element between the moved Node and its
                        # duplicate, remove it
                        self.remove_element(elem_id)
                        # Make a new local copy
                        moved_node = deepcopy(self.nodes[node_id])

                # Transfer Node connections
                # Start again, because of potential renumbering
                for elem_id in moved_node.connections:
                    other.add_connection(elem_id)
                    element = self.elements[elem_id]
                    if element.node_1.id_number == node_id:
                        element.node_1 = other
                    else:
                        element.node_2 = other
                    element.clear_cache()

                # Remove Node connections (don't remove any Elements)
                self.nodes[node_id].connections = set()
                # Remove Node
                self.remove_node(node_id)
                break
        else:
            # We can safely move the Node
            self.nodes[node_id] = moved_node
            # Clear Element caches (Node has moved)
            for elem_id in moved_node.connections:
                element = self.elements[elem_id]
                if node_id == element.node_1.id_number:
                    element.node_1 = moved_node
                else:
                    element.node_2 = moved_node
                element.clear_cache()

    def remove_node(self, node_id: int):
        """
        Remove a Node from the Geometry.

        Parameters
        ----------
        node_id:
            The id_number of the Node to remove
        """
        # Drop the node information from the Geometry
        dead_node = self.nodes.pop(node_id)
        self.node_xyz.pop(node_id)
        # Re-number the remaining Nodes
        for node in self.nodes[node_id:]:
            node.id_number -= 1

        # Remove any Elements connected to the dead node
        # Cycle backwards to avoid re-numbering
        for elem_id in sorted(deepcopy(dead_node.connections))[::-1]:
            self.remove_element(elem_id)

    def add_element(
        self,
        node_id1: int,
        node_id2: int,
        cross_section: CrossSection,
        material: Optional[StructuralMaterial] = None,
    ) -> int:
        """
        Adds an Element to the Geometry object

        Parameters
        ----------
        node_id1:
            The ID number of the first node
        node_id2:
            The ID number of the second node
        cross_section:
            The CrossSection property object of the element
        material:
            The Material property object of the element

        Returns
        -------
        The ID number of the element that was added
        """
        # Check if there is already an Element specified between the Nodes
        new_element_nodes = sorted([node_id1, node_id2])
        for elem in self.elements:
            e_nodes = sorted([elem.node_1.id_number, elem.node_2.id_number])

            if e_nodes == new_element_nodes:
                # An element already exists here, update properties
                elem_id = elem.id_number

                element = Element(
                    self.nodes[node_id1],
                    self.nodes[node_id2],
                    elem_id,
                    cross_section,
                    material,
                )

                self.elements[elem_id] = element
                return elem_id

        # There is no such Element; add a new one to the model
        element = Element(
            self.nodes[node_id1],
            self.nodes[node_id2],
            self.n_elements,
            cross_section,
            material,
        )
        self.elements.append(element)
        # Keep track of Element connectivity
        self.nodes[node_id1].add_connection(element.id_number)
        self.nodes[node_id2].add_connection(element.id_number)
        return element.id_number

    def remove_element(self, elem_id: int):
        """
        Remove an Element from the Geometry.

        Parameters
        ----------
        elem_id:
            The Element id_number to remove
        """
        # Drop the Element information from the Geometry
        self.elements.pop(elem_id)
        # Re-number the remaining Elements
        for element in self.elements[elem_id:]:
            element.id_number -= 1

        # Re-number node connections
        for node in self.nodes:
            connections = sorted(deepcopy(node.connections))
            new_connections = set()
            for connection in connections:
                if connection == elem_id:
                    # Drop connection to dead Element
                    pass
                elif connection > elem_id:
                    # Re-number connection
                    new_connections.add(connection - 1)
                else:
                    # Preserve connection
                    new_connections.add(connection)
            node.connections = new_connections

    def add_coordinates(
        self,
        coordinates: Coordinates,
        cross_section: CrossSection,
        material: Optional[StructuralMaterial] = None,
    ):
        """
        Adds a Coordinates object to the Geometry

        Parameters
        ----------
        coordinates:
            The coordinates to transform into connected Nodes and Elements
        cross_section:
            The cross section of all the Elements in the Coordinates
        material:
            The material of all the Elements in the Coordinates
        """
        n_start = self.add_node(*coordinates.points[0])  # Add first Node

        n1 = n_start
        for point in coordinates.points[1:]:
            n2 = self.add_node(*point)
            self.add_element(n1, n2, cross_section, material)
            n1 = n2

        if coordinates.closed:
            self.add_element(n2, n_start, cross_section, material)

    def k_matrix(self) -> np.ndarray:
        """
        Builds the global stiffness matrix K

        Returns
        -------
        The global stiffness matrix of the Geometry ((6*n_nodes, 6*n_nodes))
        """
        # Explore how scipy sparse matrices or numba fares on this
        k = np.zeros((6 * self.n_nodes, 6 * self.n_nodes))
        # Loop through elements, adding local stiffness matrices into global
        for element in self.elements:
            k_elem = element.k_matrix_glob
            i = 6 * element.node_1.id_number
            j = 6 * element.node_2.id_number
            k[i : i + 6, i : i + 6] += k_elem[:6, :6]
            k[i : i + 6, j : j + 6] += k_elem[:6, 6:]
            k[j : j + 6, i : i + 6] += k_elem[6:, :6]
            k[j : j + 6, j : j + 6] += k_elem[6:, 6:]
        return k

    def k_matrix_sparse(self) -> lil_matrix:
        """
        Builds the sparse global stiffness matrix K

        Returns
        -------
        The sparse global stiffness matrix of the Geometry ((6*n_nodes, 6*n_nodes))
        """
        k = lil_matrix((6 * self.n_nodes, 6 * self.n_nodes))
        # Loop through elements, adding local stiffness matrices into global
        for element in self.elements:
            k_elem = element.k_matrix_glob
            i = 6 * element.node_1.id_number
            j = 6 * element.node_2.id_number
            k[i : i + 6, i : i + 6] += k_elem[:6, :6]
            k[i : i + 6, j : j + 6] += k_elem[:6, 6:]
            k[j : j + 6, i : i + 6] += k_elem[6:, :6]
            k[j : j + 6, j : j + 6] += k_elem[6:, 6:]
        return k

    def bounds(self) -> Tuple[float, float, float, float, float, float]:
        """
        Calculates the bounds of the geometry
        """
        x = [node.x for node in self.nodes]
        y = [node.y for node in self.nodes]
        z = [node.z for node in self.nodes]
        return max(x), min(x), max(y), min(y), max(z), min(z)

    def bounding_box(self) -> BoundingBox:
        """
        Calculates a bounding box for the Geometry object

        Returns
        -------
        BoundingBox rectangular cuboid of the geometry
        """
        x = np.array([node.x for node in self.nodes])
        y = np.array([node.y for node in self.nodes])
        z = np.array([node.z for node in self.nodes])
        return BoundingBox.from_xyz(x, y, z).get_box_arrays()

    def interpolate(self, scale: float = 100.0):
        """
        Interpolates the geometry model between Nodes
        """
        for element in self.elements:
            element.interpolate(scale=scale)

    def rotate(self, t_matrix: np.ndarray):
        """
        Rotates a geometry by updating the positions of all nodes and their
        global displacements

        Parameters
        ----------
        t_matrix:
            The rotation matrix to use
        """
        for node in self.nodes:
            # Move all the nodes
            node.xyz = t_matrix @ node.xyz

            # Update their displacements
            node.displacements[:3] = t_matrix @ node.displacements[:3]
            node.displacements[3:] = t_matrix @ node.displacements[3:]

        # Reset all the cached lambda matrices in the elements so that they are
        # re-calculated to account for new node positions
        for element in self.elements:
            element._lambda_matrix = None

    def transfer_node(self, node: Node):
        """
        Transfer a Node into the Geometry, copying all its state.

        Parameters
        ----------
        node:
            The Node to transfer into the geometry
        """
        id_number = self.add_node(*node.xyz)
        added_node = self.nodes[id_number]
        # Transfer node boundary conditions
        added_node.symmetry = node.symmetry
        added_node.supports = node.supports
        # Transfer node loads
        added_node.loads = node.loads
        # Transfer node displacements
        added_node.displacements = node.displacements
        return id_number

    def merge(self, other: Geometry):
        """
        Combine geometry object with another

        Parameters
        ----------
        other:
            The geometry to combine with

        Notes
        -----
        Will copy across Node and Element loads, BCs, displacements, and
        stress information
        """

        def transfer_loads(old, new):
            if old.loads:
                loads = []
                for load in old.loads:
                    load["element_id"] = new.id_number
                    loads.append(load)
                new.loads = loads

        # Walk through elements
        for element in other.elements:
            # Add nodes (will auto-check for existing nodes and return IDs)
            id_1 = self.transfer_node(element.node_1)
            id_2 = self.transfer_node(element.node_2)
            # Add element
            self.add_element(id_1, id_2, element._cross_section, element._material)
            added_element = self.elements[-1]
            # Transfer element loads
            transfer_loads(element, added_element)
            # Transfer element results
            added_element.shapes = element.shapes
            added_element.stresses = element.stresses
            added_element.max_stress = element.max_stress
            added_element.safety_factor = element.safety_factor

    def plot(self, ax=None, **kwargs):
        """
        Plot the Geometry.

        Parameters
        ----------
        ax: Union[Axes, None]
            The matplotlib Axes upon which to plot
        """
        return GeometryPlotter(self, ax=ax, **kwargs)


class DeformedGeometry(Geometry):
    """
    Abstract object for the collection of nodes and elements in the finite
    element model, with the ability to be deformed.
    """

    def __init__(self, geometry: Geometry, scale: float):
        geometry = deepcopy(geometry)
        self.nodes = geometry.nodes
        self.node_xyz = geometry.node_xyz
        self.elements = geometry.elements
        self._scale = scale

        self.deform()
        self.interpolate(scale)

    def deform(self):
        """
        Deform the Geometry by displacing the nodes by their deflections.
        """
        for node in self.nodes:
            node.x += node.displacements[0] * self._scale
            node.y += node.displacements[1] * self._scale
            node.z += node.displacements[2] * self._scale
            node.displacements[0] = 0
            node.displacements[1] = 0
            node.displacements[2] = 0

    def plot(
        self, ax: Optional[Axes] = None, stress: Optional[np.ndarray] = None, **kwargs
    ):
        """
        Plot the DeformedGeometry.

        Parameters
        ----------
        ax:
            The matplotlib Axes upon which to plot
        stress:
            The stress values to use (if any) when plotting
        """
        if stress is None:
            return DeformedGeometryPlotter(self, ax=ax, **kwargs)
        return StressDeformedGeometryPlotter(self, ax=ax, stress=stress, **kwargs)
