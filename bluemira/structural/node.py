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
Finite element Node object
"""
from typing import Dict, Tuple

import numpy as np

from bluemira.structural.constants import D_TOLERANCE
from bluemira.structural.error import StructuralError
from bluemira.structural.loads import node_load


class Node:
    """
    A 3-D node point

    Parameters
    ----------
    x:
        The node global x coordinate
    y:
        The node global y coordinate
    z:
        The node global z coordinate
    id_number:
        The node number in the finite element model
    """

    __slots__ = (
        "x",
        "y",
        "z",
        "id_number",
        "displacements",
        "supports",
        "symmetry",
        "loads",
        "reactions",
        "connections",
    )

    def __init__(self, x: float, y: float, z: float, id_number: int):
        self.x = x
        self.y = y
        self.z = z
        self.id_number = id_number

        self.loads = []
        self.supports = np.zeros(6, dtype=bool)  # Defaults to False
        self.symmetry = False
        self.displacements = np.zeros(6, dtype=float)
        self.reactions = np.zeros(6, dtype=float)
        self.connections = set()

    @property
    def xyz(self) -> np.ndarray:
        """
        Coordinate vector

        Returns
        -------
        The x-y-z coordinate vector of the Node (3)
        """
        return np.array([self.x, self.y, self.z])

    @xyz.setter
    def xyz(self, xyz: np.ndarray):
        """
        Sets the Node coordinates

        Parameters
        ----------
        xyz:
            The x-y-z coordinate vector of the Node (3)
        """
        self.x, self.y, self.z = xyz

    def distance_to_other(self, node) -> float:
        """
        Calculates the distance to another Node

        Parameters
        ----------
        node:
            The other node

        Returns
        -------
        The absolute distance between the two nodes
        """
        return np.sqrt(
            (node.x - self.x) ** 2 + (node.y - self.y) ** 2 + (node.z - self.z) ** 2
        )

    def add_load(self, load: Dict[str, float]):
        """
        Applies a load to the Node object.

        Parameters
        ----------
        load:
            The dictionary of nodal load values (always in global coordinates)
        """
        self.loads.append(load)

    def clear_loads(self):
        """
        Clear all loads and displacements applied to the Node
        """
        self.loads = []
        self.displacements = np.zeros(6, dtype=np.float32)

    def clear_supports(self):
        """
        Clears all supported DOFs applied to the Node
        """
        self.supports = np.zeros(6, dtype=bool)  # Defaults to False

    def add_support(self, supports: np.ndarray):
        """
        Define a support condition at the Node

        Parameters
        ----------
        supports:
            A boolean vector of the support DOFs, [dx, dy, dz, rx, ry, rz]:
                True == supported
                False == free
        """
        self.supports = supports

    def add_connection(self, elem_id: int):
        """
        Add a connection to the Node.

        Parameters
        ----------
        elem_id:
            The Element id_number which is connected to this Node
        """
        self.connections.add(elem_id)

    def remove_connection(self, elem_id: int):
        """
        Remove a connection to the Node.

        Parameters
        ----------
        elem_id:
            The Element id_number which is to be disconnected from this Node
        """
        self.connections.remove(elem_id)

    def p_vector(self) -> np.ndarray:
        """
        Global nodal force vector

        Returns
        -------
        nfv: np.array(6)
            The global nodal force vector
        """
        nfv = np.zeros(6)
        for load in self.loads:
            if load["type"] == "Node Load":
                nfv += node_load(load["Q"], load["sub_type"])
            else:
                raise StructuralError(
                    f'Cannot apply load type "{load["type"]}" to a Node.'
                )
        return nfv

    def __eq__(self, other) -> bool:
        """
        Checks the Node for equality to another Node.

        In practice this is used to check for Node coincidence.

        Parameters
        ----------
        other:
            The other Node to check for equality

        Returns
        -------
        Whether or not the nodes are coincident
        """
        if isinstance(self, other.__class__):
            return self.distance_to_other(other) <= D_TOLERANCE

        return False

    __hash__ = None


def get_midpoint(node1: Node, node2: Node) -> Tuple[float, float, float]:
    """
    Calculates the mid-point between two 3-D nodes

    Parameters
    ----------
    node1:
        First node
    node2:
        Second node

    Returns
    -------
    The coordinates of the mid-point
    """
    return (
        0.5 * (node1.x + node2.x),
        0.5 * (node1.y + node2.y),
        0.5 * (node1.z + node2.z),
    )
