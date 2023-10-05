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
Load objects
"""
import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.structural.constants import LOAD_MAPPING, LOAD_TYPES
from bluemira.structural.error import StructuralError


def _check_load_type(load_type, sub_type="all"):
    if load_type not in LOAD_TYPES:
        raise StructuralError(f"Unrecognised load type: {load_type}.")

    if sub_type == "force" and load_type not in LOAD_TYPES[:3]:
        raise StructuralError(
            f"Cannot set a force load with a moment load type: {load_type}"
        )

    if sub_type == "moment" and load_type not in LOAD_TYPES[3:]:
        raise StructuralError(
            f"Cannot set a moment load with a force load type: {load_type}"
        )


def node_load(load: float, load_type: str) -> np.ndarray:
    """
    Calculates the reaction force vector due to a point load Q, applied at a
    Node.

    Parameters
    ----------
    load:
        The value of the point load (which may be a force or a moment)
    load_type:
        The type of load to apply (from 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])
            'Fi': force in the 'i' direction
            'Mi': moment about the 'i' direction

    Returns
    -------
    The fixed-ends reaction force vector (6)
    """
    reactions = np.zeros(6)
    _check_load_type(load_type)
    reactions[LOAD_MAPPING[load_type]] = load
    return reactions


def point_load(load: float, x: float, length: float, load_type: str) -> np.ndarray:
    """
    Calculates the reaction force vector due to a point load Q, applied at x
    along the element of length L, going from node_1 to node_2.

    Parameters
    ----------
    load:
        The value of the point load (which may be a force or a moment)
    x:
        The parameterised distance along the element from node_1 to node_2,
        from 0 to 1.
    length:
        The element length
    load_type:
        The type of load to apply (from['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])
            'Fi': force in the 'i' direction
            'Mi': moment about the 'i' direction

    Returns
    -------
    The fixed-ends reaction force vector (12)
    """
    _check_load_type(load_type)

    reactions = np.zeros(12)
    x *= length
    a = length - x
    if load_type == "Fx":
        reactions[0] = -load * a / length
        reactions[6] = -load * x / length
    elif load_type == "Fy":
        reactions[1] = load * a**2 * (length + 2 * x) / length**3
        reactions[5] = load * x * a**2 / length**2
        reactions[7] = load * x**2 * (length + 2 * a) / length**3
        reactions[11] = -load * x**2 * a / length**2
    elif load_type == "Fz":
        reactions[2] = load * a**2 * (length + 2 * x) / length**3
        reactions[4] = -load * x * a**2 / length**2
        reactions[8] = load * x**2 * (length + 2 * a) / length**3
        reactions[10] = load * x**2 * a / length**2
    elif load_type == "Mx":
        reactions[3] = -load * a / length
        reactions[9] = -load * x / length
    elif load_type == "My":
        reactions[2] = 6 * load * x * a / length**3
        reactions[4] = load * a * (2 * x - a) / length**2
        reactions[8] = -load * x * a / length**3
        reactions[10] = load * x * (2 * a - x) / length**2
    elif load_type == "Mz":
        reactions[1] = -load * x * a / length**3
        reactions[5] = load * a * (2 * x - a) / length**2
        reactions[7] = load * x * a / length**3
        reactions[11] = load * x * (2 * a - x) / length**2

    return reactions


def distributed_load(w: float, length: float, load_type: str) -> np.ndarray:
    """
    Calculates the reactor force vector due to a distributed load w, applied
    over the length of the element.

    Parameters
    ----------
    w:
        The value of the distributed load
    length:
        Length of the element along which the load is applied
    load_type:
        The type of load to apply (from ['Fx', 'Fy', 'Fz']):
            'Fi': force in the 'i' direction
    """
    _check_load_type(load_type, sub_type="force")

    reactions = np.zeros(12)
    if load_type == "Fx":
        reactions[0] = w * length / 2
        reactions[6] = -w * length / 2
    elif load_type == "Fy":
        reactions[1] = w * length / 2
        reactions[5] = w * length**2 / 12
        reactions[7] = w * length / 2
        reactions[11] = -w * length**2 / 12
    elif load_type == "Fz":
        reactions[2] = w * length / 2
        reactions[4] = -w * length**2 / 12
        reactions[8] = w * length / 2
        reactions[10] = w * length**2 / 12

    return reactions


class LoadCase(list):
    """
    A simple container object for a collection of loads
    """

    def add_node_load(self, node_id: int, load: float, load_type: str):
        """
        Adds a node load to the LoadCase

        Parameters
        ----------
        node_id:
            The id_number of the Node to apply the load at
        load:
            The value of the load
        load_type:
            The type and axis of the load from ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        """
        _check_load_type(load_type)
        self.append(
            {"type": "Node Load", "sub_type": load_type, "node_id": node_id, "Q": load}
        )

    def add_element_load(self, element_id: int, load: float, x: float, load_type: str):
        """
        Adds an element point load to the LoadCase

        Parameters
        ----------
        element_id:
            The id_number of the Element to apply the load at
        load:
            The value of the load
        x:
            The parameterised and normalised distance along the element x-axis
        load_type: str from ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
            The type and axis of the load
        """
        _check_load_type(load_type)
        x_clip = np.clip(x, 0, 1)
        if x_clip != x:
            bluemira_warn("x should be between 0 and 1.")
        self.append(
            {
                "type": "Element Load",
                "sub_type": load_type,
                "element_id": element_id,
                "Q": load,
                "x": x,
            }
        )

    def add_distributed_load(self, element_id: int, w: float, load_type: str):
        """
        Adds a distributed load to the LoadCase

        Parameters
        ----------
        element_id:
            The id_number of the Element to apply the load at
        w:
            The value of the distributed load
        load_type:
            The type and axis of the load (from ['Fx', 'Fy', 'Fz'])
        """
        _check_load_type(load_type)
        self.append(
            {
                "type": "Distributed Load",
                "sub_type": load_type,
                "element_id": element_id,
                "w": w,
            }
        )
