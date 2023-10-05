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
Finite element model
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from bluemira.geometry.coordinates import Coordinates
    from bluemira.structural.crosssection import CrossSection
    from bluemira.structural.material import StructuralMaterial

from copy import deepcopy

import numpy as np
from scipy.sparse.linalg import spsolve

from bluemira.base.constants import EPS
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.structural.constants import R_LARGE_DISP
from bluemira.structural.error import StructuralError
from bluemira.structural.geometry import Geometry
from bluemira.structural.loads import LoadCase
from bluemira.structural.result import Result
from bluemira.structural.symmetry import CyclicSymmetry


def check_matrix_condition(matrix: np.ndarray, digits: int):
    """
    Checks the condition number of a matrix and warns if it is unsuitable for
    working with.

    Parameters
    ----------
    matrix:
        The matrix to check the condition number of
    digits:
        The desired level of digit-precision (higher is less demanding)

    Raises
    ------
    StructuralError
        If the stiffness matrix is singular or ill-conditioned
    """
    condition_number = np.linalg.cond(matrix)
    digit_loss = np.log10(condition_number)

    err_txt = ""
    if condition_number > 1 / EPS:
        err_txt = """
            "Structural::FiniteElementModel:\n Singular stiffness matrix will "
            "cause LinAlgErrors.\n"
            f"matrix condition number: {condition_number}"
            """

    if digit_loss > digits:
        digit_loss = int(np.ceil(digit_loss))

        err_txt = """
            "Structural::FiniteElementModel:\n Ill-conditioned matrix"
            f"\n|\tAccuracy loss below the {digit_loss}-th digit."
        """

    if err_txt:
        err_txt += "\nProbably worth checking model boundary conditions."
        raise StructuralError(err_txt)


class FiniteElementModel:
    """
    3-D beam finite element model. The main interface object with other modules
    As such, all important functionality is brought to the surface here, hence
    the large number of class methods

    Attributes
    ----------
    geometry:
        The geometry in the FiniteElementModel
    load_case:
        The load case applied in the FiniteElementModel
    n_fixed_dofs:
        The number of fixed degrees of freedom
    fixed_dofs:
        The fixed degrees of freedom
    fixed_dof_ids:
        The id_numbers of the fixed degrees of freedom

    """

    N_INTERP = 7  # Number of interpolation points in an Element

    def __init__(self):
        self.geometry = Geometry()
        self.load_case = LoadCase()
        self.n_fixed_dofs = 0
        self.fixed_dofs = np.zeros(6, dtype=bool)  # Defaults to False
        self.fixed_dof_ids = []
        self.cycle_sym_ids = []
        self.cycle_sym = None

    # =========================================================================
    # Geometry definition methods
    # =========================================================================

    def set_geometry(self, geometry: Geometry):
        """
        Set a Geometry in the FiniteElementModel

        Parameters
        ----------
        geometry:
            The Geometry to add to the FiniteElementModel
        """
        self.geometry = geometry

    def add_node(self, x: float, y: float, z: float) -> int:
        """
        Adds a Node to the FiniteElementModel

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
        return self.geometry.add_node(x, y, z)

    def add_element(
        self,
        node_id1: int,
        node_id2: int,
        cross_section: CrossSection,
        material: Optional[StructuralMaterial] = None,
    ) -> int:
        """
        Adds an Element to the FiniteElementModel

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
        return self.geometry.add_element(node_id1, node_id2, cross_section, material)

    def add_coordinates(
        self,
        coords: Coordinates,
        cross_section: CrossSection,
        material: Optional[StructuralMaterial] = None,
    ):
        """
        Adds a Coordinates object to the FiniteElementModel

        Parameters
        ----------
        coordinates:
            The coordinates to transform into connected Nodes and Elements
        cross_section:
            The cross section of all the Elements in the Coordinates
        material:
            The material of all the Elements in the Coordinates
        """
        self.geometry.add_coordinates(coords, cross_section, material)

    def add_support(
        self,
        node_id: int,
        dx: bool = False,
        dy: bool = False,
        dz: bool = False,
        rx: bool = False,
        ry: bool = False,
        rz: bool = False,
    ):
        """
        Applies a support condition at a Node in the FiniteElementModel

        Parameters
        ----------
        node_id:
            The id_number of the Node where to apply the condition
        dx, dy, dz:
            Whether or not the linear DOFs at the Node are constrained
        rx, ry, rz:
            Whether or not the rotational DOFs at the Node
        """
        supports = np.array([dx, dy, dz, rx, ry, rz], dtype=bool)
        self.geometry.nodes[node_id].add_support(supports)

    def find_supports(self):
        """
        Find the support conditions in the FiniteElementModel.
        """
        # Clear and start from scratch
        self.n_fixed_dofs = 0
        self.fixed_dof_ids = []

        # Search each node for boundary conditions
        for node in self.geometry.nodes:
            n_supports = np.count_nonzero(node.supports)
            self.n_fixed_dofs += n_supports  # Count fixed DOFs

            if n_supports != 0:
                support_indices = np.nonzero(node.supports == True)[0]  # noqa: E712
                dofs = [6 * node.id_number + i for i in support_indices]
                # Keep tracked of fixed DOFs
                self.fixed_dof_ids.extend(dofs)
                # Keep track of which DOFs are fixed
                self.fixed_dofs[support_indices] = True

    def apply_cyclic_symmetry(
        self,
        left_node_ids: List[int],
        right_node_ids: List[int],
        p1: Optional[np.ndarray] = None,
        p2: Optional[np.ndarray] = None,
    ):
        """
        Applies a cyclic symmetry condition to the FiniteElementModel

        Parameters
        ----------
        left_node_ids:
            The id numbers of the nodes on the left boundary
        right_node_ids:
            The id numbers of the nodes on the right boundary
        p1:
            The first point of the symmetry rotation axis
        p1:
            The second point of the symmetry rotation axis
        """
        if p1 is None:
            p1 = [0, 0, 0]
        if p2 is None:
            p2 = [0, 0, 1]

        # Apply symmetry flag at node level for plotting
        for id_number in [left_node_ids, right_node_ids]:
            self.geometry.nodes[id_number].symmetry = True

        self.cycle_sym_ids.append([left_node_ids, right_node_ids, p1, p2])

    # =========================================================================
    # Load definition methods
    # =========================================================================

    def apply_load_case(self, load_case: LoadCase):
        """
        Apply a load case to the FiniteElementModel.

        Parameters
        ----------
        load_case:
            The load case to apply to the model.
        """
        self.load_case = load_case

    def add_node_load(self, node_id: int, load: float, load_type: str):
        """
        Adds a node load to the FiniteElementModel

        Parameters
        ----------
        node_id:
            The id_number of the Node to apply the load at
        load:
            The value of the load
        load_type:
            The type and axis of the load (from ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])
        """
        self.load_case.add_node_load(node_id, load, load_type)

    def add_element_load(self, element_id: int, load: float, x: float, load_type: str):
        """
        Adds an element point load to the FiniteElementModel

        Parameters
        ----------
        element_id:
            The id_number of the Element to apply the load at
        load:
            The value of the load
        x:
            The parameterised and normalised distance along the element x-axis
        load_type:
            The type and axis of the load (from ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])
        """
        self.load_case.add_element_load(element_id, load, x, load_type)

    def add_distributed_load(self, element_id: int, w: float, load_type: str):
        """
        Adds a distributed load to the FiniteElementModel

        Parameters
        ----------
        element_id:
            The id_number of the Element to apply the load at
        w:
            The value of the distributed load
        load_type:
            The type and axis of the load (from ['Fx', 'Fy', 'Fz'])
        """
        self.load_case.add_distributed_load(element_id, w, load_type)

    def add_gravity_loads(self):
        """
        Applies self-weight distributed loads to all members
        """
        for element in self.geometry.elements:
            w = np.array([0, 0, -element.weight * element.length])
            forces = element.lambda_matrix[0:3, 0:3] @ w

            for i, direction in enumerate(["Fx", "Fy", "Fz"]):
                if forces[i] != 0:
                    self.add_distributed_load(element.id_number, forces[i], direction)

    def clear_loads(self):
        """
        Clears any applied loads to the model, and removes any displacements
        """
        for node in self.geometry.nodes:
            node.clear_loads()
        for element in self.geometry.elements:
            element.clear_loads()
        self.load_case = LoadCase()

    def clear_load_case(self):
        """
        Clears the LoadCase applied to the model
        """
        self.load_case = LoadCase()

    # =========================================================================
    # Private solution methods
    # =========================================================================

    def _apply_load_case(self, load_case: LoadCase):
        """
        Applies a LoadCase to the FiniteElementModel. Maps individual loads to
        their respective Nodes and Elements.

        Parameters
        ----------
        load_case: LoadCase object
            The list of loads to apply to the model
        """
        for load in load_case:
            if load["type"] == "Node Load":
                node = self.geometry.nodes[load["node_id"]]
                node.add_load(load)

            elif load["type"] in ["Element Load", "Distributed Load"]:
                element = self.geometry.elements[load["element_id"]]
                element.add_load(load)
            else:
                raise StructuralError(f'Unknown load type "{load["type"]}"')

    def _get_nodal_forces(self) -> np.ndarray:
        """
        Calculates the total nodal forces (including forces applied at nodes
        and equivalent concentrated forces from element loads).

        Returns
        -------
        The global nodal force vector (6*n_nodes)
        """
        # NOTE: looping over loads in a LoadCase would no doubt be faster
        # but it is more difficult to keep track of things in OO like this
        p_vector = np.zeros(6 * self.geometry.n_nodes)

        for node in self.geometry.nodes:
            idn = 6 * node.id_number
            p_vector[idn : idn + 6] += node.p_vector()

        for element in self.geometry.elements:
            enf = element.equivalent_node_forces()
            i = 6 * element.node_1.id_number
            j = 6 * element.node_2.id_number
            p_vector[i : i + 6] += enf[:6]
            p_vector[j : j + 6] += enf[6:]
        return p_vector

    def _get_reaction_forces(self) -> np.ndarray:
        """
        Calculates the reaction forces, applying them to each of the nodes
        and returning the full vector

        Returns
        -------
        The full reaction vector in global coordinates over all the nodes (6*n_nodes)
        """
        reactions = np.zeros(6 * self.geometry.n_nodes)

        for element in self.geometry.elements:
            p_glob = element.p_vector_glob()
            element.node_1.reactions += p_glob[:6]
            element.node_2.reactions += p_glob[6:]
        for node in self.geometry.nodes:
            p = node.p_vector()
            node.reactions += p
            idn = 6 * node.id_number
            reactions[idn : idn + 6] = node.reactions
        return reactions

    def _math_checks(self, k_matrix: np.ndarray):
        """
        Performs a series of checks on the model boundary conditions and the
        global stiffness matrix K, prior to the solution of the system of
        equations.

        Parameters
        ----------
        k_matrix:
            The global stiffness matrix to be checked ((6*n_nodes, 6*n_nodes))
        """
        if self.n_fixed_dofs < 6:  # noqa: PLR2004
            # TODO: check dimensionality of problem (1-D, 2-D, 3-D) and reduce
            # number of fixed DOFs required accordingly.
            raise StructuralError(
                "Insufficient boundary conditions to carry out "
                f"analysis:\n fixed_dofs: {self.fixed_dofs}"
                "\n|\tRigid-body motion."
            )
        if not self.fixed_dofs.all():
            raise StructuralError(
                "Can only solve systems in which all DOFs have "
                "been constrained at least once."
            )
        check_matrix_condition(k_matrix, 15)

    def _displacement_check(self, deflections: np.ndarray):
        """
        Checks to see if the displacements are not too big relative to the
        size of the model.

        Parameters
        ----------
        deflections:
            The vector of absolute displacements (6*n_nodes)
        """
        xmax, xmin, ymax, ymin, zmax, zmin = self.geometry.bounds()
        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
        length = np.sqrt(dx**2 + dy**2 + dz**2)

        # Get per node displacements
        deflections = deflections.reshape((self.geometry.n_nodes, -1))
        # Get absolute x, y, z displacement at each node
        deflections = np.sum(deflections[:, :3] ** 2, axis=1) ** 0.5
        u_max = np.max(deflections)

        if u_max >= length / R_LARGE_DISP:
            bluemira_warn(
                "structural::FiniteElementModel:\n Large displacements detected"
                "!\nYou can't trust the results..."
            )

    def _apply_boundary_conditions(
        self, k: np.ndarray, p: np.ndarray, method: str = "Przemieniecki"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies the boundary conditions to the matrices to make the problem
        solvable. This is creation of the "reduced" stiffness matrix and
        force vector, Kr and Pr.

        Parameters
        ----------
        k:
            The global stiffness matrix of the problem
        p:
            The global nodal force vector of the problem

        Returns
        -------
        kr:
            The reduced global stiffness matrix of the problem
        pr:
            The reduced global nodal force vector of the problem
        """
        # TODO: determine which approach is more suitable!

        # Modification of K and P matrices to remove the rigid-body DOFs
        # This is the method recommended by Przemieniecki in the book
        # Need to check which is faster with sparse matrices on real problems
        # This method is also easier to unittest!! Indices stay the same :)
        if method == "Przemieniecki":
            for i in self.fixed_dof_ids:
                # empty row or col of 0's with back-fill of diagonal term to 1
                entry = np.zeros(6 * self.geometry.n_nodes)
                entry[i] = 1
                k[i, :] = entry
                k[:, i] = entry
                p[i] = 0

        elif method == "deletion":
            # Removes the rows and columns of the fixed degrees of freedom
            # This reduces the size of the problem being solved, but this
            # may not be the fastest way!
            for i in sorted(self.fixed_dof_ids)[::-1]:
                # Travel backwards to preserve numbering
                k = np.delete(k, i, axis=0)
                k = np.delete(k, i, axis=1)
                p = np.delete(p, i)
        else:
            raise StructuralError(f"Unrecognised method: {method}.")
        return k, p

    def _apply_boundary_conditions_sparse(self, k, p):
        for i in self.fixed_dof_ids:
            row = np.zeros((1, 6 * self.geometry.n_nodes))
            row[0, i] = 1
            k[i, :] = row
            k[:, i] = row.T
            p[i] = 0
        return k, p

    def _apply_displacements(self, u_r):
        """
        Applies the displacements to the individual nodes in the Geometry
        """
        for i, node in enumerate(self.geometry.nodes):
            node.displacements = u_r[6 * i : 6 * i + 6]

    def plot(self, ax=None, **kwargs):
        """
        Plots the model geometry, boundary conditions, and loads
        """
        return self.geometry.plot(ax=ax, **kwargs)

    def solve(
        self, load_case: Optional[LoadCase] = None, sparse: bool = False
    ) -> Result:
        """
        Solves the system of linear equations for deflection and applies the
        deflections to the nodes and elements of the geometry

        Parameters
        ----------
        load_case:
            Will default to the loads applied to the elements and nodes in the
            geometry
        sparse:
            Whether or not to use sparse matrices to solve

        Returns
        -------
        The result of the FE analysis with the applied LoadCase
        """
        # Find model physical boundary conditions
        self.find_supports()

        # If no load_case is specified, take the class load_case
        if load_case is None:
            load_case = self.load_case

        # Get the nodal force vector
        self._apply_load_case(load_case)
        p = self._get_nodal_forces()

        # Check for cyclical symmetry
        self.cycle_sym = CyclicSymmetry(self.geometry, self.cycle_sym_ids)

        if not sparse:
            # Get the global stiffness matrix
            k = self.geometry.k_matrix()

            # Constrain matrix and vector
            kr, pr = self._apply_boundary_conditions(k, p, method="Przemieniecki")

            # Check boundary conditions and stiffness matrix
            self._math_checks(kr)

            # Apply cyclic boundary condition (if present)

            kr, pr = self.cycle_sym.apply_cyclic_symmetry(kr, pr)

            ur = np.linalg.solve(kr, pr)

            ur = self.cycle_sym.reorder(ur)

        else:
            print("sparse")
            k = self.geometry.k_matrix_sparse()

            kr, pr = self._apply_boundary_conditions_sparse(k, p)
            ur = spsolve(kr.tocsr(), pr)

        # Check for large displacements
        self._displacement_check(ur)

        # Apply the displacements to the nodes (and elements)
        self._apply_displacements(ur)

        # Calculate reaction forces at the nodes
        reactions = self._get_reaction_forces()

        # Interpolate the result between nodes, and calculate stresses
        self.geometry.interpolate()

        return Result(deepcopy(self.geometry), load_case, ur, reactions, self.cycle_sym)
