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
Finite element class
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from bluemira.structural.crosssection import CrossSection
    from bluemira.structural.material import StructuralMaterial
    from bluemira.structural.node import Node

import numpy as np

from bluemira.base.constants import GRAVITY
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.structural.constants import NU, N_INTERP, SD_LIMIT
from bluemira.structural.error import StructuralError
from bluemira.structural.loads import distributed_load, point_load
from bluemira.structural.node import get_midpoint
from bluemira.structural.stress import hermite_polynomials
from bluemira.structural.transformation import lambda_matrix

# TODO: Clean up some class stuff with cached_property decorators.
# Test some existing stuff (functools?), and your own custom class.
# Check speed and so on.
# Only bother doing this if you don't rewrite in C++


# @nb.jit(nopython=True, cache=True)
def _k_array(
    k11: float,
    k22: float,
    k33: float,
    k44: float,
    k55: float,
    k66: float,
    k35: float,
    k26: float,
    k511: float,
    k612: float,
) -> np.ndarray:
    """
    3-D stiffness local member stiffness matrix, generalised for cases with
    and without shear

    Parameters
    ----------
    Local stiffness matrix non-zero elements

    Returns
    -------
    The local member stiffness matrix
    """
    return np.array(
        [
            [k11, 0, 0, 0, 0, 0, -k11, 0, 0, 0, 0, 0],
            [0, k22, 0, 0, 0, k26, 0, -k22, 0, 0, 0, k26],
            [0, 0, k33, 0, k35, 0, 0, 0, -k33, 0, k35, 0],
            [0, 0, 0, k44, 0, 0, 0, 0, 0, -k44, 0, 0],
            [0, 0, k35, 0, k55, 0, 0, 0, -k35, 0, k511, 0],
            [0, k26, 0, 0, 0, k66, 0, -k26, 0, 0, 0, k612],
            [-k11, 0, 0, 0, 0, 0, k11, 0, 0, 0, 0, 0],
            [0, -k22, 0, 0, 0, -k26, 0, k22, 0, 0, 0, -k26],
            [0, 0, -k33, 0, -k35, 0, 0, 0, k33, 0, -k35, 0],
            [0, 0, 0, -k44, 0, 0, 0, 0, 0, k44, 0, 0],
            [0, 0, k35, 0, k511, 0, 0, 0, -k35, 0, k55, 0],
            [0, k26, 0, 0, 0, k612, 0, -k26, 0, 0, 0, k66],
        ]
    )


# @nb.jit(nopython=True, cache=True)
def local_k_shear(
    EA: float,  # noqa: N803
    EIyy: float,  # noqa: N803
    EIzz: float,  # noqa: N803
    ry: float,
    rz: float,
    L: float,  # noqa: N803
    GJ: float,  # noqa: N803
    A: float,
    A_sy: float,  # noqa: N803
    A_sz: float,  # noqa: N803
    nu: float = NU,
) -> np.ndarray:
    """
    3-D stiffness local member stiffness matrix, including shear deformation

    Parameters
    ----------
    EA:
        Youngs modulus x cross-sectional area
    EIyy:
        Youngs modulus x second moment of area about the element y-axis
    EIzz:
        Youngs modulus x second moment of area about the element z-axis
    L:
        The length of the beam
    GJ:
        The rigidity modulus x torsion constant
    A_sy:
        The shear area in the y-plane
    A_sz:
        The shear area in the z-plane
    nu:
        Poisson ratio

    Returns
    -------
    The local member stiffness matrix
    """
    phi_y = 24 * (1 + nu) * (A / A_sy) * (rz / L) ** 2  # y shear deformation parameter
    phi_z = 24 * (1 + nu) * (A / A_sz) * (ry / L) ** 2  # z shear deformation parameter
    k11 = EA / L
    k22 = 12 * EIzz / (L**3 * (1 + phi_y))
    k33 = 12 * EIyy / (L**3 * (1 + phi_z))
    k44 = GJ / L
    k55 = (4 + phi_z) * EIyy / (L * (1 + phi_z))
    k66 = (4 + phi_y) * EIzz / (L * (1 + phi_y))
    k35 = -6 * EIyy / (L**2 * (1 + phi_z))
    k26 = 6 * EIzz / (L**2 * (1 + phi_y))
    k511 = (2 - phi_z) * EIyy / (L * (1 + phi_z))
    k612 = (2 - phi_y) * EIzz / (L * (1 + phi_y))
    return _k_array(k11, k22, k33, k44, k55, k66, k35, k26, k511, k612)


# @nb.jit(nopython=True, cache=True)
def local_k(
    EA: float, EIyy: float, EIzz: float, L: float, GJ: float  # noqa: N803
) -> np.ndarray:
    """
    3-D stiffness local member stiffness matrix, including shear deformation

    Parameters
    ----------
    EA:
        Youngs modulus x cross-sectional area
    EIyy:
        Youngs modulus x second moment of area about the element y-axis
    EIzz:
        Youngs modulus x second moment of area about the element z-axis
    L:
        The length of the beam
    GJ:
        The rigidity modulus x torsion constant

    Returns
    -------
    The local member stiffness matrix
    """
    k11 = EA / L
    k22 = 12 * EIzz / L**3
    k33 = 12 * EIyy / L**3
    k44 = GJ / L
    k55 = 4 * EIyy / L
    k66 = 4 * EIzz / L
    k35 = -6 * EIyy / L**2
    k26 = 6 * EIzz / L**2
    k511 = 2 * EIyy / L
    k612 = 2 * EIzz / L
    return _k_array(k11, k22, k33, k44, k55, k66, k35, k26, k511, k612)


class Element:
    """
    A 3-D beam element (Euler-Bernoulli type)

    Parameters
    ----------
    node_1:
        The first node
    node_2:
        The second node
    id_number:
        The ID number of this element
    cross_section:
        The CrossSection property object of the element
    material:
        The Material property object of the element
    """

    HERMITE_POLYS = hermite_polynomials(N_INTERP)

    __slots__ = (
        "node_1",
        "node_2",
        "id_number",
        "loads",
        "stresses",
        "max_stress",
        "safety_factor",
        "shapes",
        "material",
        "_properties",
        "_material",
        "_cross_section",
        "_length",
        "_weight",
        "_k_matrix",
        "_lambda_matrix",
        "_k_matrix_glob",
        "_s_functs",
    )

    def __init__(
        self,
        node_1: Node,
        node_2: Node,
        id_number: int,
        cross_section: CrossSection,
        material: Optional[StructuralMaterial] = None,
    ):
        # Utility properties
        self.node_1 = node_1
        self.node_2 = node_2
        self.id_number = id_number

        # Public properties
        self.loads = []

        # Calculated properties
        self.shapes = None
        self.stresses = None
        self.max_stress = None
        self.safety_factor = None

        # Private properties
        self._material = material  # Record input material
        self.material = None
        self._properties = self._process_properties(cross_section, material)

        self._cross_section = cross_section

        # Cheap memo-isation of matrices (headache-free cached properties)
        self._length = None
        self._weight = None
        self._k_matrix = None
        self._lambda_matrix = None
        self._k_matrix_glob = None

        # Private construction utilities
        self._s_functs = None

    def _process_properties(self, cross_section, material):
        """
        Handles cross-sectional and material properties, including if a
        composite material cross-section is specified.
        """
        properties = {}
        if material is None:
            # A composite cross-section was hopefully specified

            properties["EA"] = cross_section.ea
            properties["EIyy"] = cross_section.ei_yy
            properties["EIzz"] = cross_section.ei_zz
            properties["GJ"] = cross_section.gj
            properties["nu"] = cross_section.nu
            properties["ry"] = cross_section.ry
            properties["rz"] = cross_section.rz
            properties["A"] = cross_section.area
            properties["rho"] = cross_section.rho  # area-weighted density

            # Override material=None with a list of materials from the CS
            self.material = cross_section.material
        else:
            # Single material weight cross-section properties
            e_mat, g_mat = material.E, material.G
            properties["EA"] = e_mat * cross_section.area
            properties["EIyy"] = e_mat * cross_section.i_yy
            properties["EIzz"] = e_mat * cross_section.i_zz
            properties["GJ"] = g_mat * cross_section.j
            properties["ry"] = cross_section.ry
            properties["rz"] = cross_section.rz
            properties["rho"] = material.rho
            properties["A"] = cross_section.area
            self.material = material

        return properties

    @property
    def length(self) -> float:
        """
        Element length
        """
        if self._length is None:
            self._length = self.node_2.distance_to_other(self.node_1)
        return self._length

    @property
    def weight(self) -> float:
        """
        Element self-weight force per unit length
        """
        if self._weight is None:
            mass = self._properties["A"] * self._properties["rho"]
            self._weight = GRAVITY * mass
        return self._weight

    @property
    def mid_point(self) -> np.ndarray:
        """
        The mid point of the Element

        Returns
        -------
        vector: np.array(3)
            The [x, y, z] vector of the midpoint
        """
        return get_midpoint(self.node_1, self.node_2)

    @property
    def space_vector(self):
        """
        Spatial vector of the Element

        Returns
        -------
        vector: np.array(3)
            The [dx, dy, dz] vector of the Element
        """
        return np.array(
            [
                self.node_2.x - self.node_1.x,
                self.node_2.y - self.node_1.y,
                self.node_2.z - self.node_1.z,
            ]
        )

    @property
    def displacements(self) -> np.ndarray:
        """
        Element global displacement vector at nodes
        """
        u = np.zeros(12)
        u[:6] = self.node_1.displacements
        u[6:] = self.node_2.displacements
        return u

    @property
    def max_displacement(self) -> float:
        """
        Maximum element absolute displacement values

        Returns
        -------
        The maximum absolute deflection distance of the two Nodes
        """
        d_1 = self.node_1.displacements[:3]
        d_2 = self.node_2.displacements[:3]
        return max(np.sqrt(np.sum(d_1**2)), np.sqrt(np.sum(d_2**2)))

    @property
    def k_matrix(self) -> np.ndarray:
        """
        Element stiffness matrix in local coordinates
        """
        if self._k_matrix is None:
            p = self._properties
            k = local_k(p["EA"], p["EIyy"], p["EIzz"], self.length, p["GJ"])

            if (p["ry"] / self.length > SD_LIMIT) or (p["rz"] / self.length < SD_LIMIT):
                bluemira_warn(
                    "Thick cross-section detected. Slender beam approximation being"
                    " used, so be careful."
                )

            self._k_matrix = k

        return self._k_matrix

    @property
    def k_matrix_glob(self) -> np.ndarray:
        """
        Element stiffness matrix in global coordinates
        """
        if self._k_matrix_glob is None:
            lambda_m = self.lambda_matrix
            self._k_matrix_glob = lambda_m.T @ self.k_matrix @ lambda_m

        return self._k_matrix_glob

    @property
    def lambda_matrix(self) -> np.ndarray:
        """
        Transformation (direction cosine) matrix

        Notes
        -----
        This matrix is cached but involves properties that may be externally
        modified (e.g. by moving a node). Be careful to reset _lambda_matrix to
        None if you are doing this, so that it gets recalculated upon call.
        """
        if self._lambda_matrix is None:
            d_x, d_y, d_z = self.space_vector

            if d_x == 0 and d_y == 0 and d_z == 0:
                raise StructuralError("Coincident nodes!?")

            self._lambda_matrix = lambda_matrix(d_x, d_y, d_z)

        return self._lambda_matrix

    def add_load(self, load: Dict[str, float]):
        """
        Applies a load to the Element object.

        Parameters
        ----------
        load:
            The dictionary of load values (in local coordinates)
        """
        self.loads.append(load)

    def clear_loads(self):
        """
        Clears all loads applied to the Element
        """
        self.loads = []

    def clear_cache(self):
        """
        Clears all cached properties and matrices for the Element. Use if
        an Element's geometry has been modified.
        """
        self._length = None
        self._weight = None
        self._k_matrix = None
        self._lambda_matrix = None
        self._k_matrix_glob = None
        self._s_functs = None

    def u_vector(self) -> np.ndarray:
        """
        Element local displacement vector
        """
        return self.lambda_matrix @ self.displacements

    def p_vector(self) -> np.ndarray:
        """
        The local force vector of the element
        """
        return self.k_matrix @ self.u_vector() - self._equivalent_node_forces()

    def p_vector_glob(self) -> np.ndarray:
        """
        The global force vector of the element
        """
        return self.lambda_matrix.T @ self.p_vector()

    def equivalent_node_forces(self) -> np.ndarray:
        """
        Equivalent concentrated forces in global coordinates
        """
        return self.lambda_matrix.T @ self._equivalent_node_forces()

    def _equivalent_node_forces(self) -> np.ndarray:
        """
        Element local nodal force vector

        Equivalent concentrated forces in local coordinates
        """
        enf = np.zeros(12)
        for load in self.loads:
            if load["type"] == "Element Load":
                enf += point_load(load["Q"], load["x"], self.length, load["sub_type"])
            elif load["type"] == "Distributed Load":
                enf += distributed_load(load["w"], self.length, load["sub_type"])
            else:
                raise StructuralError(f'Unknown load type: "{load["type"]}"')

        return enf

    @property
    def _shape_functions(self):
        if self._s_functs is None:

            def multiply_length(i):
                n_matrix = np.copy(self.HERMITE_POLYS[i])
                n_matrix[:, 1] *= self.length
                n_matrix[:, 3] *= self.length
                return n_matrix

            self._s_functs = [multiply_length(0), multiply_length(1), multiply_length(2)]

        return self._s_functs

    def interpolate(self, scale: float):
        """
        Interpolates the displacement of the beam with Hermite polynomial
        shape functions to obtain inter-node displacement and stress

        Parameters
        ----------
        scale:
            The scale at which to calculate the interpolated displacements
        """
        length = self.length
        u = self.u_vector()  # Local displacements
        s_funcs = self._shape_functions

        d = np.zeros((N_INTERP, 2))  # displacement
        m = np.zeros((N_INTERP, 2))  # curvature
        v = np.zeros((N_INTERP, 2))  # shear

        for i in range(2):
            if i == 0:
                # dy, rz
                u1d = np.array([u[1], u[5], u[7], u[11]])
            else:
                # dz, ry
                u1d = np.array([u[2], u[4], u[8], u[10]])

            d[:, i] = np.dot(s_funcs[0], u1d)
            m[:, i] = np.dot(s_funcs[1], u1d) / length**2
            v[:, i] = np.dot(s_funcs[2], u1d) / length**3
        self.calculate_stress(u, m)
        self.calculate_shape(u, d, m, v, scale)

    def calculate_stress(self, u, m_matrix):
        """
        Calculates the stresses in the Element, using Hermite polynomials for
        interpolation between the Nodes
        """
        xsections = self._cross_section
        materials = self.material

        # Handle both single and multiple composite cross-sections
        if not isinstance(xsections, list):
            xsections = [xsections]
        if not isinstance(materials, list):
            materials = [materials]

        stresses = []
        safety_factors = []
        for c_s, mat in zip(xsections, materials):
            y = c_s.y - c_s.centroid[1]  # y-distances to centroid
            z = c_s.z - c_s.centroid[2]  # z-distances to centroid

            n_points = len(y)
            # Tile coordinates
            y = np.ones((N_INTERP, 1)) @ y.reshape(1, -1)
            z = np.ones((N_INTERP, 1)) @ z.reshape(1, -1)

            # Tile curvature (M/EI)
            kappa_z = m_matrix[:, 0].reshape(-1, 1) @ np.ones((1, n_points))
            kappa_y = m_matrix[:, 1].reshape(-1, 1) @ np.ones((1, n_points))

            # Bending stresses (at all interp points, at all loop points)
            sigma_z = -mat.E * y * kappa_z
            sigma_y = -mat.E * z * kappa_y

            # Axial stress = E*axial strain (constant along Element)
            du = u[6] - u[0]
            sigma_axial = mat.E * du / self.length

            stress = sigma_axial + sigma_y + sigma_z
            stresses.append(stress)

            argmax = np.argmax(np.abs(stress))
            safety_factor = stress.flatten()[argmax] / mat.sigma_y
            safety_factors.append(safety_factor)

        part_index = int(np.argmax(np.abs(safety_factors)))
        max_index = np.argmax(np.abs(stresses[part_index]))

        self.stresses = stresses[part_index]  # Only store most stressed mat
        # Store peak stress (with sign)
        self.max_stress = stresses[part_index].flatten()[max_index]
        self.safety_factor = min(np.abs(safety_factors))

    def calculate_shape(self, u, d, _m, _v, scale):
        """
        Calculates the interpolated shape of the Element
        """
        # Global interpolated displacements
        u = np.array([np.linspace(u[0], u[1], N_INTERP), d[:, 0], d[:, 1]])
        displacements = self.lambda_matrix[0:3, 0:3].T @ u

        # Global interpolated positions
        c = np.array(
            [
                np.linspace(self.node_1.x, self.node_2.x, N_INTERP),
                np.linspace(self.node_1.y, self.node_2.y, N_INTERP),
                np.linspace(self.node_1.z, self.node_2.z, N_INTERP),
            ]
        )
        self.shapes = c + scale * displacements
