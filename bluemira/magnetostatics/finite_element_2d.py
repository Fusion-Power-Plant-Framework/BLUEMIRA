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
Solver for a 2D magnetostatic problem with cylindrical symmetry
"""

from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import dolfin
import numpy as np

from bluemira.base.constants import MU_0


def Bz_coil_axis(
    r: float,
    z: Optional[float] = 0,
    pz: Optional[float] = 0,
    current: Optional[float] = 1,
) -> float:
    """
    Calculate the theoretical vertical magnetic field of a filament coil
    (of radius r and centred in (0, z)) on a point on the coil axis at
    a distance pz from the axis origin.

    Parameters
    ----------
    r:
        Coil radius [m]
    z:
        Vertical position of the coil centroid [m]
    pz:
        Vertical position of the point on the axis on which the magnetic field
        shall be calculated [m]
    current:
        Current of the coil [A]

    Returns
    -------
    Vertical magnetic field on the axis [T]

    Notes
    -----
    \t:math:`\\dfrac{1}{2}\\dfrac{\\mu_{0}Ir^2}{(r^{2}+(pz-z)^{2})^{3/2}}`
    """
    return 0.5 * MU_0 * current * r**2 / (r**2 + (pz - z) ** 2) ** 1.5


def _convert_const_to_dolfin(value: float):
    """Convert a constant value to a dolfin function"""
    if not isinstance(value, (int, float)):
        raise TypeError("Value must be integer or float.")

    return dolfin.Constant(value)


class ScalarSubFunc(dolfin.UserExpression):
    """
    Create a dolfin UserExpression from a set of functions defined in the subdomains

    Parameters
    ----------
    func_list:
        list of functions to be interpolated into the subdomains. Int and float values
        are considered as constant functions. Any other callable function must return
        a single value.
    mark_list:
        list of markers that identify the subdomain in which the respective functions
        of func_list must to be applied.
    subdomains:
        the whole subdomains mesh function
    """

    def __init__(
        self,
        func_list: Union[
            Iterable[Union[float, Callable[[Any], float]]], float, Callable[[Any], float]
        ],
        mark_list: Optional[Iterable[int]] = None,
        subdomains: Optional[dolfin.cpp.mesh.MeshFunctionSizet] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.functions = self.check_functions(func_list)
        self.markers = mark_list
        self.subdomains = subdomains

    @staticmethod
    def check_functions(
        functions: Union[Iterable[Union[float, Callable]], float, Callable],
    ) -> Iterable[Union[float, Callable]]:
        """Check if the argument is a function or a list of functions"""
        if not isinstance(functions, Iterable):
            functions = [functions]
        if all(isinstance(f, (float, Callable)) for f in functions):
            return functions
        raise ValueError(
            "Accepted functions are instance of (int, float, Callable)or a list of them."
        )

    def eval_cell(self, values: List, x: float, cell):
        """Evaluate the value on each cell"""
        if self.markers is None or self.subdomains is None:
            func = self.functions[0]
        else:
            m = self.subdomains[cell.index]
            func = (
                self.functions[np.nonzero(np.array(self.markers) == m)[0][0]]
                if m in self.markers
                else 0
            )
        if callable(func):
            values[0] = func(x)
        elif isinstance(func, (int, float)):
            values[0] = func
        else:
            raise TypeError(f"{func} is not callable or is not a constant")

    @staticmethod
    def value_shape() -> Tuple:
        """
        Value_shape function (necessary for a UserExpression)
        https://fenicsproject.discourse.group/t/problems-interpolating-a-userexpression-and-plotting-it/1303
        """
        return ()


class FemMagnetostatic2d:
    """
    A 2D magnetostatic solver. The solver is thought as support for the fem fixed
    boundary module and it is limited to axisymmetric magnetostatic problem
    with toroidal current sources. The Maxwell equations, as function of the poloidal
    magnetic flux (:math:`\\Psi`), are then reduced to the form ([Zohm]_, page 25):

    .. math::
        r^2 \\nabla\\cdot\\left(\\frac{\\nabla\\Psi}{r^2}\\right) = 2
        \\pi r \\mu_0 J_{\\Phi}

    whose weak formulation is defined as ([Villone]_):

    .. math::
        \\int_{D_p} {\\frac{1}{r}}{\\nabla}{\\Psi}{\\cdot}{\\nabla} v \\,dr\\,dz = 2
        \\pi \\mu_0 \\int_{D_p} J_{\\Phi} v \\,dr\\,dz

    where :math:`v` is the basis element function of the defined functional subspace
    :math:`V`.

    .. [Zohm] H. Zohm, Magnetohydrodynamic Stability of Tokamaks, Wiley-VCH, Germany,
       2015
    .. [Villone] VILLONE, F. et al. Plasma Phys. Control. Fusion 55 (2013) 095008,
       https://doi.org/10.1088/0741-3335/55/9/095008

    Parameters
    ----------
    p_order:
        Order of the approximating polynomial basis functions
    """

    def __init__(self, p_order: int = 2):
        self.p_order = p_order
        self.mesh = None
        self.a = None
        self.u = None
        self.v = None
        self.V = None
        self.g = None
        self.L = None
        self.boundaries = None
        self.bcs = None

        self.psi = None
        self.B = None

    def set_mesh(
        self,
        mesh: Union[dolfin.Mesh, str],
        boundaries: Optional[Union[dolfin.Mesh, str]] = None,
    ):
        """
        Set the mesh for the solver

        Parameters
        ----------
        mesh:
            Filename of the xml file with the mesh definition or a dolfin mesh
        boundaries:
            Filename of the xml file with the boundaries definition or a MeshFunction
            that defines the boundaries
        """
        # check whether mesh is a filename or a mesh, then load it or use it
        self.mesh = dolfin.Mesh(mesh) if isinstance(mesh, str) else mesh

        # define boundaries
        if boundaries is None:
            # initialize the MeshFunction
            self.boundaries = dolfin.MeshFunction(
                "size_t", mesh, mesh.topology().dim() - 1
            )
        elif isinstance(boundaries, str):
            # check weather boundaries is a filename or a MeshFunction,
            # then load it or use it
            self.boundaries = dolfin.MeshFunction(
                "size_t", self.mesh, boundaries
            )  # define the boundaries
        else:
            self.boundaries = boundaries

        # define the function space and bilinear forms
        # the Continuos Galerkin function space has been chosen as suitable for the
        # solution of the magnetostatic weak formulation in a Soblev Space H1(D)
        self.V = dolfin.FunctionSpace(self.mesh, "CG", self.p_order)

        # define trial and test functions
        self.u = dolfin.TrialFunction(self.V)
        self.v = dolfin.TestFunction(self.V)

        # Define r
        r = dolfin.Expression("x[0]", degree=self.p_order)

        self.a = (
            1
            / (2.0 * dolfin.pi * MU_0)
            * (1 / r * dolfin.dot(dolfin.grad(self.u), dolfin.grad(self.v)))
            * dolfin.dx
        )

        # initialize solution
        self.psi = dolfin.Function(self.V)
        self.psi.set_allow_extrapolation(True)

        # initialize g to zero
        self.g = dolfin.Function(self.V)

    def define_g(self, g: Union[dolfin.Expression, dolfin.Function]):
        """
        Define g, the right hand side function of the Poisson problem

        Parameters
        ----------
        g:
            Right hand side function of the Poisson problem
        """
        self.g = g

    def solve(
        self,
        dirichlet_bc_function: Optional[
            Union[dolfin.Expression, dolfin.Function]
        ] = None,
        dirichlet_marker: Optional[int] = None,
        neumann_bc_function: Optional[Union[dolfin.Expression, dolfin.Function]] = None,
    ) -> dolfin.Function:
        """
        Solve the weak formulation maxwell equation given a right hand side g,
        Dirichlet and Neumann boundary conditions.

        Parameters
        ----------
        dirichlet_bc_function:
            Dirichlet boundary condition function
        dirichlet_marker:
            Identification number for the dirichlet boundary
        neumann_bc_function:
            Neumann boundary condition function

        Returns
        -------
        Poloidal magnetic flux function as solution of the magnetostatic problem
        """
        if neumann_bc_function is None:
            neumann_bc_function = dolfin.Expression("0.0", degree=self.p_order)

        # define the right hand side
        self.L = self.g * self.v * dolfin.dx - neumann_bc_function * self.v * dolfin.ds

        # define the Dirichlet boundary conditions
        if dirichlet_bc_function is None:
            dirichlet_bc_function = dolfin.Expression("0.0", degree=self.p_order)
            dirichlet_bc = dolfin.DirichletBC(
                self.V, dirichlet_bc_function, "on_boundary"
            )
        else:
            dirichlet_bc = dolfin.DirichletBC(
                self.V, dirichlet_bc_function, self.boundaries, dirichlet_marker
            )
        self.bcs = [dirichlet_bc]

        # solve the system taking into account the boundary conditions
        dolfin.solve(
            self.a == self.L,
            self.psi,
            self.bcs,
            solver_parameters={"linear_solver": "default"},
        )

        return self.psi

    def calculate_b(self) -> dolfin.Function:
        """
        Calculates the magnetic field intensity from psi

        Note: code from Fenics_tutorial (
        https://link.springer.com/book/10.1007/978-3-319-52462-7), pag. 104
        """
        # new function space for mapping B as vector
        w = dolfin.VectorFunctionSpace(self.mesh, "CG", 1)

        r = dolfin.Expression("x[0]", degree=1)

        # calculate derivatives
        Bx = -self.psi.dx(1) / (2 * dolfin.pi * r)
        Bz = self.psi.dx(0) / (2 * dolfin.pi * r)

        # project B as vector to new function space
        self.B = dolfin.project(dolfin.as_vector((Bx, Bz)), w)

        return self.B
