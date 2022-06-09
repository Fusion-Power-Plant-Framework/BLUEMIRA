# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
Attempt at recreating the EU-DEMO 2017 reference equilibria from a known coilset.
"""

# %%[markdown]

# Here we explore how to optimise equilibria, coil currents, and coil positions.

# This is an in-depth example walking you through many of the objects, approaches,
# and optimisation problems that are often used when designing plasma equilibria and
# poloidal field coils.

# There are many ways of optimising equilibria, and this example shows just one
# relatively crude approach. The choice of constraints, optimisation algorithms, and even
# the sequence of operations has a big influence on the outcome. It is a bit of a dark
# art, and over time you will hopefully find an approach that works for your problem.

# Heavily constraining the plasma shape as we do here is not particularly robust, or
# even philosophically "right". It's however a common approach and comes in useful when
# one wants to optimise coil positions without affecting the plasma too much.

# %%

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython

from bluemira.display import plot_defaults
from bluemira.equilibria.coils import Coil, CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.opt_constraints import (
    CoilFieldConstraints,
    CoilForceConstraints,
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
    PsiConstraint,
)
from bluemira.equilibria.opt_problems import (
    MinimalCurrentCOP,
    PulsedNestedPositionCOP,
    TikhonovCurrentCOP,
    UnconstrainedTikhonovCurrentGradientCOP,
)
from bluemira.equilibria.profiles import CustomProfile
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.equilibria.solve import DudsonConvergence, PicardIterator
from bluemira.geometry.tools import make_polygon
from bluemira.utilities.optimiser import Optimiser
from bluemira.utilities.positioning import PositionMapper, RegionInterpolator

plot_defaults()

try:
    get_ipython().run_line_magic("matplotlib", "qt")
except AttributeError:
    pass

# %%[markdown]

# First let's create our inital coilset. This is taken from a reference EU-DEMO design
# but as you will see we will change this later on.

# %%
x = [5.4, 14.0, 17.75, 17.75, 14.0, 7.0, 2.77, 2.77, 2.77, 2.77, 2.77]
z = [9.26, 7.9, 2.5, -2.5, -7.9, -10.5, 7.07, 4.08, -0.4, -4.88, -7.86]
dx = [0.6, 0.7, 0.5, 0.5, 0.7, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4]
dz = [0.6, 0.7, 0.5, 0.5, 0.7, 1.0, 2.99 / 2, 2.99 / 2, 5.97 / 2, 2.99 / 2, 2.99 / 2]

coils = []
j = 1
for i, (xi, zi, dxi, dzi) in enumerate(zip(x, z, dx, dz)):
    if j > 6:
        j = 1
    ctype = "PF" if i < 6 else "CS"
    coil = Coil(
        xi,
        zi,
        current=0,
        dx=dxi,
        dz=dzi,
        ctype=ctype,
        control=True,
        name=f"{ctype}_{j}",
    )
    coils.append(coil)
    j += 1

coilset = CoilSet(coils)

# %%[markdown]

# Now we can also specify our coilset a little further, by assigning maximum current
# densities and peak fields. This can then be used in the constraints for our
# optimisation problems.

# We also fix the sizes of our coils for this example, and 'mesh' them. Unless otherwise
# specified a `Coil` is represented by a single current filament. `mesh`ing here refers
# to the sub-division of this coil filament into several filaments equi-spaced around
# the coil cross-section.

# %%

coilset.assign_coil_materials("CS", j_max=16.5, b_max=12.5)
coilset.assign_coil_materials("PF", j_max=12.5, b_max=11.0)

# Later on, we will optimise the PF coil positions, but for the CS coils we can fix sizes
# and mesh them already.

for coil in coilset.coils.values():
    if coil.ctype == "CS":
        coil.flag_sizefix = True
        coil.mesh_coil(0.3)

# %%[markdown]

# Now, we set up our grid, equilibrium, and profiles.

# We'll just use a `CustomProfile` for now, but you can also use a `BetaIpProfile` with
# flux function parameterisation if you want to directly constrain poloidal beta
# values in your equilibrium optimisation.

# %%

# Machine parameters
R_0 = 8.938
A = 3.1
B_0 = 4.8901  # T
I_p = 19.07e6  # A

grid = Grid(3.0, 13.0, -10.0, 10.0, 65, 65)

profiles = CustomProfile(
    np.array([86856, 86506, 84731, 80784, 74159, 64576, 52030, 36918, 20314, 4807, 0.0]),
    -np.array(
        [0.125, 0.124, 0.122, 0.116, 0.106, 0.093, 0.074, 0.053, 0.029, 0.007, 0.0]
    ),
    R_0=R_0,
    B_0=B_0,
    I_p=I_p,
)

eq = Equilibrium(coilset, grid, profiles, psi=None)

# %%[markdown]

# Now we need to specify some constraints on the plasma.

# We'll instantiate a parameterisation for the last closed flux surface (LCFS) which
# tends to do a good job at describing an EU-DEMO-like single null plasma.

# We'll use this to specify some constraints on the plasma equilibrium problem.

# %%

lcfs_parameterisation = JohnerLCFS(
    {
        "r_0": {"value": R_0},
        "z_0": {"value": 0.0},
        "a": {"value": R_0 / A},
        "kappa_u": {"value": 1.6},
        "kappa_l": {"value": 1.9},
        "delta_u": {"value": 0.4},
        "delta_l": {"value": 0.4},
        "phi_u_neg": {"value": 0.0},
        "phi_u_pos": {"value": 0.0},
        "phi_l_neg": {"value": 45.0},
        "phi_l_pos": {"value": 30.0},
    }
)

lcfs = lcfs_parameterisation.create_shape().discretize(byedges=True, ndiscr=50)

x_bdry, z_bdry = lcfs.x, lcfs.z
arg_inner = np.argmin(x_bdry)

isoflux = IsofluxConstraint(
    x_bdry,
    z_bdry,
    x_bdry[arg_inner],
    z_bdry[arg_inner],
    tolerance=0.5,  # Difficult to choose...
    constraint_value=0.0,  # Difficult to choose...
)

xp_idx = np.argmin(z_bdry)
x_point = FieldNullConstraint(
    x_bdry[xp_idx],
    z_bdry[xp_idx],
    tolerance=1e-4,
)

# %%[markdown]

# It's often very useful to solve an unconstrained optimised problem in order to get
# an initial guess for the equilibrium result. The initial equilibrium can be used as
# the "starting guess" for subsequent constrained optimisation problems.

# This is done by using the magnetic constraints in a "set" for which the error is then
# minimised with an L2 norm and a Tikhonov regularisation on the currents.

# Note that when we use equilibrium constraints in a "constraint set" as part of an
# optimisation objective (as is the case here) the tolerances are not used actively.

# We can use this to optimise the current gradients during the solution of the
# equilibrium until convergence.

# %%

current_opt_problem = UnconstrainedTikhonovCurrentGradientCOP(
    coilset, eq, MagneticConstraintSet([isoflux, x_point]), gamma=1e-7
)

program = PicardIterator(
    eq, current_opt_problem, fixed_coils=True, relaxation=0.2, plot=True
)
program()
plt.close("all")


# %%[markdown]

# Now say we want to use bounds on our current vector, and that we want to solve a
# constrained optimisation problem.

# We can minimise the error on our target isoflux surface set with some bounds on the
# current vector, and some additional constraints.

# First let's set up some constraints on the coils (peak fields and peak vertical forces)
# are common constraints. We can even use our X-point constraint from earlier as a
# constraint in the optimiser. In other words, we don't need to lump it together with the
# isoflux target minimisation objective.

# We then instantiate a new optimisation problem, and use this in a Picard iteration
# scheme, using the previously converged equilibrium as a starting point.

# Note that here we are optimising the current vector and not the current gradient
# vector.

# %%

field_constraints = CoilFieldConstraints(
    eq.coilset, eq.coilset.get_max_fields(), tolerance=1e-6
)

force_constraints = CoilForceConstraints(
    eq.coilset,
    PF_Fz_max=450,
    CS_Fz_sum_max=300,
    CS_Fz_sep_max=250,
    tolerance=5e-5,
)

current_opt_problem = TikhonovCurrentCOP(
    coilset,
    eq,
    targets=MagneticConstraintSet([isoflux]),
    gamma=0.0,
    optimiser=Optimiser("SLSQP", opt_conditions={"max_eval": 2000, "ftol_rel": 1e-6}),
    max_currents=coilset.get_max_currents(0.0),
    constraints=[x_point, field_constraints, force_constraints],
)

program = PicardIterator(
    eq,
    current_opt_problem,
    fixed_coils=True,
    convergence=DudsonConvergence(1e-4),
    relaxation=0.1,
    plot=False,
)
program()


# %%[markdown]

# Now let's say we don't actually want to minimise the error, but we want to minimise the
# coil currents, and use the constraints that we specified above as actual constraints
# in the optimisation problem (rather than in the objective function as above)

# Note that here we've got rather a lot of constraints, and that we need to choose the
# value and tolerance of the isoflux constraint (in particular) wisely.

# Too strict a tolerance will likely result in an unsuccessful optimisation with the
# final result potentially violating the constraint and probably not being an actual
# optimum.

# %%

minimal_current_eq = deepcopy(eq)
minimal_current_coilset = deepcopy(coilset)
minimal_current_opt_problem = MinimalCurrentCOP(
    minimal_current_coilset,
    minimal_current_eq,
    Optimiser(
        "SLSQP", opt_conditions={"max_eval": 5000, "ftol_rel": 1e-6, "xtol_rel": 1e-6}
    ),
    max_currents=coilset.get_max_currents(0.0),
    constraints=[isoflux, x_point, field_constraints, force_constraints],
)

program = PicardIterator(
    minimal_current_eq,
    minimal_current_opt_problem,
    fixed_coils=True,
    convergence=DudsonConvergence(1e-4),
    relaxation=0.1,
    plot=False,
)
program()

print(
    f"Total currents from minimal error optimisation problem: {np.sum(np.abs(coilset.get_control_currents()))/1e6:.2f} MA"
)
print(
    f"Total currents from minimal current optimisation problem: {np.sum(np.abs(minimal_current_coilset.get_control_currents()))/1e6:.2f} MA"
)


# %%[markdown]

# Coil position optimisation

# Now, say that we want to optimise the positions the PF coils, and the currents of the
# entire CoilSet for two different snapshots in a pulse. These snapshots are typically
# the start of flat-top (SOF) and end of flat-top (EOF) as they represent the most
# challenging conditions for the coils and plasma shape.

# Here we set the flux at the desired LCFS to be 50 V.s and -100 V.s for the SOF and EOF,
# respectively. In this example, this is an arbitrary decision. In reality, this would
# relate to the desired pulse duration for a given machine.

# We're going to optimise the positions for an objective function that takes the maximum
# objective function value of two current sub-optimisation problems.

# This is what we refer to as a `nested` optimisation, in other words that the
# positions and the currents (in two different situations) are being optimised
# separately.

# First we specify the sub-optimisation problems (objective functions and constraints).
# Then we specify the position optimisation problem for a single current sub-optimisation
# problem.


isoflux = IsofluxConstraint(
    x_bdry,
    z_bdry,
    x_bdry[arg_inner],
    z_bdry[arg_inner],
    tolerance=1e-3,
)

sof = deepcopy(eq)
sof_psi_boundary = PsiConstraint(
    x_bdry[arg_inner],
    z_bdry[arg_inner],
    target_value=50 / 2 / np.pi,
    tolerance=1e-3,
)

eof = deepcopy(eq)
eof_psi_boundary = PsiConstraint(
    x_bdry[arg_inner],
    z_bdry[arg_inner],
    target_value=-150 / 2 / np.pi,
    tolerance=1e-3,
)

current_opt_problem_sof = TikhonovCurrentCOP(
    sof.coilset,
    sof,
    targets=MagneticConstraintSet([isoflux]),
    gamma=1e-12,
    optimiser=Optimiser(
        "SLSQP", opt_conditions={"max_eval": 2000, "ftol_rel": 1e-6, "xtol_rel": 1e-6}
    ),
    max_currents=coilset.get_max_currents(I_p),
    constraints=[sof_psi_boundary, x_point, field_constraints, force_constraints],
)

current_opt_problem_eof = TikhonovCurrentCOP(
    eof.coilset,
    eof,
    targets=MagneticConstraintSet([isoflux]),
    gamma=1e-12,
    optimiser=Optimiser(
        "COBYLA", opt_conditions={"max_eval": 5000, "ftol_rel": 1e-6, "xtol_rel": 1e-6}
    ),
    max_currents=coilset.get_max_currents(I_p),
    constraints=[eof_psi_boundary, x_point, field_constraints, force_constraints],
)

# %%[markdown]

# We set up a position mapping of the regions in which we would like the PF coils
# to be. The positions themselves are bounded by the specification of the
# `RegionInterpolator`s.

# Finally, we specify our position optimisation problem, in this case with the two
# previously defined current sub-optimisation problems (but we could specify more if we
# wanted to).

# Typically the currents can be varied linearly from the SOF to the EOF, so there isn't
# really much point in doing more than two different equilibria here from an optimisation
# perspective.

# Note that there are no constraints here, and if we wanted to add some they would have
# to pertain to the position vector in some form.

# For each set of positions, we treat the plasma contribution as being "frozen" and
# optimise the coil currents (with the various constraints). This works as for relatively
# good starting guesses (converged equilibria) the plasma contribution to the various
# constraints does not change much when the equilibrium is subsequently converged.

# For the sake of brevity, we set the maximum number of iterations for the position
# optimisation to 50. This is unlikely to find the best solution given our specified
# coil regions, but demonstrates the principle with acceptable run-times.

# %%

old_coilset = deepcopy(coilset)
old_eq = deepcopy(eq)
region_interpolators = {}
for coil in coilset.coils.values():
    if coil.ctype == "PF":
        x, z = coil.x, coil.z
        region = make_polygon(
            {"x": [x - 1, x + 1, x + 1, x - 1], "z": [z - 1, z - 1, z + 1, z + 1]},
            closed=True,
        )
        region_interpolators[coil.name] = RegionInterpolator(region)

position_mapper = PositionMapper(region_interpolators)

position_opt_problem = PulsedNestedPositionCOP(
    coilset,
    position_mapper,
    sub_opt_problems=[current_opt_problem_sof, current_opt_problem_eof],
    optimiser=Optimiser(
        "COBYLA", opt_conditions={"max_eval": 50, "ftol_rel": 1e-6, "xtol_rel": 1e-6}
    ),
    debug=False,
)

optimised_coilset = position_opt_problem.optimise(verbose=True)

# %%[markdown]

# We've just optimised the PF coil positions using a single current filament at the
# centre of each PF coil. This is a reasonable approximation when performing a position
# optimisation, but we probably want to do better when it comes to our final equilbria.
# We will figure out the appropriate sizes of the PF coils, and fix them and mesh them
# accordingly.

# We also need to remember to update the bounds of the current optimisation problem!

# %%

sof_pf_currents = sof.coilset.get_control_currents()[: sof.coilset.n_PF]
eof_pf_currents = eof.coilset.get_control_currents()[: sof.coilset.n_PF]
max_pf_currents = np.max(np.abs([sof_pf_currents, eof_pf_currents]), axis=0)
pf_coil_names = optimised_coilset.get_PF_names()

max_cs_currents = optimised_coilset.get_max_currents(0.0)[optimised_coilset.n_PF :]

max_currents = np.concatenate([max_pf_currents, max_cs_currents])

for problem in [current_opt_problem_sof, current_opt_problem_eof]:
    for pf_name, max_current in zip(pf_coil_names, max_pf_currents):
        problem.eq.coilset.coils[pf_name].make_size(max_current)
        problem.eq.coilset.coils[pf_name].fix_size()
        problem.eq.coilset.coils[pf_name].mesh_coil(0.3)

    problem.set_current_bounds(max_currents)


# %%[markdown]

# Now that we've optimised the coil positions for a fixed plasma, we can run the
# Grad-Shafranov solve again to converge the equilibria for the optimised coil positions
# at SOF and EOF.

# %%


program = PicardIterator(
    sof,
    current_opt_problem_sof,
    fixed_coils=True,
    convergence=DudsonConvergence(1e-4),
    relaxation=0.1,
    plot=True,
)
program()

program = PicardIterator(
    eof,
    current_opt_problem_eof,
    fixed_coils=True,
    convergence=DudsonConvergence(1e-4),
    relaxation=0.05,
    plot=True,
)
program()

# %%[markdown]

# Now let's compare the old equilibrium and coilset to the one with optimised positions.

# %%

f, ax = plt.subplots()
x_old, z_old = old_coilset.get_positions()
x_new, z_new = sof.coilset.get_positions()
old_eq.get_LCFS().plot(ax=ax, edgecolor="b", fill=False)
sof.get_LCFS().plot(ax=ax, edgecolor="r", fill=False)
eof.get_LCFS().plot(ax=ax, edgecolor="g", fill=False)
ax.plot(x_old, z_old, linewidth=0, marker="o", color="b")
ax.plot(x_new, z_new, linewidth=0, marker="+", color="r")
isoflux.plot(ax=ax)
plt.show()

# %%[markdown]

# Note that one could converge the Grad-Shafranov equation for each set of coil positions
# but this would be much slower and probably less robust. Personally, I don't think it is
# worthwhile, but were it to succeed it would be fair to say it would be a better
# optimum.

# Let's look at the full result: reference equilibrium, SOF, and EOF

# %%
f, (ax_1, ax_2, ax_3) = plt.subplots(1, 3)

old_eq.plot(ax=ax_1)
old_coilset.plot(ax=ax_1)

sof.plot(ax=ax_2)
sof.coilset.plot(ax=ax_2)
ax_2.set_title("SOF $\\Psi_{b} = $" + f"{sof.get_OX_psis()[1] * 2*np.pi:.2f} V.s")


eof.plot(ax=ax_3)
eof.coilset.plot(ax=ax_3)
ax_3.set_title("EOF $\\Psi_{b} = $" + f"{eof.get_OX_psis()[1] * 2*np.pi:.2f} V.s")
plt.show()
