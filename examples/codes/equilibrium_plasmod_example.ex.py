# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
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
An example that shows how to set up a fixed boundary equilibrium problem.
"""

# %% [markdown]
# # 1.5-D transport + fixed boundary equilibrium <-> a 2-D fixed boundary equilibrium
#
# ## Introduction
#
# In this example, we will show how to couple PLASMOD (1.5-D current diffusion
# and fixed boundary equilibrium solver with an up-down symmetric plasma boundary)
# to our 2-D finite element fixed boundary equilibrium solver with an up-down
# asymmetric boundary and X-point.

# This procedure is known to not be particularly robust, please use with caution.
#
# ## Imports
#
# Import necessary module definitions.

# %%
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import get_bluemira_path, get_bluemira_root
from bluemira.base.logs import set_log_level
from bluemira.codes import transport_code_solver
from bluemira.codes.plasmod.equilibrium_2d_coupling import solve_transport_fixed_boundary
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemGradShafranovFixedBoundary,
)
from bluemira.equilibria.fem_fixed_boundary.file import save_fixed_boundary_to_file
from bluemira.equilibria.shapes import JohnerLCFS

set_log_level("NOTSET")

# %% [markdown]
# In this example a fixed boundary equilibrium problem is solved using PLASMOD as
# the transport solver.

# We're going to use the following parameters

# %%

A = 3.1  # Aspect ratio
R_0 = 8.983  # Major radius
a_minor = R_0 / A  # Minor radius
I_p = 19e6  # Plasma current
B_0 = 5.31  # Toroidal field at major radius
kappa_95 = 1.652  # 95th percentile flux surface elongation
delta_95 = 0.333  # 95th percentile flux surface triangularity
q_95 = 3.25  # 95th percentile flux surface safety factor


# %% [markdown]
# Fixed Boundary Equilibrium
# Setup the Plasma shape parameterisation variables. A Johner parameterisation is used.

# %%
johner_parameterisation = JohnerLCFS(
    {
        "r_0": {"value": R_0},
        "a": {"value": a_minor},
        "kappa_u": {"value": 1.6},
        "kappa_l": {"value": 1.75},
        "delta_u": {"value": 0.33},
        "delta_l": {"value": 0.45},
    }
)

# %% [markdown]
# Initialise the transport solver (in this case PLASMOD is used)
#
# Note: it is necessary to manually ensure consistency between transport solver
# and plasma parameters (as for R_0, A, etc.). In particular, since PLASMOD
# is using a symmetric plasma, delta and kappa are set up as the average of
# plasma's kappa_u and kappa_l and delta_u and delta_l, respectively.

# %%
if plasmod_binary := shutil.which("plasmod"):
    PLASMOD_PATH = Path(plasmod_binary).parent
else:
    PLASMOD_PATH = Path(Path(get_bluemira_root()).parent, "plasmod/bin")
binary = Path(PLASMOD_PATH, "plasmod").as_posix()


source = "Plasmod Example"
plasmod_params = {
    "A": {"value": A, "unit": "", "source": source},
    "R_0": {
        "value": R_0,
        "unit": "m",
        "source": source,
    },
    "I_p": {"value": I_p, "unit": "A", "source": source},
    "B_0": {"value": B_0, "unit": "T", "source": source},
    "V_p": {"value": -2500, "unit": "m^3", "source": source},
    "v_burn": {"value": -1.0e6, "unit": "V", "source": source},
    "kappa_95": {"value": kappa_95, "unit": "", "source": source},
    "delta_95": {"value": delta_95, "unit": "", "source": source},
    "delta": {
        "value": (
            johner_parameterisation.variables.delta_l
            + johner_parameterisation.variables.delta_u
        )
        / 2,
        "unit": "",
        "source": source,
    },
    "kappa": {
        "value": (
            johner_parameterisation.variables.kappa_l
            + johner_parameterisation.variables.kappa_u
        )
        / 2,
        "unit": "",
        "source": source,
    },
    "q_95": {"value": q_95, "unit": "", "source": source},
    "f_ni": {"value": 0, "unit": "", "source": source},
}

problem_settings = {
    "amin": a_minor,
    "pfus_req": 1800.0,
    "pheat_max": 100.0,
    "q_control": 50.0,
    "i_impmodel": "PED_FIXED",
    "i_modeltype": "GYROBOHM_2",
    "i_equiltype": "q95_sawtooth",
    "i_pedestal": "SAARELMA",
    "isiccir": "EICH_FIT",
    "isawt": "SAWTEETH",
}

plasmod_build_config = {
    "problem_settings": problem_settings,
    "mode": "run",
    "binary": binary,
    "directory": get_bluemira_path("", subfolder="generated_data"),
}

plasmod_solver = transport_code_solver(
    params=plasmod_params,
    build_config=plasmod_build_config,
    module="PLASMOD",
)

# %% [markdown]
# Initialise the FEM problem

# %%
fem_GS_fixed_boundary = FemGradShafranovFixedBoundary(
    p_order=2,
    max_iter=30,
    iter_err_max=1e-3,
)

# %% [markdown]
# Solve the fixed boundary problem. Set plot = True if you want to check the
# solution at each iteration.

# NOTE: The procedure is known to be sensitive to low mesh sizes. There is
# a TODO on this, see issue #2140.

# %%
equilibrium = solve_transport_fixed_boundary(
    johner_parameterisation,
    plasmod_solver,
    fem_GS_fixed_boundary,
    kappa95_t=kappa_95,  # Target kappa_95
    delta95_t=delta_95,  # Target delta_95
    lcar_mesh=0.2,  # Best to not go lower than this!
    max_iter=15,
    iter_err_max=1e-3,
    max_inner_iter=20,
    inner_iter_err_max=1e-3,
    relaxation=0.0,
    plot=False,
    debug=False,
    gif=False,
    refine=True,
    num_levels=2,
    distance=1.0,
)

# %% [markdown]
# Save to a file

# %%
data = save_fixed_boundary_to_file(
    Path(
        get_bluemira_path("", subfolder="generated_data"), "fixed_boundary_data.json"
    ).as_posix(),
    "equilibrium_example",
    equilibrium,
    100,
    110,
    file_format="json",
)

# %% [markdown]
# Inspect the final converged equilibrium

# %%
xx, zz = np.meshgrid(data.x, data.z, indexing="ij")
f, ax = plt.subplots()
ax.contour(xx, zz, data.psi)
ax.plot(data.xbdry, data.zbdry)
ax.set_aspect("equal")

f, ax = plt.subplots(2, 2)
ax[0, 0].plot(data.psinorm, data.pprime, label="p'")
ax[0, 1].plot(data.psinorm, data.ffprime, label="FF'")
ax[1, 1].plot(data.psinorm, data.fpol, label="F")
ax[1, 0].plot(data.psinorm, data.pressure, label="p")
for axi in ax.flat:
    axi.legend()

plt.show()
