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
Example single null first wall particle heat flux
"""

# %%
import os

import matplotlib.pyplot as plt

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria import Equilibrium
from bluemira.geometry.coordinates import Coordinates
from bluemira.radiation_transport.advective_transport import ChargedParticleSolver

# %%[markdown]

# First we load an up equilibrium

# %%
read_path = get_bluemira_path("equilibria", subfolder="data")
eq_name = "EU-DEMO_EOF.json"
eq_name = os.path.join(read_path, eq_name)
eq = Equilibrium.from_eqdsk(eq_name)

# %%[markdown]

# Now we load a first wall geometry, so that the solver can determine where the flux
# surfaces intersect the first wall.

# %%
read_path = get_bluemira_path("radiation_transport/test_data", subfolder="tests")
fw_name = "first_wall.json"
fw_name = os.path.join(read_path, fw_name)
fw_shape = Coordinates.from_json(fw_name)

# %%[markdown]

# Then we define some input `Parameter`s for the solver.

# %%

params = {
    "P_sep_particle": 150,
    "f_p_sol_near": 0.50,
    "fw_lambda_q_near_omp": 0.01,
    "fw_lambda_q_far_omp": 0.05,
    "f_lfs_lower_target": 0.75,
    "f_hfs_lower_target": 0.25,
    "f_lfs_upper_target": 0,
    "f_hfs_upper_target": 0,
}

# %%[markdown]

# Finally, we initialise the `ChargedParticleSolver` and run it.

# %%
solver = ChargedParticleSolver(params, eq, dx_mp=0.001)
x, z, hf = solver.analyse(first_wall=fw_shape)

# Plot the analysis
solver.plot()
plt.show()
