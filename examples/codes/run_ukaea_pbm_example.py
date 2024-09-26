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
Test for power balance run
"""

# %%
import bluemira.codes.ukaea_powerbalance as power_balance
from bluemira.base.config import Configuration
from bluemira.base.logs import set_log_level
from bluemira.codes.ukaea_powerbalance.constants import MODEL_NAME

set_log_level("DEBUG", logger_names=["PowerBalance"])

# %%[markdown]
# # Configure Power Balance Models (UKAEA)
# Currently BLUEMIRA does not contain any parameters usable by Power Balance
# instead these are read from PROCESS

# %%
param_list = {}
params = Configuration(param_list)

# %%[markdown]
# Some values are not linked into bluemira. These power balance parameters can be set
# directly in `problem_settings`.

# %%
problem_settings = {
    "currdrive_eff_model": 8,
    "structural.Magnets.isMagnetTFSuperconCoil": 2,
    f"{MODEL_NAME}.CryogenicPower.CD.FBCol_2": 0.00041,
    f"{MODEL_NAME}.wasteheatpower.wasteHeatCryo.Height": 3.2,
    f"{MODEL_NAME}.magnetpower.magnetTF.numCoils": 20,
}

# %%[markdown]
# Finally the `build_config` dictionary collates the configuration settings for
# the solver.

# %%
build_config = {"problem_settings": problem_settings, "mode": "run"}

# %%[markdown]
# Now we can create the solver object with the parameters and build configuration

# %%
pbm_solver = power_balance.Solver(params=params, build_config=build_config)

# %%[markdown]
# ### Running the solver
# Very simply use the `run` method of the solver
pbm_solver.run()
