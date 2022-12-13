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
Fusion reactivity example
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from bluemira.display import plot_defaults
from bluemira.plasma_physics.reactions import reactivity

# %%[markdown]

# Let's plot the reactivity of a couple of well-known fusion reactions.

# %%
plot_defaults()

temperature = np.linspace(0.5, 100, 1000)  # [keV]

sigma_v_DT = reactivity(temperature, "D-T")  # noqa: N816
sigma_v_DT2 = reactivity(temperature, "D-T", method="Johner")  # noqa: N816
sigma_v_DD = reactivity(temperature, "D-D")  # noqa: N816
sigma_v_DHe3 = reactivity(temperature, "D-He3")  # noqa: N816

f, ax = plt.subplots()
ax.loglog(temperature, sigma_v_DT, label="D-T")
ax.loglog(temperature, sigma_v_DT2, label="D-T Johner")
ax.loglog(temperature, sigma_v_DD, label="D-D")
ax.loglog(temperature, sigma_v_DHe3, label="D-He3")

ax.grid(which="both")
ax.set_xlabel("T [keV]")
ax.set_ylabel("$\\sigma_{v}$ [$m^{3}/s$]")
ax.legend()
plt.show()
