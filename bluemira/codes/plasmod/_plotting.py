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
Plotting for PLASMOD.
"""

import matplotlib.pyplot as plt

from bluemira.display import plot_defaults


def plot_default_profiles(solver, show=True):
    """
    Plot a default set of profiles from a PLASMOD solver.

    Parameters
    ----------
    solver: plasmod.Solver
        Solver for which to plot profiles
    show: bool
        Whether or not to show the plot

    Returns
    -------
    f: Figure
        Matplotlib figure
    ax: np.ndarray[Axes]
        Array of matplotlib Axes
    """
    from bluemira.codes.plasmod.mapping import Profiles

    plot_defaults()

    f, ax = plt.subplots(2, 3)
    rho = solver.get_profile(Profiles.x)

    # Temperature profiles
    ti = solver.get_profile(Profiles.Ti)
    te = solver.get_profile(Profiles.Te)
    ax[0, 0].plot(rho, ti, label="$T_{i}$")
    ax[0, 0].plot(rho, te, label="$T_{e}$")
    ax[0, 0].set_ylabel("Temperature [keV]")

    # Current profiles
    jpar = solver.get_profile(Profiles.jpar)
    jbs = solver.get_profile(Profiles.jbs)
    jcd = solver.get_profile(Profiles.jcd)
    ax[0, 1].plot(rho, jpar, label="$j_{||}$")
    ax[0, 1].plot(rho, jbs, label="$j_{BS}$")
    ax[0, 1].plot(rho, jcd, label="$j_{CD}$")
    ax[0, 1].set_ylabel("Current density [A/m²]")

    # Density profiles
    ni = solver.get_profile(Profiles.nions)
    ne = solver.get_profile(Profiles.ne)
    ax[1, 0].plot(rho, ni, label="$n_{i}$")
    ax[1, 0].plot(rho, ne, label="$n_{e}$")
    ax[1, 0].set_ylabel("Density [10¹⁹/m3]")

    # q profile
    qprof = solver.get_profile(Profiles.qprof)
    ax[1, 1].plot(rho, qprof, label="$q$")
    ax[1, 1].set_ylabel("Safety factor")

    # Flux functions
    pprime = solver.get_profile(Profiles.pprime)
    ax[0, 2].plot(rho, pprime, label="p'")
    ax[0, 2].set_ylabel("[Pa/Wb]")
    axi: plt.Axes = ax[0, 2]
    axi.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    ffprime = solver.get_profile(Profiles.ffprime)
    ax[1, 2].plot(rho, ffprime, label="FF'")
    ax[1, 2].set_ylabel("[T]")

    for axe in ax.flat:
        axe.grid()
        axe.set_xlabel("$\\rho$")
        axe.set_xlim([0.0, 1.0])
        axe.legend(loc="best")

    if show:
        plt.show()
    return f, ax
