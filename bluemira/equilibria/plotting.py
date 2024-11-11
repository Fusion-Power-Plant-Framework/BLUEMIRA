# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Plot utilities for equilibria
"""

from __future__ import annotations

import warnings
from itertools import cycle
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.interpolate import RectBivariateSpline

from bluemira.base.constants import CoilType, raw_uc
from bluemira.base.error import BluemiraError
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.display.plotter import Zorder, plot_coordinates
from bluemira.equilibria.constants import DPI_GIF, J_TOR_MIN, M_PER_MN, PLT_PAUSE
from bluemira.equilibria.diagnostics import EqSubplots, LCFSMask, PsiPlotType
from bluemira.equilibria.find import Xpoint, _in_plasma, get_contours, grid_2d_contour
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.physics import calc_psi
from bluemira.utilities.plot_tools import save_figure, smooth_contour_fill, str_to_latex

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.axes import Axes

    from bluemira.equilibria.diagnostics import EqDiagnosticOptions
    from bluemira.equilibria.equilibrium import (
        Equilibrium,
        FixedPlasmaEquilibrium,
    )

__all__ = [
    "BreakdownPlotter",
    "CoilGroupPlotter",
    "ConstraintPlotter",
    "EquilibriumPlotter",
    "GridPlotter",
    "LimiterPlotter",
    "PlasmaCoilPlotter",
    "RegionPlotter",
    "XZLPlotter",
]

PLOT_DEFAULTS = {
    "psi": {
        "nlevels": 15,
        "cmap": "viridis",
    },
    "field": {
        "nlevels": 15,
        "cmap": "magma",
    },
    "current": {
        "nlevels": 30,
        "cmap": "plasma",
    },
    "separatrix": {
        "color": "r",
        "linewidth": 1.5,
    },
    "opoint": {
        "marker": "o",
        "color": "g",
    },
    "xpoint": {
        "marker": "x",
        "color": "k",
        "linewidth": 1.4,
        "size": 5,
    },
    "grid": {
        "edgewidth": 2,
        "linewidth": 1,
        "color": "k",
    },
    "limiter": {
        "marker": "o",
        "color": "b",
    },
    "coil": {
        "facecolor": {
            "PF": "#0098D4",
            "CS": "#003688",
            "Plasma": "r",
            "NONE": "grey",
        },
        "edgecolor": "k",
        "linewidth": 1,
        "fontsize": 6,
        "alpha": 0.5,
    },
    "contour": {"linewidths": 1.5},
}


class Plotter:
    """
    Utility plotter abstract object
    """

    def __init__(self, ax=None, *, subplots=EqSubplots.XZ, nrows=1, ncols=2, **kwargs):
        for kwarg in kwargs:
            if kwarg not in PLOT_DEFAULTS:
                bluemira_warn(f"Unrecognised plot kwarg: {kwarg}")

        if subplots is EqSubplots.XZ:
            if ax is None:
                self.f, self.ax = plt.subplots()
            else:
                self.ax = ax
            self.ax.set_xlabel("$x$ [m]")
            self.ax.set_ylabel("$z$ [m]")
            self.ax.set_aspect("equal")

        elif subplots is EqSubplots.XZ_COMPONENT_PSI:
            if ax is None:
                self.f, self.ax = plt.subplots(
                    nrows=nrows, ncols=ncols, sharex=True, sharey=True
                )
            else:
                self.ax = ax
            self.ax[0].set_xlabel("$x$ [m]")
            self.ax[0].set_ylabel("$z$ [m]")
            self.ax[0].set_title("Coilset")
            self.ax[0].set_aspect("equal")
            self.ax[1].set_xlabel("$x$ [m]")
            self.ax[1].set_ylabel("$z$ [m]")
            self.ax[1].set_title("Plasma")
            self.ax[1].set_aspect("equal")

        elif subplots is EqSubplots.VS_PSI_NORM:
            if ax is None:
                gs = GridSpec(nrows, ncols)
                self.ax = [plt.subplot(gs[i]) for i in range(nrows * ncols)]
            else:
                self.ax = ax
            for c in range(1, ncols):
                self.ax[(nrows * ncols) - c].set_xlabel("$\\psi_{n}$")

        elif subplots is EqSubplots.VS_X:
            if ax is None:
                gs = GridSpec(nrows, ncols)
                self.ax = [plt.subplot(gs[i]) for i in range(nrows * ncols)]
            else:
                self.ax = ax
            for a in self.ax:
                a.set_xlabel("$x$ [m]")

        else:
            BluemiraError(f"{subplots} is not a valid option for subplots.")


class GridPlotter(Plotter):
    """
    Utility class for plotting Grid objects
    """

    def __init__(self, grid: Grid, ax=None, *, edge: bool = False, **kwargs):
        super().__init__(ax)
        self.grid = grid
        self.plot_grid(**kwargs)
        if edge:
            self.plot_edge(**kwargs)

    def plot_grid(self, **kwargs):
        """
        Plots the gridlines of the grid
        """
        lw = kwargs.get("linewidth", PLOT_DEFAULTS["grid"]["linewidth"])
        color = kwargs.get("color", PLOT_DEFAULTS["grid"]["color"])
        for i in self.grid.x_1d:
            self.ax.plot([i, i], [self.grid.z_min, self.grid.z_max], color, linewidth=lw)
        for i in self.grid.z_1d:
            self.ax.plot([self.grid.x_min, self.grid.x_max], [i, i], color, linewidth=lw)

    def plot_edge(self, **kwargs):
        """
        Plots a thicker boundary edge for the grid
        """
        lw = kwargs.get("edgewidth", PLOT_DEFAULTS["grid"]["edgewidth"])
        color = kwargs.get("color", PLOT_DEFAULTS["grid"]["color"])
        self.ax.plot(*self.grid.bounds, color, linewidth=lw)


class ConstraintPlotter(Plotter):
    """
    Utility class for Constraint plotting.
    """

    def __init__(self, constraint_set, ax=None):
        super().__init__(ax)
        self.constraint_set = constraint_set

        for constraint in self.constraint_set.constraints:
            constraint.plot(self.ax)


class LimiterPlotter(Plotter):
    """
    Utility class for plotting Limiter objects
    """

    def __init__(self, limiter, ax=None, **kwargs):
        super().__init__(ax)
        self.limiter = limiter
        self.plot_limiter(**kwargs)

    def plot_limiter(self, **kwargs):
        """
        Plot the limiter onto the Axes.
        """
        color = kwargs.get("color", PLOT_DEFAULTS["limiter"]["color"])
        marker = kwargs.get("marker", PLOT_DEFAULTS["limiter"]["marker"])
        self.ax.plot(self.limiter.x, self.limiter.z, "s", color=color, marker=marker)


class CoilGroupPlotter(Plotter):
    """
    Utility class for plotting individual Coils (for checking / testing only)

    Parameters
    ----------
    coil: CoilGroup
        The Coil to be plotted
    ax: Matplotlib axis object
        The ax on which to plot the Coil. (plt.gca() default)
    subcoil: bool
        Whether or not to plot the Coil subcoils
    label: bool
        Whether or not to plot labels on the coils
    force: None or np.array((1, 2))
        Whether to plot force vectors, and if so the array of force vectors
    """

    def __init__(
        self, coil, ax=None, *, subcoil=True, label=False, force=None, **kwargs
    ):
        super().__init__(ax)
        self._cg = coil
        self.colors = kwargs.pop("facecolor", None)
        self.linewidth = kwargs.pop(
            "linewidth", PLOT_DEFAULTS["coil"]["linewidth"] + 0.5
        )
        self.edgecolor = kwargs.pop("edgecolor", PLOT_DEFAULTS["coil"]["edgecolor"])
        if "alpha" in kwargs:
            # Alpha can be provided as a list or cycle to other systems, so make sure we
            # support that here.
            alpha = kwargs["alpha"]
            if isinstance(alpha, cycle):
                kwargs["alpha"] = next(alpha)
            if isinstance(alpha, list):
                kwargs["alpha"] = alpha[0]

        self.plot_coil(subcoil=subcoil, label=label, force=force, **kwargs)
        if label:  # Margins and labels fighting
            self.ax.set_xlim(left=-2)
            ymin, ymax = self.ax.get_ylim()
            self.ax.set_ylim(bottom=ymin - 1)
            self.ax.set_ylim(top=ymax + 1)

    def plot_coil(self, subcoil, *, label=False, force=None, **kwargs):
        """
        Plot a coil onto the Axes.
        """
        centre, *arrays = self._plotting_array_shaping()

        if subcoil:
            qb = self._cg._quad_boundary

            if isinstance(qb, tuple):
                qb = [qb]

        if force is not None:
            _d_fx, d_fz = force / M_PER_MN

        for i, (x, z, dx, x_b, z_b, ct, n, cur, ctrl) in enumerate(
            zip(*arrays, strict=False)
        ):
            if ctrl:
                if self.colors is not None:
                    ctype = CoilType(ct.name)
                    if ctype is CoilType.PF:
                        kwargs["facecolor"] = self.colors[0]
                    elif ctype is CoilType.CS:
                        kwargs["facecolor"] = self.colors[1]

                self._plot_coil(
                    x_b,
                    z_b,
                    ct,
                    color=self.edgecolor,
                    linewidth=self.linewidth,
                    **kwargs,
                )
                if subcoil:
                    _qx_b, _qz_b = qb[i]
                    for ind in range(_qx_b.shape[0]):
                        self._plot_coil(_qx_b[ind], _qz_b[ind], ct, fill=False, **kwargs)
                if label:
                    self._annotate_coil(x, z, dx, n, cur, ct, force=force, centre=centre)
                if force is not None:
                    self.ax.arrow(x, z, 0, d_fz[i], color="r", width=0.1)
            else:
                self._plot_coil(x_b, z_b, ct, **kwargs)

    def _plotting_array_shaping(self):
        """
        Shape arrays to account for single coils or groups of coils
        """  # noqa: DOC201
        xx = np.atleast_1d(self._cg.x)
        control_ind = np.zeros_like(xx, dtype=bool)

        if hasattr(self._cg, "_control_ind"):
            control = self._cg._control_ind
            centre = self._get_centre()
        else:
            control = slice(None)
            centre = None

        control_ind[control] = True

        return (
            centre,
            xx,
            np.atleast_1d(self._cg.z),
            np.atleast_1d(self._cg.dx),
            np.atleast_2d(self._cg.x_boundary),
            np.atleast_2d(self._cg.z_boundary),
            np.atleast_1d(self._cg.ctype).tolist(),
            np.atleast_1d(self._cg.name).tolist(),
            np.atleast_1d(self._cg.current),
            control_ind,
        )

    def _get_centre(self):
        """
        Returns
        -------
        :
            A "centre" position for the coils to arrange the labels.
        """
        try:
            x, z = self._cg.get_control_coils().position
            xc = (max(x) + min(x)) / 2
            zc = (max(z) + min(z)) / 2
        except AttributeError:
            # Not a coilset
            return None
        else:
            return xc, zc

    def _annotate_coil(self, x, z, dx, name, current, ctype, force=None, centre=None):
        """
        Single coil annotation utility function
        """
        off = max(0.2, dx + 0.02)
        ctype_name = CoilType(ctype.name)
        if ctype_name is CoilType.CS:
            drs = -1.5 * off
            ha = "right"
        else:
            drs = 2 * off
            ha = "left"
        text = "\n".join([str_to_latex(name), f"{raw_uc(current, 'A', 'MA'):.2f} MA"])
        if force is not None:
            text = "\n".join([text, f"{raw_uc(force[1], 'N', 'MN'):.2f} MN"])
        x = float(x) + drs
        z = float(z)
        if centre is not None and ctype_name is CoilType.PF:
            v = np.array([x - centre[0], z - centre[1]])
            v /= np.sqrt(sum(v**2))
            d = 1 + np.sqrt(2) * dx
            x += d * v[0] - drs * 1.5
            z += d * v[1]
        self.ax.text(
            x,
            z,
            text,
            fontsize=PLOT_DEFAULTS["coil"]["fontsize"],
            ha=ha,
            va="center",
            color="k",
            backgroundcolor="white",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "linewidth": 1,
                "edgecolor": "k",
            },
            zorder=Zorder.TEXT.value,
        )

    def _plot_coil(self, x_boundary, z_boundary, ctype, *, fill=True, **kwargs):
        """
        Single coil plot utility
        """
        mask = kwargs.pop("mask", True)

        fcolor = kwargs.pop("facecolor", PLOT_DEFAULTS["coil"]["facecolor"][ctype.name])

        color = kwargs.pop("edgecolor", PLOT_DEFAULTS["coil"]["edgecolor"])
        linewidth = kwargs.pop("linewidth", PLOT_DEFAULTS["coil"]["linewidth"])
        alpha = kwargs.pop("alpha", PLOT_DEFAULTS["coil"]["alpha"])

        x = np.append(x_boundary, x_boundary[0])
        z = np.append(z_boundary, z_boundary[0])
        if all(x_boundary == x_boundary[0]) or all(z_boundary == z_boundary[0]):
            self.ax.plot(
                x[0], z[0], zorder=Zorder.WIRE.value, color="k", lw=linewidth, marker="+"
            )
        else:
            self.ax.plot(
                x, z, zorder=Zorder.WIRE.value, color=color, linewidth=linewidth
            )

        if fill:
            if mask:
                self.ax.fill(x, z, color="w", zorder=Zorder.FACE.value, alpha=1)

            self.ax.fill(x, z, zorder=Zorder.FACE.value, color=fcolor, alpha=alpha)


class PlasmaCoilPlotter(Plotter):
    """
    Utility class for plotting PlasmaCoils

    Parameters
    ----------
    plasma_coil: PlasmaCoil
        The PlasmaCoil to be plotted
    ax: Matplotlib axis object
        The ax on which to plot the PlasmaCoil. (plt.gca() default)
    """

    def __init__(self, plasma_coil, ax=None, **kwargs):
        super().__init__(ax)
        self.plasma_coil = plasma_coil
        if self.plasma_coil._j_tor is None:
            # No coils to plot
            pass
        else:
            contour = get_contours(
                self.plasma_coil._grid.x,
                self.plasma_coil._grid.z,
                self.plasma_coil._j_tor,
                J_TOR_MIN,
            )
            x, z = contour[0].T
            sq_x, sq_z = grid_2d_contour(x, z)

            nlevels = kwargs.pop("nlevels", PLOT_DEFAULTS["current"]["nlevels"])
            cmap = kwargs.pop("cmap", PLOT_DEFAULTS["current"]["cmap"])

            levels = np.linspace(J_TOR_MIN, np.amax(self.plasma_coil._j_tor), nlevels)
            self.ax.contourf(
                self.plasma_coil._grid.x,
                self.plasma_coil._grid.z,
                self.plasma_coil._j_tor,
                cmap=cmap,
                levels=levels,
            )
            self.ax.plot(sq_x, sq_z, linewidth=1.5, color="k")


class EquilibriumPlotterMixin:
    """
    DRY plotting mixin class.
    """

    eq: Equilibrium | FixedPlasmaEquilibrium
    ax: Axes
    psi: float | npt.NDArray[np.float64]

    def plot_Bp(self, **kwargs):
        """
        Plots the poloidal field onto the Axes.
        """
        nlevels = kwargs.pop("nlevels", PLOT_DEFAULTS["field"]["nlevels"])
        cmap = kwargs.pop("cmap", PLOT_DEFAULTS["field"]["cmap"])

        Bp = self.eq.Bp()
        levels = np.linspace(1e-36, np.amax(Bp), nlevels)
        c = self.ax.contourf(self.eq.x, self.eq.z, Bp, levels=levels, cmap=cmap)
        cbar = plt.colorbar(c)
        cbar.set_label("$B_{p}$ [T]")

    def plot_psi(self, **kwargs):
        """
        Plot flux surfaces
        """
        nlevels = kwargs.pop("nlevels", PLOT_DEFAULTS["psi"]["nlevels"])
        cmap = kwargs.pop("cmap", PLOT_DEFAULTS["psi"]["cmap"])

        levels = np.linspace(np.amin(self.psi), np.amax(self.psi), nlevels)
        self.ax.contour(
            self.eq.x,
            self.eq.z,
            self.psi,
            levels=levels,
            cmap=cmap,
            zorder=Zorder.PSI.value,
            linewidths=PLOT_DEFAULTS["contour"]["linewidths"],
        )

    def plot_plasma_current(self, *, smooth: bool = True, **kwargs):
        """
        Plots flux surfaces inside plasma
        """
        if self.eq._jtor is None:
            return

        nlevels = kwargs.pop("nlevels", PLOT_DEFAULTS["current"]["nlevels"])
        cmap = kwargs.pop("cmap", PLOT_DEFAULTS["current"]["cmap"])

        levels = np.linspace(J_TOR_MIN, np.amax(self.eq._jtor), nlevels)
        cont = self.ax.contourf(
            self.eq.x,
            self.eq.z,
            self.eq._jtor,
            levels=levels,
            cmap=cmap,
            zorder=Zorder.PLASMACURRENT.value,
        )
        if smooth:
            smooth_contour_fill(self.ax, cont, self.eq.get_LCFS())


class FixedPlasmaEquilibriumPlotter(EquilibriumPlotterMixin, Plotter):
    """
    Utility class for FixedPlasmaEquilibrium plotting
    """

    eq: FixedPlasmaEquilibrium

    def __init__(
        self, equilibrium: FixedPlasmaEquilibrium, ax=None, *, field: bool = False
    ):
        super().__init__(ax)
        self.eq = equilibrium
        self.psi = self.eq.psi(self.eq.x, self.eq.z)

        if not field:
            self.plot_plasma_current()
            self.plot_psi()
        else:
            self.plot_Bp()
        self.plot_LCFS()

    def plot_LCFS(self):
        """
        Plot the last closed flux surface
        """
        try:
            lcfs = self.eq.get_LCFS()
        except Exception:  # noqa: BLE001
            bluemira_warn("Unable to plot LCFS")
            return
        x, z = lcfs.xz
        self.ax.plot(
            x,
            z,
            color=PLOT_DEFAULTS["separatrix"]["color"],
            linewidth=PLOT_DEFAULTS["separatrix"]["linewidth"],
            zorder=Zorder.SEPARATRIX.value,
        )


class EquilibriumPlotter(EquilibriumPlotterMixin, Plotter):
    """
    Utility class for Equilibrium plotting
    """

    eq: Equilibrium

    def __init__(
        self,
        equilibrium: Equilibrium,
        ax=None,
        *,
        plasma=False,
        show_ox=True,
        field=False,
    ):
        super().__init__(ax)
        self.eq = equilibrium

        # Do some housework
        self.psi = self.eq.psi()

        self.o_points, self.x_points = self.eq.get_OX_points(self.psi, force_update=True)

        if self.x_points:
            self.xp_psi = self.x_points[0][2]  # Psi at separatrix
        else:
            bluemira_warn(
                "No X-point found in plotted equilibrium. Cannot normalise psi."
            )
            self.xp_psi = np.amax(self.psi)

        if self.o_points:
            self.op_psi = self.o_points[0][2]  # Psi at O-point
        else:
            bluemira_warn(
                "No O-point found in plotted equilibrium. Cannot normalise psi."
            )
            self.op_psi = np.amin(self.psi)

        if not field:
            self.plot_plasma_current()
            self.plot_psi()
        else:
            self.plot_Bp()

        if self.o_points and self.x_points:
            # Only plot if we can normalise psi
            self.plot_separatrix()
            self.plot_flux_surface(1.05, "pink")

        if show_ox:
            self.plot_X_points()
            self.plot_O_points()

        if plasma:
            self.plot_plasma_coil()

    def plot_flux_surface(self, psi_norm, color="k"):
        """
        Plots a normalised flux surface relative to the separatrix with
        increasing values going outwards from plasma core.
        """
        psi = calc_psi(psi_norm, self.op_psi, self.xp_psi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ax.contour(
                self.eq.x,
                self.eq.z,
                self.psi,
                levels=[psi],
                colors=color,
                zorder=Zorder.FLUXSURFACE.value,
                linewidths=PLOT_DEFAULTS["contour"]["linewidths"],
            )

    def plot_separatrix(self):
        """
        Plot the separatrix.
        """
        try:
            separatrix = self.eq.get_separatrix()
        except Exception:  # noqa: BLE001
            bluemira_warn("Unable to plot separatrix")
            return

        coords = separatrix if isinstance(separatrix, list) else [separatrix]

        for coord in coords:
            x, z = coord.xz
            self.ax.plot(
                x,
                z,
                color=PLOT_DEFAULTS["separatrix"]["color"],
                linewidth=PLOT_DEFAULTS["separatrix"]["linewidth"],
                zorder=Zorder.SEPARATRIX.value,
            )

    def plot_X_points(self):  # noqa: N802
        """
        Plot X-points.
        """
        for p in self.x_points:
            if isinstance(p, Xpoint):
                self.ax.plot(
                    p.x,
                    p.z,
                    marker=PLOT_DEFAULTS["xpoint"]["marker"],
                    markersize=PLOT_DEFAULTS["xpoint"]["size"],
                    markeredgewidth=PLOT_DEFAULTS["xpoint"]["linewidth"],
                    color=PLOT_DEFAULTS["xpoint"]["color"],
                    zorder=Zorder.OXPOINT.value,
                )

    def plot_O_points(self):  # noqa: N802
        """
        Plot O-points.
        """
        for p in self.o_points:
            self.ax.plot(
                p.x,
                p.z,
                marker=PLOT_DEFAULTS["opoint"]["marker"],
                color=PLOT_DEFAULTS["opoint"]["color"],
                zorder=Zorder.OXPOINT.value,
            )

    def plot_plasma_coil(self):
        """
        Plot the plasma coil.
        """
        PlasmaCoilPlotter(self.eq.plasma, ax=self.ax)


class EquilibriumComparisonBasePlotter(EquilibriumPlotterMixin, Plotter):
    """
    Utility class for Equilibrium comparison plotting
    """

    def __init__(
        self,
        equilibrium,
        diag_ops: EqDiagnosticOptions,
        ax=None,
    ):
        self.diag_ops = diag_ops
        self.eq = equilibrium
        self.reference_eq = diag_ops.reference_eq

        super().__init__(ax, subplots=self.diag_ops.split_psi_plots)

        self.total_psi = self.eq.psi()
        self.plasma_psi = self.eq.plasma.psi()
        self.coilset_psi = self.eq.coilset.psi(self.eq.x, self.eq.z)
        self.ref_total_psi = self.reference_eq.psi()
        self.ref_plasma_psi = self.reference_eq.plasma.psi()
        if (self.ref_total_psi - self.ref_plasma_psi == 0).all():
            # Fill with zeros if there is no coilset
            self.ref_coilset_psi = 0.0 * self.reference_eq.grid.x
        else:
            self.ref_coilset_psi = self.reference_eq.coilset.psi(
                self.reference_eq.x, self.reference_eq.z
            )

    def plot_reference_LCFS(self, ref_lcfs_label=None):
        """
        Plot the last closed flux surface for the reference equilibria
        """
        try:
            ref_lcfs = self.reference_eq.get_LCFS()
        except Exception:  # noqa: BLE001
            bluemira_warn("Unable to plot reference LCFS")
            return
        x, z = ref_lcfs.xz

        if self.diag_ops.split_psi_plots is EqSubplots.XZ_COMPONENT_PSI:
            for i in range(2):
                self.ax[i].plot(
                    x,
                    z,
                    color="blue",
                    linewidth=PLOT_DEFAULTS["separatrix"]["linewidth"],
                    zorder=9,
                    linestyle="--",
                    label=ref_lcfs_label,
                )
        else:
            self.ax.plot(
                x,
                z,
                color="blue",
                linewidth=PLOT_DEFAULTS["separatrix"]["linewidth"],
                zorder=9,
                linestyle="--",
                label="Reference LCFS",
            )

    def plot_LCFS(self, lcfs_label=None):
        """
        Plot the last closed flux surface
        """
        try:
            lcfs = self.eq.get_LCFS()
        except Exception:  # noqa: BLE001
            bluemira_warn("Unable to plot LCFS")
            return
        x, z = lcfs.xz

        if self.diag_ops.split_psi_plots is EqSubplots.XZ_COMPONENT_PSI:
            for i in range(2):
                self.ax[i].plot(
                    x,
                    z,
                    color=PLOT_DEFAULTS["separatrix"]["color"],
                    linewidth=PLOT_DEFAULTS["separatrix"]["linewidth"],
                    zorder=9,
                    label=lcfs_label,
                )
        else:
            self.ax.plot(
                x,
                z,
                color=PLOT_DEFAULTS["separatrix"]["color"],
                linewidth=PLOT_DEFAULTS["separatrix"]["linewidth"],
                zorder=9,
                label="Current LCFS",
            )

    def plot_psi_coilset(self, grid: Grid = None, **kwargs):
        """
        Plot flux surfaces - coilset contribution
        """
        nlevels = kwargs.pop("nlevels", PLOT_DEFAULTS["psi"]["nlevels"])
        cmap = kwargs.pop("cmap", PLOT_DEFAULTS["psi"]["cmap"])

        if self.coilset_psi is not None:
            levels = np.linspace(
                np.amin(self.coilset_psi), np.amax(self.coilset_psi), nlevels
            )

        if grid is None:
            x, z = self.eq.x, self.eq.z
        else:
            x, z = grid.x, grid.z

        if self.diag_ops.psi_diff in PsiPlotType.PSI_REL_DIFF:
            vmin = 0
            vmax = 1
        else:
            vmin = np.amin(self.total_psi)
            vmax = np.amax(self.total_psi)

        title_type = "Difference "
        if self.diag_ops.psi_diff in PsiPlotType.PSI_REL_DIFF:
            title_type = "Relative difference "
        if self.diag_ops.psi_diff in PsiPlotType.PSI_ABS_DIFF:
            title_type = "Absolute difference "

        if self.diag_ops.psi_diff in PsiPlotType.DIFF:
            if self.coilset_psi is None:
                bluemira_warn(
                    "Coilset_psi all 0s. Will only plot current and reference LCFS"
                )
            else:
                im = self.ax[0].contourf(
                    x,
                    z,
                    self.coilset_psi,
                    levels=levels,
                    cmap=cmap,
                    zorder=8,
                    vmin=vmin,
                    vmax=vmax,
                )
                plt.colorbar(
                    mappable=im, cax=self.cax1, ticks=np.linspace(vmin, vmax, 10)
                )
                plt.suptitle(
                    title_type + "in psi between reference equilibrium"
                    " and current equilibrium, \n split by contribution from"
                    " coilset and plasma"
                )
                plt.tight_layout()

        else:
            self.ax[0].contour(
                x,
                z,
                self.coilset_psi,
                levels=levels,
                cmap=cmap,
                zorder=8,
            )
            plt.suptitle(
                "Psi split by contribution from coilset and plasma for current"
                " equilibrium"
            )
        # Plot current and reference lcfs
        self.plot_LCFS()
        self.plot_reference_LCFS()

    def plot_psi_plasma(self, grid: Grid = None, **kwargs):
        """
        Plot flux surfaces - plasma contribution
        """
        nlevels = kwargs.pop("nlevels", PLOT_DEFAULTS["psi"]["nlevels"])
        cmap = kwargs.pop("cmap", PLOT_DEFAULTS["psi"]["cmap"])

        if self.plasma_psi is not None:
            levels = np.linspace(
                np.amin(self.plasma_psi), np.amax(self.plasma_psi), nlevels
            )

        if grid is None:
            x, z = self.eq.x, self.eq.z
        else:
            x, z = grid.x, grid.z

        if self.diag_ops.psi_diff in PsiPlotType.PSI_REL_DIFF:
            vmin = 0
            vmax = 1
        else:
            vmin = np.amin(self.total_psi)
            vmax = np.amax(self.total_psi)

        if self.diag_ops.psi_diff in PsiPlotType.DIFF:
            if self.plasma_psi is None:
                bluemira_warn(
                    "Plasma_psi all 0s. Will only plot current and reference LCFS"
                )
            else:
                im = self.ax[1].contourf(
                    x,
                    z,
                    self.plasma_psi,
                    levels=levels,
                    cmap=cmap,
                    zorder=8,
                    vmin=vmin,
                    vmax=vmax,
                )
                plt.colorbar(
                    mappable=im, cax=self.cax2, ticks=np.linspace(vmin, vmax, 10)
                )
                plt.tight_layout()

        else:
            self.ax[1].contour(
                x,
                z,
                self.plasma_psi,
                levels=levels,
                cmap=cmap,
                zorder=8,
            )
        # Plot current and reference lcfs
        self.plot_LCFS(lcfs_label="Current LCFS")
        self.plot_reference_LCFS(ref_lcfs_label="Reference LCFS")

    def plot_psi(self, grid: Grid = None, **kwargs):
        """
        Plot flux surfaces
        """
        nlevels = kwargs.pop("nlevels", PLOT_DEFAULTS["psi"]["nlevels"])
        cmap = kwargs.pop("cmap", PLOT_DEFAULTS["psi"]["cmap"])

        if self.total_psi is not None:
            levels = np.linspace(
                np.amin(self.total_psi), np.amax(self.total_psi), nlevels
            )

        if grid is None:
            x, z = self.eq.x, self.eq.z
        else:
            x, z = grid.x, grid.z

        if self.diag_ops.psi_diff in PsiPlotType.PSI_REL_DIFF:
            vmin = 0
            vmax = 1
        else:
            vmin = np.amin(self.total_psi)
            vmax = np.amax(self.total_psi)

        title_type = "Difference "
        if self.diag_ops.psi_diff in PsiPlotType.PSI_REL_DIFF:
            title_type = "Relative difference "
        if self.diag_ops.psi_diff in PsiPlotType.PSI_ABS_DIFF:
            title_type = "Absolute difference "

        if self.diag_ops.psi_diff in PsiPlotType.DIFF:
            if self.total_psi is None:
                bluemira_warn(
                    "Total_psi all 0s. Will only plot current and reference LCFS"
                )
            else:
                im = self.ax.contourf(
                    x,
                    z,
                    self.total_psi,
                    levels=levels,
                    cmap=cmap,
                    zorder=8,
                    vmin=vmin,
                    vmax=vmax,
                )
                plt.colorbar(
                    mappable=im, cax=self.cax, ticks=np.linspace(vmin, vmax, 10)
                )
                plt.suptitle(
                    title_type + "in total psi between reference equilibrium and"
                    " current equilibrium"
                )
                plt.tight_layout()

        else:
            self.ax.contour(x, z, self.total_psi, levels=levels, cmap=cmap, zorder=8)
            plt.title("Total psi for current equilibrium")
        # Plot current and reference lcfs
        self.plot_LCFS()
        self.plot_reference_LCFS()


class EquilibriumComparisonPlotter(EquilibriumComparisonBasePlotter):
    """
    Utility class for Equilibrium plotting and comparing to a reference equilibrium

    Notes
    -----
    If diag_ops.psi_diff is True then plot the relative difference between the current
    plasma psi and the reference plasma psi, as calculated by:
    .. math::
        plasma_psi_diff = |reference_plasma_psi - current_plasma_psi| /
                                max(|reference_plasma_psi - current_plasma_psi|)
    """

    def __init__(
        self,
        equilibrium,
        diag_ops: EqDiagnosticOptions,
        ax=None,
    ):
        super().__init__(equilibrium, diag_ops, ax)

        if np.shape(self.eq.grid.x) != np.shape(self.reference_eq.grid.x):
            bluemira_warn("Reference psi must have same grid size as input equilibria.")

        if np.min(self.eq.grid.x) != np.min(self.reference_eq.grid.x):
            bluemira_warn(
                "The minimum value of x is not the same for the reference equilibrium"
                " and the input equilibrium."
            )

        if np.min(self.eq.grid.z) != np.min(self.reference_eq.grid.z):
            bluemira_warn(
                "The minimum value of z is not the same for the reference equilibrium"
                " and the input equilibrium."
            )

        if np.max(self.eq.grid.x) != np.max(self.reference_eq.grid.x):
            bluemira_warn(
                "The maximum value of x is not the same for the reference equilibrium"
                " and the input equilibrium."
            )

        if np.max(self.eq.grid.z) != np.max(self.reference_eq.grid.z):
            bluemira_warn(
                "The maximum value of z is not the same for the reference equilibrium"
                " and the input equilibrium."
            )

        if self.diag_ops.psi_diff in PsiPlotType.DIFF:
            self._calculate_psi()

        self.i = 0

    def _calculate_psi(self):
        diff_coilset_psi = self.ref_coilset_psi - self.eq.coilset.psi(
            self.eq.x, self.eq.z
        )
        # if all zeros
        if not np.all(diff_coilset_psi):
            self.coilset_psi = None
        elif self.diag_ops.psi_diff in PsiPlotType.PSI_ABS_DIFF:
            self.coilset_psi = np.abs(diff_coilset_psi)
        elif self.diag_ops.psi_diff in PsiPlotType.PSI_REL_DIFF:
            self.coilset_psi = np.abs(diff_coilset_psi) / np.max(
                np.abs(diff_coilset_psi)
            )
        else:
            self.coilset_psi = diff_coilset_psi

        diff_plasma_psi = self.ref_plasma_psi - self.eq.plasma.psi()
        # if all zeros
        if not np.all(diff_plasma_psi):
            self.plasma_psi = None
        elif self.diag_ops.psi_diff in PsiPlotType.PSI_ABS_DIFF:
            self.plasma_psi = np.abs(diff_plasma_psi)
        elif self.diag_ops.psi_diff in PsiPlotType.PSI_REL_DIFF:
            self.plasma_psi = np.abs(diff_plasma_psi) / np.max(np.abs(diff_plasma_psi))
        else:
            self.plasma_psi = diff_plasma_psi

        diff_total_psi = self.ref_total_psi - self.eq.psi()
        # if all zeros
        if not np.all(diff_total_psi):
            self.total_psi = None
        elif self.diag_ops.psi_diff in PsiPlotType.PSI_ABS_DIFF:
            self.total_psi = np.abs(diff_total_psi)
        elif self.diag_ops.psi_diff in PsiPlotType.PSI_REL_DIFF:
            self.total_psi = np.abs(diff_total_psi) / np.max(np.abs(diff_total_psi))
        else:
            self.total_psi = diff_total_psi

    def _clean_plots(self):
        if self.i == 0 and (self.diag_ops.psi_diff in PsiPlotType.DIFF):
            if self.diag_ops.split_psi_plotsis is EqSubplots.XZ_COMPONENT_PSI:
                self.cax1 = make_axes_locatable(self.ax[0]).append_axes(
                    "right", size="5%", pad="2%"
                )
                self.cax2 = make_axes_locatable(self.ax[1]).append_axes(
                    "right", size="5%", pad="2%"
                )
            else:
                self.cax = make_axes_locatable(self.ax).append_axes(
                    "right", size="5%", pad="2%"
                )
        else:
            if self.diag_ops.psi_diff in PsiPlotType.DIFF:
                if self.diag_ops.split_psi_plots is EqSubplots.XZ_COMPONENT_PSI:
                    self.cax1.clear()
                    self.cax2.clear()

                else:
                    self.cax.clear()

            if self.diag_ops.split_psi_plots is EqSubplots.XZ_COMPONENT_PSI:
                for _ax in self.ax:
                    _ax.clear()
                    if legend := _ax.get_legend():
                        legend.remove()
            else:
                self.ax.clear()
                if legend := self.ax.get_legend():
                    legend.remove()

    def update_plot(self):
        self._calculate_psi()
        self._clean_plots()

        # update the plot
        if self.diag_ops.split_psi_plots is EqSubplots.XZ_COMPONENT_PSI:
            self.plot_psi_coilset()
            self.plot_psi_plasma()
            legend = self.ax[0].legend()
            legend.set_zorder(10)
            legend = self.ax[1].legend()
            legend.set_zorder(10)
            self.ax[0].set_xlabel("$x$ [m]")
            self.ax[0].set_ylabel("$z$ [m]")
            self.ax[0].set_title("Coilset")
            self.ax[0].set_aspect("equal")
            self.ax[1].set_xlabel("$x$ [m]")
            self.ax[1].set_ylabel("$z$ [m]")
            self.ax[1].set_title("Plasma")
            self.ax[1].set_aspect("equal")
        else:
            self.plot_psi()
            legend = self.ax.legend()
            legend.set_zorder(10)
            self.ax.set_xlabel("$x$ [m]")
            self.ax.set_ylabel("$z$ [m]")
            self.ax.set_aspect("equal")

        plt.pause(PLT_PAUSE)

        save_figure(
            fig=self.f,
            name=f"{self.diag_ops.plot_name}{self.i}",
            save=self.diag_ops.save,
            folder=self.diag_ops.folder,
            dpi=DPI_GIF,
        )
        self.i += 1


class EquilibriumComparisonPostOptPlotter(EquilibriumComparisonBasePlotter):
    """
    Class for comparing equilibria during post opt. anaylsys.
    Allows for different equilibrium types, grid sizes, etc.,
    to be compared to each other.
    """

    def __init__(
        self,
        equilibrium,
        diag_ops: EqDiagnosticOptions,
        ax=None,
    ):
        super().__init__(equilibrium, diag_ops, ax)

        # Interpolation:
        if (
            (self.reference_eq.grid.x_size != self.eq.grid.x_size)
            or (self.reference_eq.grid.z_size != self.eq.grid.z_size)
            or (self.reference_eq.grid.dx != self.eq.grid.dx)
            or (self.reference_eq.grid.dz != self.eq.grid.dz)
        ):
            self.grid = self.make_comparison_grid()
            self.interpolate_psi_for_comparison()
        else:
            self.grid = self.reference_eq.grid
        self.mask = self.make_lcfs_mask()
        if self.diag_ops.psi_diff in PsiPlotType.DIFF:
            self.calculate_psi_diff()
            if self.diag_ops.split_psi_plots is EqSubplots.XZ_COMPONENT_PSI:
                self.cax1 = make_axes_locatable(self.ax[0]).append_axes(
                    "right", size="5%", pad="2%"
                )
                self.cax2 = make_axes_locatable(self.ax[1]).append_axes(
                    "right", size="5%", pad="2%"
                )
            else:
                self.cax = make_axes_locatable(self.ax).append_axes(
                    "right", size="5%", pad="2%"
                )

    def make_comparison_grid(self):
        """
        If the grids are different, make a new one to interpolate over.
        """
        x_min = np.max([self.reference_eq.grid.x_min, self.eq.grid.x_min])
        x_max = np.min([self.reference_eq.grid.x_max, self.eq.grid.x_max])
        z_min = np.max([self.reference_eq.grid.z_min, self.eq.grid.z_min])
        z_max = np.min([self.reference_eq.grid.z_max, self.eq.grid.z_max])
        nx = np.min([self.reference_eq.grid.nx, self.eq.grid.nx])
        nz = np.min([self.reference_eq.grid.nz, self.eq.grid.nz])
        return Grid(x_min, x_max, z_min, z_max, nx, nz)

    def interpolate_psi(self, psi, psi_grid):
        """Interpolate psi over new comparision grid"""
        psi_func = RectBivariateSpline(psi_grid.x[:, 0], psi_grid.z[0, :], psi)
        return psi_func.ev(self.grid.x, self.grid.z)

    def interpolate_psi_for_comparison(self):
        """Interpolate all psi components over new grid."""
        self.ref_coilset_psi = self.interpolate_psi(
            self.ref_coilset_psi, self.reference_eq.grid
        )
        self.ref_plasma_psi = self.interpolate_psi(
            self.ref_plasma_psi, self.reference_eq.grid
        )
        self.ref_total_psi = self.interpolate_psi(
            self.ref_total_psi, self.reference_eq.grid
        )
        self.coilset_psi = self.interpolate_psi(self.coilset_psi, self.eq.grid)
        self.plasma_psi = self.interpolate_psi(self.plasma_psi, self.eq.grid)
        self.total_psi = self.interpolate_psi(self.total_psi, self.eq.grid)

    def make_lcfs_mask(self):
        """
        Make a LCFS shaped mask to use with equilibria comparisions.

        Returns
        -------
        mask:
            A mask array to be used in plotting. 1 inside the LCFS, 0 outside.

        """
        mask_matx = np.zeros_like(self.grid.x)
        try:
            ref_lcfs = self.reference_eq.get_LCFS()
        except Exception:  # noqa: BLE001
            bluemira_warn("Unable to find reference LCFS")
            return None
        return _in_plasma(
            self.grid.x, self.grid.z, mask_matx, ref_lcfs.xz.T, include_edges=False
        )

    def apply_mask(self, mask_type):
        """Apply mask to psi from equilibrium and reference equilibrium."""
        if mask_type is LCFSMask.IN:
            self.coilset_psi *= self.mask
            self.plasma_psi *= self.mask
            self.total_psi *= self.mask
        elif mask_type is LCFSMask.OUT:
            self.coilset_psi *= abs(self.mask - 1)
            self.plasma_psi *= abs(self.mask - 1)
            self.total_psi *= abs(self.mask - 1)

    def calculate_psi_diff(self):
        """
        Find the difference betwwen the reference and choisen equilibrium psi values.
        """
        diff_coilset_psi = self.ref_coilset_psi - self.coilset_psi
        # if all zeros
        if not np.all(diff_coilset_psi):
            self.coilset_psi = None
        elif self.diag_ops.psi_diff in PsiPlotType.PSI_ABS_DIFF:
            self.coilset_psi = np.abs(diff_coilset_psi)
        elif self.diag_ops.psi_diff in PsiPlotType.PSI_REL_DIFF:
            self.coilset_psi = np.abs(diff_coilset_psi) / np.max(
                np.abs(diff_coilset_psi)
            )
        else:
            self.coilset_psi = diff_coilset_psi

        diff_plasma_psi = self.ref_plasma_psi - self.plasma_psi
        # if all zeros
        if not np.all(diff_plasma_psi):
            self.plasma_psi = None
        elif self.diag_ops.psi_diff in PsiPlotType.PSI_ABS_DIFF:
            self.plasma_psi = np.abs(diff_plasma_psi)
        elif self.diag_ops.psi_diff in PsiPlotType.PSI_REL_DIFF:
            self.plasma_psi = np.abs(diff_plasma_psi) / np.max(np.abs(diff_plasma_psi))
        else:
            self.plasma_psi = diff_plasma_psi

        diff_total_psi = self.ref_total_psi - self.total_psi
        # if all zeros
        if not np.all(diff_total_psi):
            self.total_psi = None
        elif self.diag_ops.psi_diff in PsiPlotType.PSI_ABS_DIFF:
            self.total_psi = np.abs(diff_total_psi)
        elif self.diag_ops.psi_diff in PsiPlotType.PSI_REL_DIFF:
            self.total_psi = np.abs(diff_total_psi) / np.max(np.abs(diff_total_psi))
        else:
            self.total_psi = diff_total_psi

    def plot_compare_psi(self):
        """FIXME"""
        # Apply mask
        if self.diag_ops.lcfs_mask is not None:
            self.apply_mask(self.diag_ops.lcfs_mask)

        if self.diag_ops.split_psi_plots is EqSubplots.XZ_COMPONENT_PSI:
            self.plot_psi_coilset(self.grid)
            self.plot_psi_plasma(self.grid)
            legend = self.ax[0].legend()
            legend.set_zorder(10)
            legend = self.ax[1].legend()
            legend.set_zorder(10)
            self.ax[0].set_xlabel("$x$ [m]")
            self.ax[0].set_ylabel("$z$ [m]")
            self.ax[0].set_title("Coilset")
            self.ax[0].set_aspect("equal")
            self.ax[1].set_xlabel("$x$ [m]")
            self.ax[1].set_ylabel("$z$ [m]")
            self.ax[1].set_title("Plasma")
            self.ax[1].set_aspect("equal")
        else:
            self.plot_psi(self.grid)
            legend = self.ax.legend()
            legend.set_zorder(10)
            self.ax.set_xlabel("$x$ [m]")
            self.ax.set_ylabel("$z$ [m]")
            self.ax.set_aspect("equal")


class BreakdownPlotter(Plotter):
    """
    Utility class for Breakdown plotting
    """

    def __init__(self, breakdown, ax=None, *, Bp=False, B_breakdown=0.003):
        super().__init__(ax)
        self.bd = breakdown

        self.psi = self.bd.psi()
        self.psi_bd = self.bd.breakdown_psi
        self.Bp = self.bd.Bp(self.bd.x, self.bd.z)

        self.plot_contour()
        self.plot_zone(B_breakdown)
        if Bp:
            self.plot_Bp()

    def plot_contour(self):
        """
        Plot flux surfaces.
        """
        levels = np.linspace(self.psi_bd - 0.1, self.psi_bd, 3)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ax.contour(
                self.bd.x,
                self.bd.z,
                self.psi,
                levels=levels,
                colors="r",
                linewidths=PLOT_DEFAULTS["contour"]["linewidths"],
            )

    def plot_Bp(self, **kwargs):
        """
        Plots the poloidal field onto the Axes.
        """
        nlevels = kwargs.pop("nlevels", PLOT_DEFAULTS["field"]["nlevels"])
        cmap = kwargs.pop("cmap", PLOT_DEFAULTS["field"]["cmap"])
        levels = np.linspace(1e-36, np.amax(self.Bp), nlevels)
        c = self.ax.contourf(self.bd.x, self.bd.z, self.Bp, levels=levels, cmap=cmap)
        cbar = plt.colorbar(c)
        cbar.set_label("$B_{p}$ [T]")

    def plot_zone(self, field):
        """
        Plot the low field zones with a dashed line.
        """
        colors = ["b"]
        self.ax.contour(
            self.bd.x,
            self.bd.z,
            self.Bp,
            levels=[field],
            colors=colors,
            linestyles="dashed",
            linewidths=PLOT_DEFAULTS["contour"]["linewidths"],
        )

        if self.psi_bd is not None:
            self.ax.set_title("$\\psi_{b}$ = " + f"{2 * np.pi * self.psi_bd:.2f} V.s")


class XZLPlotter(Plotter):
    """
    Utility class for plotting L constraints
    """

    def __init__(self, xzl_mapper, ax=None):
        super().__init__(ax)
        self.xzl = xzl_mapper

        for coords in self.xzl.excl_zones:
            plot_coordinates(
                coords, self.ax, fill=True, alpha=0.2, facecolor="r", edgecolor="r"
            )

        for coords in self.xzl.excl_loops:
            plot_coordinates(
                coords,
                self.ax,
                fill=False,
                edgecolor="r",
                zorder=Zorder.POSITION_1D.value,
                linestyle="--",
            )

        for coords in self.xzl.incl_loops:
            plot_coordinates(
                coords,
                self.ax,
                fill=False,
                edgecolor="k",
                zorder=Zorder.POSITION_1D.value,
                linestyle="--",
            )


class RegionPlotter(Plotter):
    """
    Utility class for plotting 2-D L constraints
    """

    def __init__(self, region_mapper, ax=None):
        super().__init__(ax)
        self.rmp = region_mapper

        for intpltr in self.rmp.regions.values():
            plot_coordinates(
                intpltr.coords,
                self.ax,
                fill=True,
                alpha=0.2,
                zorder=Zorder.POSITION_2D.value,
                facecolor="g",
                edgecolor="g",
            )


class CorePlotter(Plotter):
    """
    Utility class for plotting equilibrium normalised radius characteristic
    profiles.
    """

    def __init__(self, results, ax=None, eq_name=None):
        num_plots = len(results.__dict__)
        r, c = int((num_plots - 1) / 2) + 1, 2
        super().__init__(ax, subplots=EqSubplots.VS_PSI_NORM, nrows=r, ncols=c)
        self.plot_core(results, eq_name)
        for a in ax[num_plots:]:
            a.axis("off")

    def plot_core(self, results, eq_name=None):
        if eq_name is None:
            ccycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
            for i, (k, v) in enumerate(results.__dict__.items()):
                color = next(ccycle)
                self.ax[i].plot(results.psi_n, v, label=str_to_latex(k), color=color)
                self.ax[i].legend()
        else:
            for i, (k, v) in enumerate(results.__dict__.items()):
                self.ax[i].plot(results.psi_n, v)
                self.ax[i].set_ylabel(str_to_latex(k))
            self.ax[0].legend()
            self.ax[0].set_label(eq_name)


class CorePlotter2(Plotter):
    """
    Utility class for plotting plasma equilibrium cross-core profiles.
    """

    def __init__(self, eq, ax=None, n=50):
        super().__init__(ax, subplots=EqSubplots.VS_X, nrows=3, ncols=1)
        self.plot_core2(eq, n)

    def plot_core2(self, eq, n=50):
        """
        Plot the plasma equilibrium cross-core profiles.
        """
        jfunc = RectBivariateSpline(eq.x[:, 0], eq.z[0, :], eq._jtor)
        p = eq.pressure_map()
        pfunc = RectBivariateSpline(eq.x[:, 0], eq.z[0, :], p)
        o_points, _ = eq.get_OX_points()
        xmag, zmag = o_points[0].x, o_points[0].z
        _psia, psib = eq.get_OX_psis()
        n = 50
        xx = np.linspace(eq.grid.x_min, eq.grid.x_max, n)
        zz = np.linspace(zmag, zmag, n)
        psi = eq.psi(xx, zz) * 2 * np.pi
        ccycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        self.ax[0].plot(xx, pfunc(xx, zz, grid=False), color=next(ccycle))
        self.ax[0].annotate("$p$", xy=[0.05, 0.8], xycoords="axes fraction")
        self.ax[0].set_ylabel("[Pa]")
        self.ax[1].plot(xx, jfunc(xx, zz, grid=False), color=next(ccycle))
        self.ax[1].set_ylabel("[A/m^2]")
        self.ax[1].annotate("$J_{\\phi}$", xy=[0.05, 0.8], xycoords="axes fraction")
        self.ax[2].plot(xx, psi, color=next(ccycle))
        self.ax[2].set_ylabel("[V.s]")
        self.ax[2].annotate("$\\psi$", xy=[0.05, 0.8], xycoords="axes fraction")
        self.ax[2].axhline(psib * 2 * np.pi, color="r", linestyle="--")
        for ax in self.ax:
            ax.axvline(xmag, color="r")


class ProfilePlotter(Plotter):
    """
    Utility class for plotting profile objects
    """

    def __init__(self, profiles, ax=None):
        super().__init__(ax)
        self.prof = profiles
        self.plot_profiles()

    def plot_profiles(self, n=50):
        """
        Plot the plasma profiles.
        """
        x = np.linspace(0, 1, n)
        self.ax.plot(x, self.prof.shape(x), label="shape function")
        self.ax.plot(x, self.prof.fRBpol(x) / max(self.prof.fRBpol(x)), label="fRBpol")
        self.ax.plot(
            x, self.prof.ffprime(x) / max(abs(self.prof.ffprime(x))), label="FFprime"
        )
        self.ax.plot(
            x, self.prof.pprime(x) / max(abs(self.prof.pprime(x))), label="pprime"
        )
        self.ax.plot(
            x, self.prof.pressure(x) / max(abs(self.prof.pressure(x))), label="pressure"
        )
        self.ax.legend()
