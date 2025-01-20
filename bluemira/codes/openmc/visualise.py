# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Functions to visualise the csg geometry being described by the openmc objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import openmc
import openmc.region

if TYPE_CHECKING:
    from collections.abc import Iterable

TOP, BOTTOM = 1000.0, -1000.0
LEFT, RIGHT = -1000.0, 1000.0


def plot_surfaces(
    surfaces_list: Iterable[openmc.Surface], *, ax=None, plot_both_sides: bool = False
) -> plt.Axes:
    """
    Plot a list of surfaces in matplotlib.

    Parameters
    ----------
    surface_list:
        list of openmc.Surface that we are trying to plot.
    ax:
        The matplotlib axes object that we are trying to plot onto.
    plot_both_sides:
        Whether to plot from x=-1000 to x=1000cm, or from x=0 to x=1000cm.
        Due to model being axis-symmetric we expect the plot to have a reflective
        symmetry line of x=0.

    Returns
    -------
    ax:
        The matplotlib axes object on which the surfaces' cross-sections are plotted
        onto.
    """
    ax = ax or plt.subplot()
    # ax.set_aspect(1.0) # don't do this as it makes the plot hard to read.
    for i, surfaces in enumerate(surfaces_list):
        surface_tuple = (surfaces,) if isinstance(surfaces, openmc.Surface) else surfaces
        for sf in surface_tuple:
            plot_surface_at_1000cm(ax, sf, color_num=i, plot_both_sides=plot_both_sides)
    ax.legend()
    ax.set_aspect("equal")
    ax.set_ylim([BOTTOM, TOP])
    if plot_both_sides:
        ax.set_xlim([LEFT, RIGHT])
    else:
        ax.set_xlim([0, RIGHT])
    ax.set_aspect(1.0)
    return ax


def plot_surface_at_1000cm(
    ax: plt.Axes, surface: openmc.Surface, color_num: int, *, plot_both_sides: bool
) -> None:
    """
    In the range [-1000, 1000], plot the RZ cross-section of the ZCylinder/ZPlane/ZCone.

    Parameters
    ----------
    ax:
        The axes object on which the surface shall be drawn.
    surface:
        The surface to be drawn.
    color_num:
        The C? number that is parsed onto matplotlib, to specify what colour should be
        used to draw the line representing the surface.
    plot_both_sides:
        If it is a ZCylinder, then its RZ cross-section is symmetric along the x=0 line.
        Therefore if we only want to look at the RHHP (which is going to be a mirrored
        copy of the LHHP) there is no need to plot the LHS line (the part of the
        ZCylinder intersecting the x<0 LHHP). So when plot_both_sides is false, we skip
        plotting the LHS line.
    """
    label_str = f"{surface.id}: {surface.name}"
    color = f"C{color_num}"
    if isinstance(surface, openmc.ZCylinder):
        ax.plot(
            [surface.r, surface.r],
            [BOTTOM, TOP],
            label=(label_str + " (RHHP)") if plot_both_sides else label_str,
            color=color,
        )
        if plot_both_sides:
            ax.plot(
                [-surface.r, -surface.r],
                [BOTTOM, TOP],
                label=label_str + " (LHHP)",
                color=color,
                linestyle="-.",
            )
    elif isinstance(surface, openmc.ZPlane):
        ax.plot(
            [LEFT, RIGHT],
            [surface.z0, surface.z0],
            label=label_str,
            color=color,
        )
    elif isinstance(surface, openmc.ZCone):
        intercept = surface.z0
        slope = 1 / np.sqrt(surface.r2)

        def equation_pos(x):
            return slope * np.array(x) + intercept

        def equation_neg(x):
            return -slope * np.array(x) + intercept

        y_pos, y_neg = equation_pos([LEFT, RIGHT]), equation_neg([LEFT, RIGHT])
        ax.plot(
            [LEFT, RIGHT],
            y_pos,
            label=label_str + " (upper)",
            linestyle=":",
            color=color,
        )
        ax.plot(
            [LEFT, RIGHT],
            y_neg,
            label=label_str + " (lower)",
            linestyle="--",
            color=color,
        )
    elif isinstance(surface, openmc.ZTorus):
        ax.add_patch(
            plt.Circle(
                (surface.a, surface.z0),
                surface.b,
                color=color,
                fill=False,
                label=(label_str + " (RHHP)") if plot_both_sides else label_str,
                linewidth=2.54,
            )
        )
        if plot_both_sides:
            ax.add_patch(
                plt.Circle(
                    (-surface.a, surface.z0),
                    surface.b,
                    color=color,
                    fill=False,
                    label=label_str + " (LHHP)",
                    linestyle="-.",
                    linewidth=2.54,
                )
            )


def plot_regions(surfaces_list: list[openmc.Region], *, ax=None) -> plt.Axes: ...


def _patch_circle(torus: openmc.ZTorus):
    pass


def create_region_patch_up_to_1000cm(
    ax: plt.Axes,
    region: openmc.Region,
    color_num: int,
    *,
    plot_both_sides: bool,
    alpha: float = 0.4,
) -> None:
    """
    Plot the openmc.Region selected by the expression.

    Parameters
    ----------
    ax:
        The axes object on which the regions shall be plotted
    region:
        region to be plotted
    color_num:
        The C? number that is parsed onto matplotlib, to specify what colour should be
        used to draw the line representing the surface.

    Raises
    ------
    NotImplementedError
        We can only accept ZCylinder, ZPlane, ZCone, and halfcones

    Returns
    -------
    :
        The axes on which the region is plotted
    """
    tr, br = np.array([[RIGHT, TOP], [RIGHT, BOTTOM]])
    tl, bl = (
        np.array([[LEFT, TOP], [LEFT, BOTTOM]])
        if plot_both_sides
        else np.array([[0, TOP], [0, BOTTOM]])
    )

    if isinstance(openmc.Halfspace):
        if (
            isinstance(region.surface, openmc.ZCylinder)
            or isinstance(region.surface, openmc.ZPlane)
            or isinstance(region.surface, openmc.ZCone)
        ):
            ax.add_patch(plt.Polygon())
        elif isinstance(region.surface, openmc.ZTorus):
            ax.add_patch(_patch_circle(region.surface))
        else:
            raise NotImplementedError(
                f"type{region.surface} is not one of the accepted surfaces!"
            )
    elif isinstance(openmc.Intersection):
        surfaces = list(region.get_surfaces())
        if len(surfaces) != 2:
            raise NotImplementedError(
                f"Expected intersection of a cone with a ZPlane, instead got {surfaces}!"
            )
        ax.add_patch(plt.Polygon())
    else:
        raise NotImplementedError(f"This function cannot plot type {type(region)}!")
    return ax
