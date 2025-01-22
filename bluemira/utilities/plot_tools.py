# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
A collection of plotting tools.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as Path_mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import PathPatch3D

import bluemira.display.error as bm_display_error
from bluemira.base.constants import GREEK_ALPHABET, GREEK_ALPHABET_CAPS
from bluemira.base.file import get_bluemira_path, try_get_bluemira_path
from bluemira.geometry.coordinates import check_ccw, rotation_matrix_v1v2
from bluemira.geometry.placement import BluemiraPlacement

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.contour import ContourSet

    from bluemira.base.components import Component
    from bluemira.geometry.coordinates import Coordinates

__all__ = [
    "BluemiraPathPatch3D",
    "Plot3D",
    "coordinates_to_path",
    "make_gif",
    "save_figure",
    "smooth_contour_fill",
    "str_to_latex",
]


def gsymbolify(string: str) -> str:
    """
    Convert a string to a LaTEX printable greek letter if detected.

    Parameters
    ----------
    string:
        The string to add Greek symbols to

    Returns
    -------
    The modified string. Returns input if no changes made
    """
    if string in GREEK_ALPHABET or string in GREEK_ALPHABET_CAPS:
        return "\\" + string
    return string


def str_to_latex(string: str) -> str:
    """
    Create a new string which can be printed in LaTEX nicely.

    Parameters
    ----------
    string:
        The string to be converted

    Returns
    -------
    The mathified string

    'I_m_p' ==> '$I_{m_{p}}$'
    """
    s = string.split("_")
    s = [gsymbolify(sec) for sec in s]
    ss = "".join(["_" + "{" + lab for i, lab in enumerate(s[1:])])
    return "$" + s[0] + ss + "}" * (len(s) - 1) + "$"


def make_gif(folder: str, figname: str, file_format: str = "png", *, clean: bool = True):
    """
    Make a GIF image from a set of images with similar names in a folder.
    Figures are sorted in increasing order based on a trailing number, e.g.
    'figure_A[1, 2, 3, ..].png'
    Cleans up the temporary figure files (deletes!)
    Creates a GIF file in the folder directory

    Parameters
    ----------
    folder:
        Full path folder name
    figname:
        Figure name prefix
    file_format:
        Figure filename extension
    clean:
        Delete figures after completion?
    """
    find_digit = re.compile(r"(\\d+)")

    ims = sorted(
        [
            Path(folder, filename)
            for filename in Path(folder).iterdir()
            if str(filename).startswith(figname) and str(filename).endswith(file_format)
        ],
        key=lambda x: int(find_digit.findall(x)[-1]),
    )
    imageio.mimsave(
        Path(folder, f"{figname}.gif"),
        [imageio.imread(fp) for fp in ims],
        "GIF-FI",
        duration=0.5,
        loop=3,
    )
    if clean:
        for fp in ims:
            fp.unlink()


def xz_plot_setup(
    pname,
    folder,
    *,
    save: bool = False,
    split_psi_plots: bool = False,
) -> dict:
    """
    Set up for an xz plot (poloidal slice).

    Returns
    -------
    f:
        figure
    ax:
        axis
    pname:
        plot name
    folder:
        plot folder
    save:
        to save or not to save

    """
    if folder is None:
        folder = try_get_bluemira_path(
            "", subfolder="generated_data", allow_missing=not save
        )

    if split_psi_plots:
        f, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        ax[0].set_xlabel("$x$ [m]")
        ax[0].set_ylabel("$z$ [m]")
        ax[0].set_title("Coilset")
        ax[0].set_aspect("equal")
        ax[1].set_xlabel("$x$ [m]")
        ax[1].set_ylabel("$z$ [m]")
        ax[1].set_title("Plasma")
        ax[1].set_aspect("equal")

    else:
        f, ax = plt.subplots()
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$z$ [m]")
        ax.set_aspect("equal")
    return {
        "f": f,
        "ax": ax,
        "pname": pname,
        "folder": folder,
        "save": save,
    }


def save_figure(
    fig, name, *, save=False, folder=None, dpi=600, file_format="svg", **kwargs
):
    """
    Saves a figure to the directory if save flag active
    """
    if save is True:
        if folder is None:
            folder = get_bluemira_path("plots", subfolder="data")
        name = Path(folder, f"{name}.{file_format}")
        if name.is_file():
            name.unlink()  # f.savefig will otherwise not overwrite
        fig.savefig(name, dpi=dpi, bbox_inches="tight", format=file_format, **kwargs)


def ring_coding(n: int) -> np.ndarray:
    """
    The codes will be all "LINETO" commands, except for "MOVETO"s at the
    beginning of each subpath

    Returns
    -------
    :
        The ring coded array
    """
    codes = np.ones(n, dtype=Path_mpl.code_type) * Path_mpl.LINETO
    codes[0] = Path_mpl.MOVETO
    return codes


def coordinates_to_path(x: np.ndarray, z: np.ndarray) -> Path_mpl:
    """
    Convert coordinates to path vertices.

    Returns
    -------
    :
        The path vertices
    """
    if not check_ccw(x, z):
        x = x[::-1]
        z = z[::-1]
    vertices = np.array([x, z]).T
    codes = ring_coding(len(x))
    return Path_mpl(vertices, codes)


def set_component_view(comp: Component, placement: str | BluemiraPlacement):
    if placement not in {"xy", "xz", "yz"} and not isinstance(
        placement, BluemiraPlacement
    ):
        raise bm_display_error.DisplayError(
            f"Not a valid view {placement} - select either xy, xz, yz, "
            "or a BluemiraPlacement"
        )

    comp.plot_options.view = placement
    for child in comp.children:
        set_component_view(child, placement)


def smooth_contour_fill(ax: Axes, contour: ContourSet, cut_edge: Coordinates):
    """Smooths the edge of a filled contour with a set of coordinates"""
    edge_arr = cut_edge.xz.T
    clip_patch = PathPatch(
        Path_mpl(edge_arr, ring_coding(len(edge_arr))),
        facecolor="none",
        edgecolor="none",
    )
    ax.add_patch(clip_patch)
    contour.set_clip_path(clip_patch)


class Plot3D(Axes3D):
    """
    Cheap and cheerful
    """

    def __init__(self):
        fig = plt.figure(figsize=[14, 14])
        super().__init__(fig, auto_add_to_figure=False)
        fig.add_axes(self)
        # \n to offset labels from axes
        self.set_xlabel("\n\nx [m]")
        self.set_ylabel("\n\ny [m]")
        self.set_zlabel("\n\nz [m]")


class BluemiraPathPatch3D(PathPatch3D):
    """
    Class for a 3-D PathPatch which can actually be filled properly!

    Parameters
    ----------
    path:
        The path object to plot in 3-D
    normal:
        The 3-D normal vector of the face
    translation:
        Translation vector to apply to the face
    color:
        The color to plot the fill
    """

    # Thank you StackOverflow
    # https://stackoverflow.com/questions/18228966/how-can-matplotlib-2d-patches-be-transformed-to-3d-with-arbitrary-normals
    def __init__(
        self,
        path: Path_mpl,
        normal: np.ndarray,
        translation: np.ndarray | None = None,
        color: str = "b",
        **kwargs,
    ):
        # Initialise the patch first, or we can get into nasty recursive
        # calls in __getattr__
        self._patch2d = PathPatch(path, color=color, **kwargs)

        Patch.__init__(self, **kwargs)

        if translation is None:
            translation = [0, 0, 0]

        self._path2d = path
        self._code3d = path.codes
        self._facecolor3d = self._patch2d.get_facecolor

        r_matrix = rotation_matrix_v1v2(normal, (0, 0, 1))
        t_matrix = np.array(translation)

        points = path.vertices

        new_points = np.array([np.dot(r_matrix, np.array([x, y, 0])) for x, y in points])

        self._segment3d = new_points + t_matrix

    def __getattr__(self, key: str):
        """
        Transfer the key getattr to underlying PathPatch object.

        Returns
        -------
        :
            The key's value
        """
        return getattr(self._patch2d, key)
