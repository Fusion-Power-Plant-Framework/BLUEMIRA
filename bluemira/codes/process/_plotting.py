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
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.codes.error import CodesError
from bluemira.codes.process.constants import NAME as PROCESS
from bluemira.utilities.tools import is_num


def boxr(
    ri: float, ro: float, w: float, off: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate coordinates for an arbitrary height radial width. Used in plotting.
    """
    xc = [ri, ri, ro, ro, ri]
    yc = [-w, w, w, -w, -w]
    yc = [i + off for i in yc]
    return xc, yc


def read_rb_line(line: str):
    """
    Inputs: a line from the PROCESS radial/vertical build
    Outputs: the first three columns from that line
    """
    line = line.split()
    for i, v in enumerate(line):
        if is_num(v) is False:
            if i > 0:
                line[0] = " ".join([line[0], v])
        elif is_num(v) is True:
            line[1] = float(v)
            line[2] = float(line[i + 1])
            return line[:3]
    return None


def strip_num(line: str, typ: str = "float", n: int = 0) -> int:
    """
    Returns a single number in a line
    """
    numb = [float(i) for i in line.split() if is_num(i) is True][n]
    if typ == "int":
        numb = int(numb)
    return numb


def read_n_line(line: str):
    """
    Reads a line from the PROCESS output in the format below:
    Major radius (m)   / (rmajor)     /           9.203  ITV
    Returns the variable name [0], its value [1], and the rest [2]
    """
    line = line.split()
    out = [""] * 3
    for word in line:
        if word.startswith("(") is True or word.endswith(")") is True:
            out[2] = " ".join([out[2], word]).lstrip()
        elif is_num(word) is True:
            out[1] = float(word)
        elif word.isupper() is True:
            out[2] = " ".join([out[2], word]).lstrip()
        else:
            out[0] = " ".join([out[0], word]).lstrip()
    return out


def setup_radial_build(run: Dict[str, Any], width: float = 1.0):
    """
    Plots radial and vertical build of a PROCESS run.

    Parameters
    ----------
    run:
        Dictionary of PROCESS outputs.

    Returns
    -------
    plots: Axes
        The Matplotlib Axes object.
    """
    from bluemira.display.plotter import plot_coordinates
    from bluemira.geometry.coordinates import Coordinates

    R_0 = run["R_0"]

    col = {
        "Gap": "w",
        "blanket": "#edb120",
        "TF coil": "#7e2f8e",
        "Vacuum vessel": "k",
        "Plasma": "#f77ec7",
        "first wall": "#edb120",
        "Machine bore": "w",
        "precomp": "#0072bd",
        "scrape-off": "#a2142f",
        "solenoid": "#0072bd",
        "Thermal shield": "#77ac30",
    }

    f, ax = plt.subplots(figsize=[14, 10])

    lpatches = []
    gkeys = [
        "blanket",
        "TF coil",
        "Vacuum vessel",
        "Plasma",
        "scrape-off",
        "solenoid",
        "Thermal shield",
    ]
    glabels = {
        "blanket": "Breeding blanket",
        "TF coil": "TF coil",
        "Plasma": "Plasma",
        "Vacuum vessel": "Vacuum vessel",
        "scrape-off": "Scrape-off layer",
        "solenoid": "Central solenoid",
        "Thermal shield": "Thermal shield",
    }
    for comp in run["Radial Build"]:
        xc, yc = boxr(comp[2] - comp[1], comp[2], width)
        yc = np.array(yc)
        coords = Coordinates({"x": xc, "y": yc})
        for key, c in col.items():
            if key in comp[0]:
                ax.plot(xc, yc, color=c, linewidth=0, label=key)
                if comp[1] > 0:
                    plot_coordinates(
                        coords, ax=ax, facecolor=c, edgecolor="k", linewidth=0
                    )
                if key in gkeys:
                    gkeys.remove(key)
                    lpatches.append(patches.Patch(color=c, label=glabels[key]))

    ax.set_xlim([0, np.ceil(run["Radial Build"][-1][-1])])
    ax.set_ylim([-width * 0.5, width * 0.5])
    ax.set_xticks([*list(ax.get_xticks()), R_0])
    ax.axes.set_axisbelow(False)

    def tick_format(value, n):  # noqa: ARG001
        if value == R_0:
            return "\n$R_{0}$"
        return int(value)

    def tick_formaty(value, n):  # noqa: ARG001
        if value == 0:
            return int(value)
        return ""

    ax.xaxis.set_major_formatter(plt.FuncFormatter(tick_format))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(tick_formaty))
    ax.set_xlabel("$x$ [m]")
    ax.set_aspect("equal")
    ax.legend(
        handles=lpatches,
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        frameon=False,
    )
    return ax


def process_RB_fromOUT(f):
    """
    Parse PROCESS radial build from an OUT.DAT file.
    """
    # If the input is a string, treat as file name, and ensure it is closed.
    if isinstance(f, (str, Path)):
        with open(f) as fh:
            return process_RB_fromOUT(fh)  # Recursive call with file object
    raw = f.readlines()
    raw = raw[1:]
    if not raw:
        raise OSError("Cannot read from input file.")
    if PROCESS not in raw[1] and PROCESS not in raw[2]:
        bluemira_warn(
            "Either this ain't a PROCESS OUT.DAT file, or those hijos "
            "changed the format."
        )

    def read_radial_build(num):  # Be careful that the numbers don't change
        rb = []
        num += 1
        while "***" not in raw[num]:
            if read_rb_line(raw[num]) is None:
                pass
            else:
                rb.append(read_rb_line(raw[num]))
            num += 1
        return rb

    flag1, flag2, flag3 = False, False, False
    for num, line in enumerate(raw):
        if "* Radial Build *" in line:
            flag1 = True
            rb = read_radial_build(num)
        if "n_tf" in line:
            flag2 = True
            n_TF = strip_num(line, typ="int")
        if "Major radius" in line:
            flag3 = True
            R_0 = strip_num(line)
        if flag1 and flag2 and flag3:
            break
    return {"Radial Build": rb, "n_TF": n_TF, "R_0": R_0}


def plot_radial_build(
    sys_code_dir: str, width: float = 1.0, show: bool = True
) -> plt.Axes:
    """
    Plot PROCESS radial build.

    Parameters
    ----------
    sys_code_dir:
        OUT.DAT directory location
    width:
        The relative width of the plot.
    show:
        If True then immediately display the plot, else delay displaying the plot until
        the user shows it, by default True.

    Returns
    -------
    The plot Axes object.
    """
    filename = Path(sys_code_dir, "OUT.DAT")

    if not filename.is_file():
        raise CodesError(f"Could not find PROCESS OUT.DAT file '{filename}'.")

    radial_build = process_RB_fromOUT(filename)
    ax = setup_radial_build(radial_build, width=width)
    if show:
        plt.show()
    return ax
