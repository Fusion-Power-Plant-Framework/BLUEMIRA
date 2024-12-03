# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.codes.error import CodesError
from bluemira.codes.process.constants import NAME as PROCESS
from bluemira.utilities.tools import is_num


def boxr(
    ri: float, ro: float, w: float, off: float = 0
) -> tuple[np.ndarray, np.ndarray]:
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
                line[0] = " ".join([line[0], v])  # noqa: B909
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


def setup_radial_build(run: dict[str, Any], width: float = 1.0):
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
    from bluemira.display.plotter import plot_coordinates  # noqa: PLC0415
    from bluemira.geometry.coordinates import Coordinates  # noqa: PLC0415

    R_0 = run["R_0"]

    col = {
        "Gap": "w",
        "blanket": "#edb120",
        "TF coil": "#7e2f8e",
        "Vacuum vessel": "k",
        "Radiation shield": "#5dbb63",
        "Plasma": "#f77ec7",
        "first wall": "#edb120",
        "Machine bore": "w",
        "precomp": "#0072bd",
        "scrape-off": "#a2142f",
        "solenoid": "#0072bd",
        "Thermal shield": "#77ac30",
    }

    _, ax = plt.subplots(figsize=[14, 10])

    lpatches = []
    gkeys = [
        "blanket",
        "TF coil",
        "Vacuum vessel",
        "Radiation shield",
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
        "Radiation shield": "Radiation shield",
        "scrape-off": "Scrape-off layer",
        "solenoid": "Central solenoid",
        "Thermal shield": "Thermal shield",
    }
    for comp in run["Radial Build"]:
        xc, yc = boxr(comp[2] - comp[1], comp[2], width)
        yc = np.array(yc)
        coords = Coordinates({"x": xc, "y": yc})
        for key, c in col.items():
            if key.upper() in comp[0].upper():
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
    ax.axes.set_axisbelow(b=False)

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

    Raises
    ------
    OSError
        Cannot read file
    """
    # If the input is a string, treat as file name, and ensure it is closed.
    if isinstance(f, str | Path):
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
            if "TF coil radial placement switch" in raw[num]:
                # PROCESS v3.0.0 added this line to the start of the RB
                # TF coil radial placement switch ... (tf_in_cs)  ....           0
                pass
            elif read_rb_line(raw[num]) is None:
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


def process_RB_fromMFILE(f):
    """
    Parse PROCESS radial build from an MFILE.DAT file.

    Raises
    ------
    OSError
        If unable to read file or parse the expected structure.
    """
    # If the input is a string, treat as file name, and ensure it is closed.
    if isinstance(f, str | Path):
        with open(f) as fh:
            return process_RB_fromMFILE(fh)  # Recursive call with file object
    raw = f.readlines()
    raw = raw[1:]
    if not raw:
        raise OSError("Cannot read from input file.")

    if PROCESS not in raw[1] and PROCESS not in raw[2]:
        bluemira_warn(
            "Either this ain't a PROCESS MFILE.DAT file, or they changed the format."
        )

    def read_radial_build(num):
        """
        Reads a radial build parameter from the line number `num`.

        In the MFILE, the Components are listed in 3 lines:

        - Gap_radial_thickness_(m)___ (vvblgap)____<value>
        - Radial_build_component_11 ___(radial_label(11))___vvblgap
        - Radial_build_cumulative_radius_11___(radial_cum(11))____<value>

        Be careful that `num` refers to the line number of the last line
        (i.e., with (radial_cum(<component number>))).
        """
        # Extract relevant lines
        l1, l3 = raw[num - 2], raw[num]

        # Extract values
        tk = strip_num(l1, typ="float")
        label = l1.split("_radial_thickness_(")[0].replace("_", " ")

        cum_tk = strip_num(l3, typ="float")

        if "gap" in label:
            # do not need to know gap between whom
            # for plotting
            label = "Gap"

        return [label, tk, cum_tk]

    rb = []
    flag1, flag2, flag3 = False, False, False

    for num, line in enumerate(raw):
        # Extract Radial Build Parameters
        if "# Vertical Build #" in line:
            # Stop collecting Radial Build lines
            flag1 = True
        if "(radial_cum(" in line:
            # Read Radial Build line
            rb.append(read_radial_build(num))

        if "(n_tf)" in line:
            # Flag that n_TF is read
            flag2 = True
            n_TF = strip_num(line, typ="int")
        if "(rmajor)" in line:
            # Flag that rmajor is read
            flag3 = True
            R_0 = strip_num(line)

        # If all necessary data is read, break out of the loop
        if flag1 and flag2 and flag3:
            break

    # Return parsed radial build data and other extracted parameters
    return {"Radial Build": rb, "n_TF": n_TF, "R_0": R_0}


def process_RB_from_dir(sys_code_dir: str):
    """
    Read Radial Build from output file in sys_code_dir

    If PROCESS_version_number >=3.1.0, Uses MFILE.DAT to plot

    Raises
    ------
    CodesError
        Cannot find OUT.DAT
    """
    mfilename = Path(sys_code_dir, "MFILE.DAT")

    if mfilename.is_file():
        with open(mfilename) as f:
            # Read all lines from the file and strip the first line (header)
            raw = f.readlines()[1:]
            # Get the Version Number
            process_version = float(
                ".".join(raw[3].split('"')[-2].strip().split(".")[:2])
            )
            min_version = 3.1
            if process_version >= min_version:
                # Return radial build from MFILE
                return process_RB_fromMFILE(mfilename)
            bluemira_warn("MFILE.DAT file in old format. Trying to use OUT.DAT")

    # Return radial build from OUT File
    # (older versions)
    ofilename = Path(sys_code_dir, "OUT.DAT")
    if not ofilename.is_file():
        raise CodesError(f"Could not find PROCESS OUT.DAT file '{ofilename}' either.")
    return process_RB_fromOUT(ofilename)


def plot_radial_build(
    sys_code_dir: str, width: float = 1.0, *, show: bool = True
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

    Raises
    ------
    CodesError
        Cannot find OUT.DAT
    """
    radial_build = process_RB_from_dir(sys_code_dir)
    ax = setup_radial_build(radial_build, width=width)
    if show:
        plt.show()
    return ax
