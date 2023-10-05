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
Importer for external code API and related functions
"""


def freecad_message_removal():
    """
    Remove annoying message about freecad libdir not being set
    """
    import importlib
    import os

    if "PATH_TO_FREECAD_LIBDIR" in os.environ:
        return os.environ["PATH_TO_FREECAD_LIBDIR"]
    freecad_default_path = None
    with open(importlib.util.find_spec("freecad").origin) as rr:
        for line in rr:
            if '_path_to_freecad_libdir = "' in line:
                freecad_default_path = line.split('"')[1]
                break
    if freecad_default_path is not None:
        os.environ["PATH_TO_FREECAD_LIBDIR"] = freecad_default_path

    return freecad_default_path


freecad_default_path = freecad_message_removal()

from bluemira.codes._freecadconfig import _freecad_save_config

_freecad_save_config()


# External codes wrapper imports
from bluemira.codes.wrapper import (
    plot_radial_build,
    systems_code_solver,
    transport_code_solver,
)

__all__ = [
    "systems_code_solver",
    "transport_code_solver",
    "plot_radial_build",
]
