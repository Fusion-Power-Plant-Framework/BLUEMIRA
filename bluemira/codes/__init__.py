# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Importer for external code API and related functions
"""


def freecad_message_removal():
    """
    Remove annoying message about freecad libdir not being set
    """
    import importlib  # noqa: PLC0415
    import os  # noqa: PLC0415

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
    powerbalance_solver,
    systems_code_solver,
    transport_code_solver,
)

__all__ = [
    "plot_radial_build",
    "powerbalance_solver",
    "systems_code_solver",
    "transport_code_solver",
]
