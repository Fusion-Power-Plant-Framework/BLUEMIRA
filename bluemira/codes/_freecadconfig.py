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
FreeCAD configuration
"""
import enum  # noqa: I001

import freecad  # noqa: F401
import FreeCAD


class _Unit(enum.IntEnum):
    """Available units in FreeCAD"""

    MM = 0  # mmKS
    SI = 1  # MKS
    US = 2  # in/lb
    IMP_DEC = 3  # imperial_decimal
    BUILD_EURO = 4  # cm/m2/m3
    BUILD_US = 5  # ft-in/sqft/cft
    CNC = 6  # mm, mm/min
    IMP_CIV = 7  # ft, ft/sec
    FEM = 8  # mm/N/s


class _StpFileScheme(enum.Enum):
    """Available STEP file schemes in FreeCAD"""

    AP203 = enum.auto()
    AP214CD = enum.auto()
    AP214DIS = enum.auto()
    AP214IS = enum.auto()
    AP242DIS = enum.auto()


def _freecad_save_config(
    unit: str = "SI",
    no_dp: int = 5,
    author: str = "Bluemira",
    stp_file_scheme: str = "AP242DIS",
):
    """
    Attempts to configure FreeCAD with units file schemes and attributions

    This must be run before Part is imported for legacy exporters
    """
    unit_prefs = FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Units")
    # Seems to have little effect on anything but its an option to set
    # does effect the GUI be apparently not the base unit of the built part...
    unit_prefs.SetInt("UserSchema", _Unit[unit].value)
    unit_prefs.SetInt("Decimals", no_dp)  # 100th mm

    part_step_prefs = FreeCAD.ParamGet(
        "User parameter:BaseApp/Preferences/Mod/Part/STEP"
    )
    part_step_prefs.SetString("Scheme", _StpFileScheme[stp_file_scheme].name)
    part_step_prefs.SetString("Author", author)
    part_step_prefs.SetString("Company", "Bluemira")
    # Seems to have little effect on anything but its an option to set
    part_step_prefs.SetInt("Unit", _Unit[unit].value)

    part_gen_prefs = FreeCAD.ParamGet(
        "User parameter:BaseApp/Preferences/Mod/Part/General"
    )
    part_gen_prefs.SetInt("WriteSurfaceCurveMode", 1)

    import_prefs = FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Mod/Import")
    import_prefs.SetInt("ImportMode", 0)
    import_prefs.SetBool("ExportLegacy", False)
