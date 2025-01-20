# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Neutronics sources"""

import numpy as np
import openmc

from bluemira.base.constants import raw_uc
from bluemira.codes.openmc.params import PlasmaSourceParameters
from bluemira.radiation_transport.error import SourceError
from bluemira.radiation_transport.neutronics.constants import DT_NEUTRON_ENERGY

try:
    from pps_isotropic.source import create_parametric_plasma_source

    PPS_ISO_INSTALLED = True
except ImportError:
    PPS_ISO_INSTALLED = False


def make_pps_source(source_parameters: PlasmaSourceParameters) -> openmc.Source:
    """Make a plasma source

    Raises
    ------
    SourceError
        Source not found
    """  # noqa: DOC201
    if not PPS_ISO_INSTALLED:
        raise SourceError("pps_isotropic installation not found")
    return create_parametric_plasma_source(
        # tokamak geometry
        major_r=source_parameters.plasma_physics_units.major_radius,
        minor_r=source_parameters.plasma_physics_units.minor_radius,
        elongation=source_parameters.plasma_physics_units.elongation,
        triangularity=source_parameters.plasma_physics_units.triangularity,
        # plasma geometry
        peaking_factor=source_parameters.plasma_physics_units.peaking_factor,
        temperature=source_parameters.plasma_physics_units.temperature,
        radial_shift=source_parameters.plasma_physics_units.shaf_shift,
        vertical_shift=source_parameters.plasma_physics_units.vertical_shift,
        # plasma type
        mode="DT",
    )


def make_ring_source(source_parameters: PlasmaSourceParameters) -> openmc.Source:
    """Create the ring source"""  # noqa: DOC201
    return create_ring_source(
        source_parameters.plasma_physics_units.major_radius,
        source_parameters.plasma_physics_units.shaf_shift,
    )


def create_ring_source(major_r_cm: float, shaf_shift_cm: float) -> openmc.Source:
    """
    Creating simple line ring source lying on the Z=0 plane,
    at r = major radius + shafranov shift,
    producing 14.1 MeV neutrons with no variation in energy.
    A more accurate source will slightly affect the wall loadings and dpa profiles.

    Parameters
    ----------
    major_r_cm:
        major radius [cm]
    shaf_shift_cm:
        shafranov shift [cm]
    """  # noqa: DOC201
    ring_source = openmc.IndependentSource()
    source_radii_cm = openmc.stats.Discrete([major_r_cm + shaf_shift_cm], [1])
    source_z_values = openmc.stats.Discrete([0], [1])
    source_angles = openmc.stats.Uniform(a=0.0, b=2 * np.pi)
    ring_source.space = openmc.stats.CylindricalIndependent(
        r=source_radii_cm, phi=source_angles, z=source_z_values, origin=(0.0, 0.0, 0.0)
    )
    ring_source.angle = openmc.stats.Isotropic()
    ring_source.energy = openmc.stats.Discrete(
        [raw_uc(DT_NEUTRON_ENERGY, "J", "eV")], [1]
    )

    return ring_source
