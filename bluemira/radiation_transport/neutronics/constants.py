# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""constants used for the neutronics module"""

from periodictable import elements

from bluemira.base.constants import (
    ELECTRON_MOLAR_MASS,
    HE_MOLAR_MASS,
    NEUTRON_MOLAR_MASS,
    N_AVOGADRO,
    raw_uc,
)
from bluemira.geometry.constants import D_TOLERANCE
from bluemira.plasma_physics.reactions import E_DT_fusion

DTOL_CM = raw_uc(D_TOLERANCE, "m", "cm")


def to_cm(m):
    """
    Converter for m to cm

    Parameters
    ----------
    :
        quantity in m

    Returns
    -------
    :
        same quantity but expressed in cm
    """
    return raw_uc(m, "m", "cm")


def to_m(cm):
    """
    Converter for cm to m

    Parameters
    ----------
    :
        quantity in cm

    Returns
    -------
    :
        same quantity but expressed in m
    """
    return raw_uc(cm, "cm", "m")


def to_cm3(m3):
    """
    Converter for m3 to cm3

    Parameters
    ----------
    :
        quantity in m3

    Returns
    -------
    :
        same quantity but expressed in cm^3
    """
    return raw_uc(m3, "m^3", "cm^3")


# Amount of energy carried away by the neutron, which is about 4/5 of that.
ALHPA_MOLAR_MASS = HE_MOLAR_MASS - ELECTRON_MOLAR_MASS

# ignoring the binding energy of the electron, too minute.
DT_NEUTRON_ENERGY = E_DT_fusion() * (
    ALHPA_MOLAR_MASS / (ALHPA_MOLAR_MASS + NEUTRON_MOLAR_MASS)
)  # [J]

# Energy required to displace an Fe atom in Fe. See docstring of DPACoefficients
DPA_FE_THRESHOLD_EV = 40  # Source cites 40 eV.

# how many degrees misalignment tolerated while merging almost-parallel wires into one.
TOLERANCE_DEGREES = 6.0

# Default value to discretise the BluemiraWire.
# Set to 10 to preserve speed without too much loss in precision.
DISCRETISATION_LEVEL = 10


# The following material science constants are in cgs.
FE_MOLAR_MASS_G = elements.isotope("Fe").mass
FE_DENSITY_G_CC = elements.isotope("Fe").density


class DPACoefficients:
    """
    Get the coefficients required

    To convert the number of damage into the number of displacements.
    number of atoms in region = avogadro * density * volume / molecular mass
    number of atoms in 1 cc   = avogadro * density          / molecular mass
    dpa_per_second_of_operation = src_rate * displacements / atoms
    dpa_fpy = dpa_per_second_of_operation / S_TO_YEAR

    Notes
    -----
    .. doi:: 10.1016/j.rinp.2019.102835
        :title: Shengli Chena, David Bernard,
                "On the calculation of atomic displacements using damage energy"
                Results in Physics 16 (2020)
    """

    def __init__(
        self,
        density_g_cc: float = FE_DENSITY_G_CC,
        molar_mass_g: float = FE_MOLAR_MASS_G,
        dpa_threshold_eV: float = DPA_FE_THRESHOLD_EV,
    ):
        """
        Parameters
        ----------
        density_g_cc: float [g/cm^2]
            density of the wall material,
            where the damage (in DPA) would be calculated later.
        molar_mass_g: float [g/mole]
            molar mass of the wall material,
            where the damage (in DPA) would be calculated later.
        dpa_threshold_eV: float [eV/count]
            the average amount of energy dispersed
            by displacing one atom in the wall material's lattice.

        Attributes/values
        -----------------
        atoms_per_cc: number density, given in cgs.
        displacements_per_damage_eV:
        """
        self.atoms_per_cc = N_AVOGADRO * density_g_cc / molar_mass_g
        self.displacements_per_damage_eV = 0.8 / (2 * dpa_threshold_eV)
