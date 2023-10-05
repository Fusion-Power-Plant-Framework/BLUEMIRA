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
from dataclasses import dataclass
from typing import Dict, Union

from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.equilibria import Equilibrium
from bluemira.equilibria.profiles import BetaIpProfile, Profile
from bluemira.geometry.wire import BluemiraWire
from eudemo.equilibria.tools import make_grid
from eudemo.pf_coils.tools import make_coilset, make_reference_coilset

KAPPA_95_TO_100 = 1.12


@dataclass
class EquilibriumParams(ParameterFrame):
    """Parameters required to make a new equilibrium."""

    A: Parameter[float]
    B_0: Parameter[float]
    beta_p: Parameter[float]
    CS_bmax: Parameter[float]
    CS_jmax: Parameter[float]
    delta_95: Parameter[float]
    g_cs_mod: Parameter[float]
    I_p: Parameter[float]
    kappa_95: Parameter[float]
    n_CS: Parameter[int]
    n_PF: Parameter[int]
    PF_bmax: Parameter[float]
    PF_jmax: Parameter[float]
    R_0: Parameter[float]
    r_cs_in: Parameter[float]
    tk_cs_casing: Parameter[float]
    tk_cs_insulation: Parameter[float]
    tk_cs: Parameter[float]


def make_equilibrium(
    _params: Union[EquilibriumParams, Dict],
    tf_coil_boundary: BluemiraWire,
    grid_settings: dict,
):
    """
    Build an equilibrium using a coilset and a `BetaIpProfile` profile.
    """
    if isinstance(_params, dict):
        params = EquilibriumParams.from_dict(_params)
    else:
        params = _params

    kappa = KAPPA_95_TO_100 * params.kappa_95.value
    coilset = make_coilset(
        tf_coil_boundary,
        R_0=params.R_0.value,
        kappa=kappa,
        delta=params.delta_95.value,
        r_cs=params.r_cs_in.value + params.tk_cs.value / 2,
        tk_cs=params.tk_cs.value / 2,
        g_cs=params.g_cs_mod.value,
        tk_cs_ins=params.tk_cs_insulation.value,
        tk_cs_cas=params.tk_cs_casing.value,
        n_CS=params.n_CS.value,
        n_PF=params.n_PF.value,
        CS_jmax=params.CS_jmax.value,
        CS_bmax=params.CS_bmax.value,
        PF_jmax=params.PF_jmax.value,
        PF_bmax=params.PF_bmax.value,
    )
    profiles = BetaIpProfile(
        params.beta_p.value,
        params.I_p.value,
        params.R_0.value,
        params.B_0.value,
    )
    grid = make_grid(params.R_0.value, params.A.value, kappa, grid_settings)

    return Equilibrium(coilset, grid, profiles)


@dataclass
class ReferenceEquilibriumParams(ParameterFrame):
    """Parameters required to make a new reference equilibrium."""

    A: Parameter[float]
    B_0: Parameter[float]
    I_p: Parameter[float]
    kappa: Parameter[float]
    R_0: Parameter[float]
    r_cs_in: Parameter[float]
    g_cs_mod: Parameter[float]
    tk_cs_casing: Parameter[float]
    tk_cs_insulation: Parameter[float]
    tk_cs: Parameter[float]
    beta_p: Parameter[float]
    l_i: Parameter[float]
    n_CS: Parameter[int]
    n_PF: Parameter[int]


def make_reference_equilibrium(
    _params: Union[ReferenceEquilibriumParams, Dict],
    tf_track: BluemiraWire,
    lcfs_shape: BluemiraWire,
    profiles: Profile,
    grid_settings: dict,
):
    """
    Make a crude reference equilibrium, scaling coils and grid for a first pass
    solve.
    """
    if isinstance(_params, dict):
        params = ReferenceEquilibriumParams.from_dict(_params)
    else:
        params = _params

    coilset = make_reference_coilset(
        tf_track,
        lcfs_shape,
        r_cs=params.r_cs_in.value + 0.5 * params.tk_cs.value,
        tk_cs=0.5 * params.tk_cs.value,
        g_cs_mod=params.g_cs_mod.value,
        tk_cs_casing=params.tk_cs_casing.value,
        tk_cs_insulation=params.tk_cs_insulation.value,
        n_CS=params.n_CS.value,
        n_PF=params.n_PF.value,
    )

    grid = make_grid(
        params.R_0.value,
        params.A.value,
        params.kappa.value,
        grid_settings,
    )

    return Equilibrium(coilset, grid, profiles)
