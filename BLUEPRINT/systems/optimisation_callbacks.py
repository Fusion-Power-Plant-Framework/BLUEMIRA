"""Callbacks which can be used to perform optimisations on various
components."""

from time import time

import numpy as np

from bluemira.base.look_and_feel import bluemira_print
from bluemira.equilibria.constants import (
    NBTI_J_MAX,
    NB3SN_J_MAX,
)
from BLUEPRINT.nova.optimiser import StructuralOptimiser


def TF_optimiser(TF, verbose, kwargs):
    """The optimisation step when building TF coils."""
    TF.shp.optimise(verbose=verbose, **kwargs)


def TF_loader(TF, verbose, kwargs):
    """Instead of optimising TF coils, load a previous optimisation."""
    TF.shp.load()


def EQ_optimiser(EQ, TF, params, exclusions, plot_flag):
    eta_pf_imax = 1.4  # Maximum current scaling for PF coil
    if params.PF_material == "NbTi":
        jmax = NBTI_J_MAX
    elif params.PF_material == "Nb3Sn":
        jmax = NB3SN_J_MAX
    else:
        raise ValueError("Ainda nao!")

    offset = params.g_tf_pf + np.sqrt(eta_pf_imax * params.I_p / jmax) / 2
    tf_loop = TF.get_TF_track(offset)

    bluemira_print(
        "Designing plasma equilibria and PF coil system.\n"
        "|   optimising: positions and currents\n"
        "|   subject to: F, B, I, L, and plasma shape constraints"
    )
    t = time()
    EQ.optimise_positions(
        max_PF_current=eta_pf_imax * params.I_p * 1e6,
        PF_Fz_max=params.F_pf_zmax * 1e6,
        CS_Fz_sum=params.F_cs_ztotmax * 1e6,
        CS_Fz_sep=params.F_cs_sepmax * 1e6,
        tau_flattop=params.tau_flattop,
        v_burn=params.v_burn,
        psi_bd=None,  # Will calculate BD flux
        pfcoiltrack=tf_loop,
        pf_exclusions=exclusions,
        CS=False,
        plot=plot_flag,
        gif=False,
    )
    bluemira_print(f"optimisation time: {time()-t:.2f} s")


# Note: This doesn't seem to get called from anywhere.
def ATEC_optimiser(ATEC, TF, EQ):
    """
    Optimise the TF coil casing. WIP.
    """
    bluemira_print("Optimising coil structures.")
    SO = StructuralOptimiser(ATEC, TF.cage, [s.eq for s in EQ.snapshots.values()])
    t = time()
    SO.optimise()
    bluemira_print(f"Optimisation time: {time()-t:.2f} s")


def FW_optimiser(FW, hf_limit, n_iteration_max):
    """
    Optimises the initial preliminary first wall profile in terms of heat flux.
    The divertor will be attached to this profile.

    Parameters
    ----------
    FW: FirstWall
        The first wall system being optimised.

    n_iteration_max: integer
        Max number of iterations after which the optimiser is stopped.
    hf_limit: float
        Heat flux limit for the optimisation.
    """
    # NOTE: Not an optimisation
    FW.initial_profile = FW.make_preliminary_profile()
    profile = FW.initial_profile
    for _ in range(n_iteration_max):
        x_wall, z_wall, hf_wall = FW.hf_firstwall_params(profile)

        for x_hf, z_hf, hf in zip(x_wall, z_wall, hf_wall):
            if hf > hf_limit:
                profile = FW.modify_fw_profile(profile, x_hf, z_hf)

        heat_flux_max = max(hf_wall)
        print(heat_flux_max)
        FW.optimised_profile = profile
        if heat_flux_max < hf_limit:
            break
