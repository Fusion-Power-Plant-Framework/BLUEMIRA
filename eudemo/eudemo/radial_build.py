# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Functions to optimise an EUDEMO radial build"""

from typing import Dict, TypeVar

from bluemira.base.parameter_frame import ParameterFrame
from bluemira.codes import plot_radial_build, systems_code_solver
from bluemira.codes.process.api import Impurities
from bluemira.codes.process.equation_variable_mapping import Constraint, Objective
from bluemira.codes.process.model_mapping import (
    AlphaPressureModel,
    AvailabilityModel,
    BetaLimitModel,
    BootstrapCurrentScalingLaw,
    CSSuperconductorModel,
    ConfinementTimeScalingLaw,
    CostModel,
    CurrentDriveEfficiencyModel,
    DensityLimitModel,
    EPEDScalingModel,
    OperationModel,
    OutputCostsSwitch,
    PFSuperconductorModel,
    PROCESSOptimisationAlgorithm,
    PlasmaCurrentScalingLaw,
    PlasmaGeometryModel,
    PlasmaNullConfigurationModel,
    PlasmaPedestalModel,
    PlasmaProfileModel,
    PowerFlowModel,
    PrimaryPumpingModel,
    SecondaryCycleModel,
    ShieldThermalHeatUse,
    SolenoidSwitchModel,
    TFNuclearHeatingModel,
    TFSuperconductorModel,
    TFWindingPackTurnModel,
)
from bluemira.codes.process.template_builder import PROCESSTemplateBuilder

_PfT = TypeVar("_PfT", bound=ParameterFrame)


template_builder = PROCESSTemplateBuilder()
template_builder.set_optimisation_algorithm(PROCESSOptimisationAlgorithm.VMCON)
template_builder.set_optimisation_numerics(max_iterations=1000, tolerance=1e-8)

template_builder.set_minimisation_objective(Objective.MAJOR_RADIUS)

for constraint in (
    Constraint.BETA_CONSISTENCY,
    Constraint.GLOBAL_POWER_CONSISTENCY,
    Constraint.DENSITY_UPPER_LIMIT,
    Constraint.NWL_UPPER_LIMIT,
    Constraint.RADIAL_BUILD_CONSISTENCY,
    Constraint.BURN_TIME_LOWER_LIMIT,
    Constraint.LH_THRESHHOLD_LIMIT,
    Constraint.NET_ELEC_LOWER_LIMIT,
    Constraint.BETA_UPPER_LIMIT,
    Constraint.CS_EOF_DENSITY_LIMIT,
    Constraint.CS_BOP_DENSITY_LIMIT,
    Constraint.PINJ_UPPER_LIMIT,
    Constraint.TF_CASE_STRESS_UPPER_LIMIT,
    Constraint.TF_JACKET_STRESS_UPPER_LIMIT,
    Constraint.TF_JCRIT_RATIO_UPPER_LIMIT,
    Constraint.TF_DUMP_VOLTAGE_UPPER_LIMIT,
    Constraint.TF_CURRENT_DENSITY_UPPER_LIMIT,
    Constraint.TF_T_MARGIN_LOWER_LIMIT,
    Constraint.CS_T_MARGIN_LOWER_LIMIT,
    Constraint.CONFINEMENT_RATIO_LOWER_LIMIT,
    Constraint.DUMP_TIME_LOWER_LIMIT,
    Constraint.PSEPB_QAR_UPPER_LIMIT,
    Constraint.CS_STRESS_UPPER_LIMIT,
    Constraint.DENSITY_PROFILE_CONSISTENCY,
    # Constraint.CS_FATIGUE,  TODO(je-cook) should be re-enabled
):
    template_builder.add_constraint(constraint)

# Variable vector values and bounds
template_builder.add_variable("bt", 5.3292, upper_bound=20.0)
template_builder.add_variable("rmajor", 9.2901, upper_bound=13.0)
template_builder.add_variable("te", 12.33, upper_bound=150.0)
template_builder.add_variable("beta", 3.4421e-2)
template_builder.add_variable("dene", 7.4321e19)
template_builder.add_variable("q95", 3.5, lower_bound=3.5)
template_builder.add_variable("pheat", 50.0)
template_builder.add_variable("ralpne", 6.8940e-02)
template_builder.add_variable("bore", 2.3322, lower_bound=0.1)
template_builder.add_variable("ohcth", 0.55242, lower_bound=0.1)
template_builder.add_variable("thwcndut", 8.0e-3, lower_bound=8.0e-3)
template_builder.add_variable("thkcas", 0.52465)
template_builder.add_variable("tfcth", 1.2080)
template_builder.add_variable("gapoh", 0.05, lower_bound=0.05, upper_bound=0.1)
template_builder.add_variable("gapds", 0.02, lower_bound=0.02)
template_builder.add_variable("oh_steel_frac", 0.57875)
template_builder.add_variable("coheof", 2.0726e07)
template_builder.add_variable("cpttf", 6.5e4, lower_bound=6.0e4, upper_bound=9.0e4)
template_builder.add_variable("tdmptf", 2.5829e01)
template_builder.add_variable("fimp(13)", 3.573e-04)

# Some constraints require multiple f-values, but they are getting ridding of those,
# so no fancy mechanics for now...
template_builder.add_variable("fcutfsu", 0.80884, lower_bound=0.5, upper_bound=0.94)
template_builder.add_variable("fcohbop", 0.93176)
template_builder.add_variable("fvsbrnni", 0.39566)
template_builder.add_variable("fncycle", 1.0)
# template_builder.add_variable("feffcd", 1.0, lower_bound=0.001, upper_bound=1.0)

# Modified f-values and bounds w.r.t. defaults
template_builder.adjust_variable("fne0", 0.6, upper_bound=0.95)
template_builder.adjust_variable("fdene", 1.2, upper_bound=1.2)
template_builder.adjust_variable("flhthresh", 1.2, lower_bound=1.1, upper_bound=1.2)
template_builder.adjust_variable("ftburn", 1.0, upper_bound=1.0)

# Modifying the initial variable vector to improve convergence
template_builder.adjust_variable("fpnetel", 1.0)
template_builder.adjust_variable("fstrcase", 1.0)
template_builder.adjust_variable("ftmargtf", 1.0)
template_builder.adjust_variable("ftmargoh", 1.0)
template_builder.adjust_variable("ftaulimit", 1.0)
template_builder.adjust_variable("fjohc", 0.57941, upper_bound=1.0)
template_builder.adjust_variable("fjohc0", 0.53923, upper_bound=1.0)
template_builder.adjust_variable("foh_stress", 1.0)
template_builder.adjust_variable("fbetatry", 0.48251)
template_builder.adjust_variable("fwalld", 0.131)
template_builder.adjust_variable("fmaxvvstress", 1.0)
template_builder.adjust_variable("fpsepbqar", 1.0)
template_builder.adjust_variable("fvdump", 1.0)
template_builder.adjust_variable("fstrcond", 0.92007)
template_builder.adjust_variable("fiooic", 0.63437, upper_bound=1.0)
template_builder.adjust_variable("fjprot", 1.0)

# Set model switches
for model_choice in (
    BootstrapCurrentScalingLaw.SAUTER,
    ConfinementTimeScalingLaw.IPB98_Y2_H_MODE,
    PlasmaCurrentScalingLaw.ITER_REVISED,
    PlasmaProfileModel.CONSISTENT,
    PlasmaPedestalModel.PEDESTAL_GW,
    PlasmaNullConfigurationModel.SINGLE_NULL,
    EPEDScalingModel.SAARELMA,
    BetaLimitModel.THERMAL,
    DensityLimitModel.GREENWALD,
    AlphaPressureModel.WARD,
    PlasmaGeometryModel.CREATE_A_M_S,
    PowerFlowModel.SIMPLE,
    ShieldThermalHeatUse.LOW_GRADE_HEAT,
    SecondaryCycleModel.INPUT,
    CurrentDriveEfficiencyModel.ECRH_UI_GAM,
    OperationModel.PULSED,
    PFSuperconductorModel.NBTI,
    SolenoidSwitchModel.SOLENOID,
    CSSuperconductorModel.NB3SN_WST,
    TFSuperconductorModel.NB3SN_WST,
    TFWindingPackTurnModel.INTEGER_TURN,
    PrimaryPumpingModel.PRESSURE_DROP_INPUT,
    TFNuclearHeatingModel.INPUT,
    CostModel.TETRA_1990,
    AvailabilityModel.INPUT,
    OutputCostsSwitch.NO,
):
    template_builder.set_model(model_choice)

template_builder.add_impurity(Impurities.H, 1.0)
template_builder.add_impurity(Impurities.He, 0.1)
template_builder.add_impurity(Impurities.W, 5.0e-5)

# Set fixed input values
template_builder.add_input_values({
    # CS fatigue variables
    "residual_sig_hoop": 150.0e6,
    # "n_cycle_min": ,
    # "t_crack_radial": ,
    # "t_structural_radial": ,
    "t_crack_vertical": 0.65e-3,
    "sf_vertical_crack": 1.0,
    "sf_radial_crack": 1.0,
    "sf_fast_fracture": 1.0,
    "paris_coefficient": 3.86e-11,
    "paris_power_law": 2.394,
    "walker_coefficient": 0.5,
    "fracture_toughness": 150.0,
    # Undocumented danger stuff
    "iblanket": 1,
    "lsa": 2,
    # Profile parameterisation inputs
    "alphan": 1.0,
    "alphat": 1.45,
    "rhopedn": 0.94,
    "rhopedt": 0.94,
    "tbeta": 2.0,
    "teped": 5.5,
    "tesep": 0.1,
    "fgwped": 0.85,
    "neped": 0.678e20,
    "nesep": 0.2e20,
    "dnbeta": 3.0,
    # Plasma impurity stuff
    "coreradius": 0.75,
    "coreradiationfraction": 0.6,
    # Important stuff
    "pnetelin": 500.0,
    "tbrnmn": 7.2e3,
    "sig_tf_case_max": 5.8e8,
    "sig_tf_wp_max": 5.8e8,
    "alstroh": 6.6e8,
    "psepbqarmax": 9.2,
    "aspect": 3.1,
    "m_s_limit": 0.1,
    "triang": 0.5,
    "q0": 1.0,
    "ssync": 0.6,
    "plasma_res_factor": 0.66,
    "gamma": 0.3,
    "hfact": 1.1,
    "life_dpa": 70.0,
    # Radial build inputs
    "tftsgap": 0.05,
    "vvblgap": 0.02,
    "blnkith": 0.755,
    "scrapli": 0.225,
    "scraplo": 0.225,
    "blnkoth": 0.982,
    "ddwex": 0.15,
    "gapomin": 0.2,
    # Vertical build inputs
    "vgap2": 0.05,
    "divfix": 0.621,
    # HCD inputs
    "pinjalw": 51.0,
    "gamma_ecrh": 0.3,
    "etaech": 0.4,
    "bscfmax": 0.99,
    # BOP inputs
    "etath": 0.375,
    "etahtp": 0.87,
    "etaiso": 0.9,
    "vfshld": 0.6,
    "tdwell": 0.0,
    "tramp": 500.0,
    # CS / PF coil inputs
    "fcuohsu": 0.7,
    "ohhghf": 0.9,
    "rpf2": -1.825,
    "cptdin": [4.22e4, 4.22e4, 4.22e4, 4.22e4, 4.3e4, 4.3e4, 4.3e4, 4.3e4],
    "ipfloc": [2, 2, 3, 3],
    "ncls": [1, 1, 2, 2],
    "ngrp": 4,
    "rjconpf": [1.1e7, 1.1e7, 6.0e6, 6.0e6, 8.0e6, 8.0e6, 8.0e6, 8.0e6],
    # TF coil inputs
    "n_tf": 16,
    "casthi": 0.06,
    "casths": 0.05,
    "ripmax": 0.6,
    "dhecoil": 0.01,
    "tftmp": 4.75,
    "thicndut": 2.0e-3,
    "tinstf": 0.008,
    # "tfinsgap": 0.01,
    "tmargmin": 1.5,
    "vftf": 0.3,
    "n_pancake": 20,
    "n_layer": 10,
    "qnuc": 1.292e4,
    "vdalw": 10.0,
    # Inputs we don't care about but must specify
    "cfactr": 0.75,  # Ha!
    "kappa": 1.848,  # Should be overwritten
    "walalw": 8.0,  # Should never get even close to this
    "tlife": 40.0,
    "abktflnc": 15.0,
    "adivflnc": 20.0,
    # For sanity...
    "hldivlim": 10,
    "ksic": 1.4,
    "prn1": 0.4,
    "zeffdiv": 3.5,
    "bmxlim": 11.2,
    "ffuspow": 1.0,
    "fpeakb": 1.0,
    "divdum": 1,
    "ibkt_life": 1,
    "fkzohm": 1.0245,
    "iinvqd": 1,
    "dintrt": 0.0,
    "fcap0": 1.15,
    "fcap0cp": 1.06,
    "fcontng": 0.15,
    "fcr0": 0.065,
    "fkind": 1.0,
    "ifueltyp": 1,
    "discount_rate": 0.06,
    "bkt_life_csf": 1,
    "ucblvd": 280.0,
    "ucdiv": 5e5,
    "ucme": 3.0e8,
    # Suspicous stuff
    "zref": [3.6, 1.2, 1.0, 2.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "fpinj": 1.0,
})


def apply_specific_interface_rules(params: _PfT):
    """
    Apply specific rules for the interface between PROCESS and BLUEMIRA
    that relate to the EU-DEMO design parameterisation
    """
    # Apply q_95 as a boundary on the iteration vector rather than a fixed input
    q_95_min = params.q95.value
    template_builder.adjust_variable("q95", value=q_95_min, lower_bound=q_95_min)

    # Apply thermal shield thickness to all values in PROCESS
    tk_ts = params.tk_ts.value
    template_builder.add_input_values({
        "thshield_ib": tk_ts,
        "thshield_ob": tk_ts,
        "thshield_vb": tk_ts,
    })

    # Apply the summation of "shield" and "VV" thicknesses in PROCESS
    default_vv_tk = 0.3
    tk_vv_ib = params.tk_vv_in.value
    tk_vv_ob = params.tk_vv_out.value
    tk_sh_ib = tk_vv_ib - default_vv_tk
    tk_sh_ob = tk_vv_ob - default_vv_tk
    template_builder.add_input_values({
        "shldith": tk_sh_ib,
        "shldoth": tk_sh_ob,
        "shldtth": tk_sh_ib,
        "shldlth": tk_sh_ib,
        "d_vv_in": default_vv_tk,
        "d_vv_out": default_vv_tk,
        "d_vv_top": default_vv_tk,
        "d_vv_bot": default_vv_tk,
    })


def radial_build(params: _PfT, build_config: Dict) -> _PfT:
    """
    Update parameters after a radial build is run/read/mocked using PROCESS.

    Parameters
    ----------
    params:
        Parameters on which to perform the solve (updated)
    build_config:
        Build configuration

    Returns
    -------
    Updated parameters following the solve.
    """
    run_mode = build_config.pop("run_mode", "mock")
    plot = build_config.pop("plot", False)
    if run_mode == "run":
        template_builder.set_run_title(
            build_config.pop("PROCESS_runtitle", "Bluemira EUDEMO")
        )
        apply_specific_interface_rules(params)
        build_config["template_in_dat"] = template_builder.make_inputs()
    solver = systems_code_solver(params, build_config)
    new_params = solver.execute(run_mode)

    if plot:
        plot_radial_build(solver.read_directory)

    params.update_from_frame(new_params)
    return params
