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
"""Functions to optimise an EUDEMO radial build"""

from typing import Dict, TypeVar

from bluemira.base.parameter_frame import ParameterFrame
from bluemira.codes import plot_radial_build, systems_code_solver
from bluemira.codes.process._equation_variable_mapping import Constraint, Objective
from bluemira.codes.process._model_mapping import (
    AlphaPressureModel,
    AvailabilityModel,
    BetaLimitModel,
    BlanketModel,
    BootstrapCurrentScalingLaw,
    CSSuperconductorModel,
    ConfinementTimeScalingLaw,
    CostModel,
    CurrentDriveEfficiencyModel,
    DensityLimitModel,
    EPEDScalingModel,
    FISPACTSwitchModel,
    LHThreshholdScalingLaw,
    OperationModel,
    OutputCostsSwitch,
    PFConductorModel,
    PFSuperconductorModel,
    PROCESSOptimisationAlgorithm,
    PlasmaCurrentScalingLaw,
    PlasmaGeometryModel,
    PlasmaNullConfigurationModel,
    PlasmaPedestalModel,
    PlasmaProfileModel,
    PlasmaWallGapModel,
    PowerFlowModel,
    PrimaryPumpingModel,
    PulseTimingModel,
    SecondaryCycleModel,
    ShieldThermalHeatUse,
    SolenoidSwitchModel,
    TFCSTopologyModel,
    TFCasingGeometryModel,
    TFCoilConductorTechnology,
    TFCoilShapeModel,
    TFNuclearHeatingModel,
    TFSuperconductorModel,
    TFWindingPackGeometryModel,
    TFWindingPackTurnModel,
)
from bluemira.codes.process.api import Impurities
from bluemira.codes.process.template_builder import PROCESSTemplateBuilder

_PfT = TypeVar("_PfT", bound=ParameterFrame)


template_builder = PROCESSTemplateBuilder()
template_builder.set_optimisation_algorithm(PROCESSOptimisationAlgorithm.VMCON)
template_builder.set_optimisation_numerics(max_iterations=1000, tolerance=1e-8)

template_builder.set_minimisation_objective(Objective.MAJOR_RADIUS)

for constraint in (
    Constraint.BETA_CONSISTENCY,
    Constraint.GLOBAL_POWER_CONSISTENCY,
    Constraint.RADIAL_BUILD_CONSISTENCY,
    Constraint.CONFINEMENT_RATIO_LOWER_LIMIT,
    Constraint.DENSITY_UPPER_LIMIT,
    Constraint.DENSITY_PROFILE_CONSISTENCY,
    Constraint.BETA_UPPER_LIMIT,
    Constraint.NWL_UPPER_LIMIT,
    Constraint.BURN_TIME_LOWER_LIMIT,
    Constraint.NET_ELEC_LOWER_LIMIT,
    Constraint.LH_THRESHHOLD_LIMIT,
    Constraint.PSEPB_QAR_UPPER_LIMIT,
    Constraint.PINJ_UPPER_LIMIT,
    Constraint.DUMP_TIME_LOWER_LIMIT,
    Constraint.TF_CASE_STRESS_UPPER_LIMIT,
    Constraint.TF_JACKET_STRESS_UPPER_LIMIT,
    Constraint.TF_JCRIT_RATIO_UPPER_LIMIT,
    Constraint.TF_DUMP_VOLTAGE_UPPER_LIMIT,
    Constraint.TF_CURRENT_DENSITY_UPPER_LIMIT,
    Constraint.TF_T_MARGIN_LOWER_LIMIT,
    Constraint.CS_FATIGUE,
    Constraint.CS_STRESS_UPPER_LIMIT,
    Constraint.CS_T_MARGIN_LOWER_LIMIT,
    Constraint.CS_EOF_DENSITY_LIMIT,
    Constraint.CS_BOP_DENSITY_LIMIT,
):
    template_builder.add_constraint(constraint)

# Variable vector values and bounds
template_builder.add_variable("bt", 5.3292, upper_bound=20.0)
template_builder.add_variable("rmajor", 8.8901, upper_bound=13.0)
template_builder.add_variable("te", 12.33, upper_bound=150.0)
template_builder.add_variable("beta", 3.1421e-2)
template_builder.add_variable("dene", 7.4321e19)
template_builder.add_variable("q", 3.5, lower_bound=3.5)
template_builder.add_variable("pheat", 50.0)
template_builder.add_variable("ralpne", 6.8940e-02)
template_builder.add_variable("bore", 2.3322, lower_bound=0.1)
template_builder.add_variable("ohcth", 0.55242, lower_bound=0.1)
template_builder.add_variable("thwcndut", 8.0e-3, lower_bound=8.0e-3)
template_builder.add_variable("thkcas", 0.52465, lower_bound=0.1)
template_builder.add_variable("tfcth", 1.2080, lower_bound=0.2)
template_builder.add_variable("gapoh", 0.05, lower_bound=0.05, upper_bound=0.1)
template_builder.add_variable("gapds", 0.02, lower_bound=0.02)
template_builder.add_variable("oh_steel_frac", 0.57875)
template_builder.add_variable("coheof", 2.0726e07)
template_builder.add_variable("cpttf", 6.5e4, lower_bound=6.0e4, upper_bound=9.0e4)
template_builder.add_variable("tdmptf", 2.5829e01)
template_builder.add_variable("vdalw", 10.0, upper_bound=10.0)
template_builder.add_variable("fimp(13)", 3.573e-04)

# Some constraints require multiple f-values, but they are getting ridding of those,
# so no fancy mechanics for now...
template_builder.add_variable("fcutfsu", 0.80884, lower_bound=0.5, upper_bound=0.94)
template_builder.add_variable("fcohbop", 0.93176)
template_builder.add_variable("fvsbrnni", 0.39566)
template_builder.add_variable("fncycle", 1.0)
template_builder.add_variable("feffcd", 1.0)

# Modified f-values and bounds w.r.t. defaults [0.001 < 0.5 < 1.0]
template_builder.adjust_variable("fne0", 0.6, upper_bound=0.95)
template_builder.adjust_variable("fdene", 1.2, upper_bound=1.2)
template_builder.adjust_variable("flhthresh", 1.2, lower_bound=1.1, upper_bound=1.2)
template_builder.adjust_variable("ftburn", 1.0, upper_bound=1.0)

# Modifying the initial variable vector to improve convergence
template_builder.adjust_variable("fpnetel", 1.0)
template_builder.adjust_variable("fncycle", 1.0)
template_builder.adjust_variable("fstrcase", 1.0)
template_builder.adjust_variable("ftmargtf", 1.0)
template_builder.adjust_variable("ftmargoh", 1.0)
template_builder.adjust_variable("ftaulimit", 1.0)
template_builder.adjust_variable("fjohc", 0.57941)
template_builder.adjust_variable("fjohc0", 0.53923)
template_builder.adjust_variable("foh_stress", 1.0)
template_builder.adjust_variable("fbetatry", 0.48251)
template_builder.adjust_variable("fwalld", 0.131)
# template_builder.adjust_variable("ftaucq", 0.93)
template_builder.adjust_variable("fpsepbqar", 1.0)
template_builder.adjust_variable("fvdump", 1.0)
template_builder.adjust_variable("fstrcond", 0.92007)
template_builder.adjust_variable("fiooic", 0.63437)
template_builder.adjust_variable("fjprot", 1.0)
template_builder.adjust_variable("fpinj", 1.0)

# Set model switches
for model_choice in (
    BootstrapCurrentScalingLaw.SAUTER,
    ConfinementTimeScalingLaw.IPB98_Y2_H_MODE,
    PlasmaCurrentScalingLaw.ITER_REVISED,
    PlasmaProfileModel.CONSISTENT,
    PlasmaPedestalModel.PEDESTAL_GW,
    EPEDScalingModel.SAARELMA,
    BetaLimitModel.THERMAL,
    DensityLimitModel.GREENWALD,
    AlphaPressureModel.WARD,
    LHThreshholdScalingLaw.MARTIN_NOM,
    PlasmaNullConfigurationModel.SINGLE_NULL,
    PlasmaGeometryModel.CREATE_A_M_S,
    PlasmaWallGapModel.INPUT,
    PowerFlowModel.SIMPLE,
    PrimaryPumpingModel.PRESSURE_DROP_INPUT,
    ShieldThermalHeatUse.LOW_GRADE_HEAT,
    SecondaryCycleModel.INPUT,
    BlanketModel.CCFE_HCPB,
    CurrentDriveEfficiencyModel.ECRH_UI_GAM,
    OperationModel.PULSED,
    PulseTimingModel.RAMP_RATE,
    PFConductorModel.SUPERCONDUCTING,
    PFSuperconductorModel.NBTI,
    SolenoidSwitchModel.SOLENOID,
    CSSuperconductorModel.NB3SN_WST,
    TFCasingGeometryModel.FLAT,
    TFCoilConductorTechnology.SC,
    TFCoilShapeModel.PRINCETON,
    TFCSTopologyModel.ITER,
    TFSuperconductorModel.NB3SN_WST,
    TFWindingPackGeometryModel.RECTANGULAR,
    TFWindingPackTurnModel.INTEGER_TURN,
    FISPACTSwitchModel.OFF,
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
template_builder.add_input_values(
    {
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
        "taulimit": 5.0,
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
        "d_vv_in": 0.3,
        "shldith": 0.3,
        "vvblgap": 0.02,
        "blnkith": 0.755,
        "scrapli": 0.225,
        "scraplo": 0.225,
        "blnkoth": 0.982,
        "d_vv_out": 0.3,
        "shldoth": 0.8,
        "ddwex": 0.15,
        "gapomin": 0.2,
        "thshield_ib": 0.05,
        "thshield_ob": 0.05,
        "thshield_vb": 0.05,
        # Vertical build inputs
        "d_vv_top": 0.3,
        "vgap2": 0.05,
        "shldtth": 0.3,
        "shldlth": 0.3,
        "divfix": 0.621,
        "d_vv_bot": 0.3,
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
        # "t_crack_radial": 0.006,
        "t_crack_vertical": 0.004,
        # "t_structural_radial": 0.07,
        # "t_structural_vertical": 0.022,
        # "sf_vertical_crack": 1.0,
        # "sf_radial_crack": 1.0,
        # "sf_fast_fracture": 1.0,
        # "residual_sig_hoop": 1.5e8,
        # "paris_coefficient": 3.86e-11,
        # "paris_power_law": 2.394,
        # "walker_coefficient": 0.5,
        # "fracture_toughness": 150.0,
        "fcuohsu": 0.7,
        "ohhghf": 0.9,
        "rpf2": -1.825,
        "cptdin": [4.22e4, 4.22e4, 4.22e4, 4.22e4, 4.3e4, 4.3e4, 4.3e4, 4.3e4],
        "ipfloc": [2, 2, 3, 3],
        "ncls": [1, 1, 2, 2],
        "ngrp": 4,
        "rjconpf": [1.1e7, 1.1e7, 6.0e6, 6.0e6, 8.0e6, 8.0e6, 8.0e6, 8.0e6],
        "zref": [3.6, 1.2, 1.0, 2.8, 1.0, 1.0, 1.0, 1.0],
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
        # "max_vv_stress": 93.0e6,
        # Inputs we don't care about but must specify
        "cfactr": 0.75,  # Ha!
        "kappa": 1.848,  # Should be overwritten
        "walalw": 8.0,  # Should never get even close to this
        "tlife": 40.0,
        # For sanity...
        "divdum": 1,
        "hldivlim": 10,
        "ksic": 1.4,
        "prn1": 0.4,
        "zeffdiv": 35,
        "bmxlim": 11.2,
        "ffuspow": 1.0,
        "fpeakb": 1.0,
    }
)

template = template_builder.make_inputs()


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
        build_config["template_in_dat"] = template
    solver = systems_code_solver({}, build_config)
    new_params = solver.execute(run_mode)

    if plot:
        plot_radial_build(solver.read_directory)
    params.update_from_frame(new_params)
    return params
