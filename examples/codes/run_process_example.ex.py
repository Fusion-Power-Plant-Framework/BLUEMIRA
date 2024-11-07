# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Run PROCESS using the PROCESSTemplateBuilder
"""

# %% [markdown]
# # Running PROCESS from "scratch"
# PROCESS is one of the codes bluemira can use to compliment a reactor design.
# As with any of the external codes bluemira uses, a solver object is created.
# The solver object abstracts away most of the complexities of running different
# programs within bluemira.
#
# This example shows how to build a PROCESS template IN.DAT file

# %%
from bluemira.base.look_and_feel import bluemira_error
from bluemira.codes import systems_code_solver
from bluemira.codes.error import CodesError
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

# %% [markdown]
# First we are going to build a template using the :py:class:`PROCESSTemplateBuilder`,
# without interacting with any of PROCESS' integers.

# %%

template_builder = PROCESSTemplateBuilder()


# %% [markdown]
# Now we're going to specify which optimisation algorithm we want to use, and the
# number of iterations and tolerance.

# %%
template_builder.set_run_title("Example that won't converge")
template_builder.set_optimisation_algorithm(PROCESSOptimisationAlgorithm.VMCON)
template_builder.set_optimisation_numerics(max_iterations=1000, tolerance=1e-8)


# %% [markdown]
# Let's select the optimisation objective as the major radius:

# %%
template_builder.set_minimisation_objective(Objective.MAJOR_RADIUS)

# %% [markdown]
# You can inspect what options are available by taking a look at the
# :py:class:`Objective` Enum. The options are hopefully self-explanatory.
# The values of the options correspond to the PROCESS integers.

# %%
print("\n".join(str(o) for o in list(Objective)))


# %% [markdown]
# Now we will add a series of constraint equations to the PROCESS problem
# we wish to solve. You can read more about these constraints an what
# they mean in the PROCESS documentation

# %%
for constraint in (
    Constraint.BETA_CONSISTENCY,
    Constraint.GLOBAL_POWER_CONSISTENCY,
    Constraint.DENSITY_UPPER_LIMIT,
    Constraint.RADIAL_BUILD_CONSISTENCY,
    Constraint.BURN_TIME_LOWER_LIMIT,
    Constraint.LH_THRESHHOLD_LIMIT,
    Constraint.NET_ELEC_LOWER_LIMIT,
    Constraint.TF_CASE_STRESS_UPPER_LIMIT,
    Constraint.TF_JACKET_STRESS_UPPER_LIMIT,
    Constraint.TF_JCRIT_RATIO_UPPER_LIMIT,
    Constraint.TF_CURRENT_DENSITY_UPPER_LIMIT,
    Constraint.TF_T_MARGIN_LOWER_LIMIT,
    Constraint.PSEPB_QAR_UPPER_LIMIT,
):
    template_builder.add_constraint(constraint)


# %% [markdown]
# Many of these constraints require certain iteration variables to have been
# specified, or certain input values. The novice user can easily not be
# aware that this is the case, or simply forget to specify these.

# The :py:class:`PROCESSTemplateBuilder` will warn the user if certain
# values have not been specified. For example, if we try to make a set of
# inputs for an IN.DAT now, we will get many warning messages:

# %%
inputs = template_builder.make_inputs()


# %% [markdown]
# So let's go ahead and add the iteration variables we want to the problem:

# %%
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
template_builder.add_variable("thkcas", 0.52465)
template_builder.add_variable("tfcth", 1.2080)
template_builder.add_variable("gapoh", 0.05, lower_bound=0.05, upper_bound=0.1)
template_builder.add_variable("gapds", 0.02, lower_bound=0.02)
template_builder.add_variable("cpttf", 6.5e4, lower_bound=6.0e4, upper_bound=9.0e4)
template_builder.add_variable("tdmptf", 2.5829e01)
template_builder.add_variable("fcutfsu", 0.80884, lower_bound=0.5, upper_bound=0.94)
template_builder.add_variable("fvsbrnni", 0.39566)

# %% [markdown]
# Many of the PROCESS constraints use so-called 'f-values', which are automatically
# added to the iteration variables using this API. However, often one wants to modify
# the defaults of these f-values, which one can do as such:

# %%
# Modified f-values and bounds w.r.t. defaults
template_builder.adjust_variable("fne0", 0.6, upper_bound=0.95)
template_builder.adjust_variable("fdene", 1.2, upper_bound=1.2)


# %% [markdown]
# Often one wants to specify certain impurity concentrations, and even use
# one of these as an iteration variable.

# %%
template_builder.add_impurity(Impurities.H, 1.0)
template_builder.add_impurity(Impurities.He, 0.1)
template_builder.add_impurity(Impurities.W, 5.0e-5)
template_builder.add_variable(Impurities.Xe.id(), 3.573e-04)


# %% [markdown]
# We also want to specify some input values that are not variables:

# %%
template_builder.add_input_values({
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
    # Vertical build inputs
    "d_vv_top": 0.3,
    "vgap2": 0.05,
    "shldtth": 0.3,
    "divfix": 0.621,
    "d_vv_bot": 0.3,
    # HCD inputs
    "pinjalw": 51.0,
    "gamma_ecrh": 0.3,
    "etaech": 0.4,
    "bootstrap_current_fraction_max": 0.99,
    # BOP inputs
    "etath": 0.375,
    "etahtp": 0.87,
    "etaiso": 0.9,
    "vfshld": 0.6,
    "tdwell": 0.0,
    "tramp": 500.0,
    # CS / PF coil inputs
    "t_crack_vertical": 0.4e-3,
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
})

# %% [markdown]
# PROCESS has many different models with integer-value 'switches'. We can specify
# these choices as follows:

# %%
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

# %% [markdown]
# Some of these model choices also require certain input values
# to be specified. If these are not specified by the user, default
# values are used, which may not be desirable. Let us see what
# we're still missing:

# %%
inputs = template_builder.make_inputs()

# %% [markdown]
# And now let's add those missing inputs:

# %%
template_builder.add_input_value("qnuc", 1.3e4)
template_builder.add_input_value("n_layer", 20)
template_builder.add_input_value("n_pancake", 20)


# %% [markdown]
# Finally, let us run PROCESS with our inputs. In this case, we're just running
# PROCESS as an external code (see e.g.
# [External code example](../codes/external_code.ex.py))
# So we are not interesed in passing any parameters into it. In future, once the
# input template has been refined to something desirable, one can pass in parameters
# in mapped names to PROCESS, and not need to explicitly know all the PROCESS
# parameter names.

# %%
solver = systems_code_solver(
    params={}, build_config={"template_in_dat": template_builder.make_inputs()}
)

try:
    result = solver.execute("run")
except CodesError as ce:
    bluemira_error(ce)

# %%
# Great, so it runs! All we need to do now is make sure we have properly
# specified our design problem, and perhaps adjust the initial values
# of the iteration variables to give the optimisation algorithm a better
# chance of finding a feasible point.

# %%

# TODO @je-cook: actually get to converge
# 3667
template_builder.set_run_title("Example that should converge")
template_builder.adjust_variable("fpnetel", 1.0)
template_builder.adjust_variable("fstrcase", 1.0)
template_builder.adjust_variable("ftmargtf", 1.0)
template_builder.adjust_variable("ftmargoh", 1.0)
template_builder.adjust_variable("ftaulimit", 1.0)
template_builder.adjust_variable("fbetatry", 0.48251)
template_builder.adjust_variable("fpsepbqar", 1.0)
template_builder.adjust_variable("fvdump", 1.0)
template_builder.adjust_variable("fstrcond", 0.92007)
template_builder.adjust_variable("fjprot", 1.0)

# %%

solver = systems_code_solver(
    params={}, build_config={"template_in_dat": template_builder.make_inputs()}
)

result = solver.execute("run")
# %%
