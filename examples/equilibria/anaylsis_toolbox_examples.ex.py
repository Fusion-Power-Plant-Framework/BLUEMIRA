# %%
# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Example of equilibria anaylsis utilities.
"""

from pathlib import Path

from bluemira.equilibria.analysis import EqAnalysis, MultiEqAnalysis, select_eq
from bluemira.equilibria.diagnostics import (
    EqDiagnosticOptions,
    EqSubplots,
    FluxSurfaceType,
    LCFSMask,
    PsiPlotType,
)
from bluemira.geometry.coordinates import Coordinates

# %%
# %pdb

# %% [markdown]
# ### Analysis Examples
#
# Below we highlight some example uses of the anaylsis classes in equilibria/analysis.py.
#
# **EqAnalysis:**
#
# Used to compare selected equilibrium to a refernce equilibrium,
# e.g., for comparing optimisation results to a starting equilibrium.
#
# **MultiEqAnalysis:**
#
# Used to compare the properties of two or more equilibria,
# e.g, for examining the effects of different optimisation runs
# on key physics parameters.
#
# **Note:**
#
# Both EqAnalysis and MultiEqAnalysis can take equilibria of different types
# (i.e, fixed or free), and/or with different grid sizes and grid resolutions.
# However, it should be noted that not all of the avaible tools can be applied
# to the fixed boundary equilibria.

# %% [markdown]
# #### Inputs
#
# We are going to use MAST-U-like and DEMO-like equilibria
# as the inputs for our examples.

# %%
# MAST-U-like
masty_path = Path("../../tests/equilibria/test_data/SH_test_file.json")
masty_eq = select_eq(masty_path)
# DEMO-like-SN
single_demoish_path = Path("../../tests/equilibria/test_data/eqref_OOB.json")
single_demoish_eq = select_eq(single_demoish_path, from_cocos=7)
# DEMO-like-DN
double_demoish_path = Path("../../tests/equilibria/test_data/DN-DEMO_eqref.json")
double_demoish_eq = select_eq(double_demoish_path)

# %% [markdown]
# #### EqAnalysis
#
# Fisrt we will look at the untilities avilable in the EqAnalysis class
# for comparision to a reference equilibria.
#
# The EqDiagnosticOptions class contains the information needed to
# set up the comparison plots used during an optimisation or
# (as is the case here) for anaylsis purposes.
#
# We will use the single null DEMO-like equilibrium as our reference equilibrium.

# %%
# Set up to look at the psi difference
diag_ops_1 = EqDiagnosticOptions(
    psi_diff=PsiPlotType.PSI_DIFF,
    split_psi_plots=EqSubplots.XZ,
    reference_eq=single_demoish_eq,
)
# Set up to look at the absolute psi difference seperately
# for plamsa and coilset contributions
diag_ops_2 = EqDiagnosticOptions(
    psi_diff=PsiPlotType.PSI_ABS_DIFF,
    split_psi_plots=EqSubplots.XZ_COMPONENT_PSI,
    reference_eq=single_demoish_eq,
    lcfs_mask=LCFSMask.IN,
)

# %%
# Here we create our anaylsis class with the MAST-U-like and
# the double null DEMO-like equilibria.
eq_analysis_1 = EqAnalysis(diag_ops_1, double_demoish_eq)
eq_analysis_1b = EqAnalysis(diag_ops_1, masty_eq)
eq_analysis_2 = EqAnalysis(diag_ops_2, double_demoish_eq)

# %%
# Plot equilibrium
eq_analysis_1.plot()

# %%
# Plot field components
eq_analysis_1.plot_field()

# %%
# Plot equilibrium normalised profiles
eq_analysis_1.plot_profiles()

# %%
# Both together
ax = eq_analysis_1.plot_equilibria_with_profiles()

# %%
# Plot 1-D profiles at magnetic axis
eq_analysis_1.plot_eq_core_mag_axis()

# %%
# PLot comparison of equilibrium profiles to reference equilibrium profiles
eq_analysis_1.plot_compare_profiles()

# %%
eq_analysis_1b.plot_compare_profiles()

# %%
# Plot an assortment of physics parameters for the plamsa core
ax = eq_analysis_1.plot_eq_core_analysis()

# %%
# Key parameters as a table
eq_analysis_1.physics_info_table()

# %%
# PLot the equilibrium and reference equilibrium seperatrices
ax = eq_analysis_1.plot_compare_separatrix()

# %%
# PLot the equilibrium and reference equilibrium seperatrices
ax = eq_analysis_1b.plot_compare_separatrix()

# %%
# Plot a comparison of the input and reference equilibria.
# For eq_analysis_1 and 1b we chose to use diag_ops_1,
# which is a total psi diff plot.
eq_analysis_1.plot_compare_psi()

# %%
eq_analysis_1b.plot_compare_psi()

# %%
# For eq_analysis_2b we chose to use diag_ops_2, which is absolute psi diff
# for seperate plasma and coilset contruibutions. A mask has also been applied,
# so that only the values inside the reference LCFS are visable.
eq_analysis_2.plot_compare_psi()

# %%
# Plot the leg flux for a given divertor target and compare to reference.
ax = eq_analysis_1.plot_target_flux(
    target="lower_outer",
    target_coords=Coordinates({"x": [10, 11], "z": [-7.5, -7.5]}),
    vertical=False,
)

# %%
ax = eq_analysis_1.plot_target_flux(
    target="upper_outer",
    target_coords=Coordinates({"x": [10, 11], "z": [7.5, 7.5]}),
)

# %%
ax = eq_analysis_1.plot_target_flux(
    target="lower_inner",
    target_coords=Coordinates({"x": [5.5, 6.5], "z": [-7.5, -7.5]}),
)

# %%
ax = eq_analysis_1.plot_target_flux(
    target="upper_inner",
    target_coords=Coordinates({"x": [5.5, 6.5], "z": [7.5, 7.5]}),
)

# %% [markdown]
# #### MultiEqAnalysis
#
# Now we will look at the untilities avilable in the
# MultiEqAnalysis class for multiple equilibria.

# %%
# list of paths to equlibria to examine
paths = [masty_path, double_demoish_path, single_demoish_path]
# Correponding list of names for plt legends etc.
equilibrium_names = ["MASTy Eq", "DN DEMOish Eq", "SN DEMOish Eq"]
# Don't forget to make sure the corrcet cocos value is used
# if they are not all the same
multi_analysis = MultiEqAnalysis(
    paths, equilibrium_names=equilibrium_names, from_cocos=[3, 3, 7]
)

# %%
# The same physics info as for the EqAnalysis but for all listed equilibria.
multi_analysis.physics_info_table()

# %%
# Plot physics parameters for the plamsa core
ax = multi_analysis.plot_physics()

# %%
# Plot the noramlised profiles
ax = multi_analysis.plot_compare_profiles()

# %%
# Plot a selected flux surface from each equilibria
# Default if LCFS but can also have seperatricies...
ax = multi_analysis.plot_compare_flux_surfaces()

# %%
# ... or a flux surface with a user selected normalised psi value.
ax = multi_analysis.plot_compare_flux_surfaces(
    flux_surface=FluxSurfaceType.PSI_NORM, psi_norm=1.05
)

# %%
pfb_masty = Coordinates({
    "x": [1.75, 1.75, 0.0, 0.0, 1.75],
    "z": [-1.75, 1.75, 1.75, -1.75, -1.75],
})
pfb_demoish = Coordinates({
    "x": [14.5, 14.5, 5.75, 5.75, 14.5],
    "z": [-7.5, 7.5, 7.5, -7.5, -7.5],
})
ax = multi_analysis.plot_divertor_length_angle(
    plasma_facing_boundary_list=[pfb_masty, pfb_demoish, pfb_demoish]
)
