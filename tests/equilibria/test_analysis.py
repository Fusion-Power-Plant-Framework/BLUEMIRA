# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from pathlib import Path

from matplotlib.pyplot import Axes
from pandas import DataFrame

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.analysis import EqAnalysis, select_eq
from bluemira.equilibria.diagnostics import (
    EqDiagnosticOptions,
    EqSubplots,
    FixedOrFree,
    LCFSMask,
    PsiPlotType,
)
from bluemira.equilibria.equilibrium import Equilibrium, FixedPlasmaEquilibrium
from bluemira.equilibria.flux_surfaces import CoreResults
from bluemira.geometry.coordinates import Coordinates

TEST_PATH = get_bluemira_path("equilibria/test_data", subfolder="tests")
# DEMO-like-SN
single_demoish_path = Path(TEST_PATH, "eqref_OOB.json")
# DEMO-like-DN
double_demoish_path = Path(TEST_PATH, "DN-DEMO_eqref.json")


class TestEqAnalysis:
    """
    Tests for EqAnalysis.
    """

    @classmethod
    def setup_class(cls):
        cls.single_demoish_eq = select_eq(single_demoish_path, from_cocos=7)
        cls.single_demoish_eq_fixed = select_eq(
            single_demoish_path, from_cocos=7, fixed_or_free=FixedOrFree.FIXED
        )
        cls.double_demoish_eq = select_eq(double_demoish_path)
        cls.diag_ops_1 = EqDiagnosticOptions(
            psi_diff=PsiPlotType.PSI_ABS_DIFF,
            split_psi_plots=EqSubplots.XZ_COMPONENT_PSI,
            reference_eq=cls.single_demoish_eq,
            lcfs_mask=LCFSMask.IN,
        )
        cls.diag_ops_2 = EqDiagnosticOptions(
            psi_diff=PsiPlotType.PSI_ABS_DIFF,
            split_psi_plots=EqSubplots.XZ_COMPONENT_PSI,
            reference_eq=cls.single_demoish_eq_fixed,
            lcfs_mask=LCFSMask.IN,
        )
        cls.diag_ops_3 = EqDiagnosticOptions(
            psi_diff=PsiPlotType.PSI_ABS_DIFF,
            split_psi_plots=EqSubplots.XZ,
            reference_eq=cls.single_demoish_eq,
            lcfs_mask=LCFSMask.IN,
        )

    def test_select_eq(self):
        """Test correct class is returned."""
        assert isinstance(self.single_demoish_eq, Equilibrium)
        assert isinstance(self.single_demoish_eq_fixed, FixedPlasmaEquilibrium)

    def test_plotting(self):
        "Test plots return expected axes etc."
        # Compare free DN to free SN
        eq_analysis_1 = EqAnalysis(self.diag_ops_1, self.double_demoish_eq)
        # Compare free DN to fixed SN
        eq_analysis_2 = EqAnalysis(self.diag_ops_2, self.double_demoish_eq)
        # Compare free DN to free SN, with psi components splitting for plots
        eq_analysis_3 = EqAnalysis(self.diag_ops_3, self.double_demoish_eq)

        # Target for legflux plotting.
        target = "lower_outer"
        target_coords = Coordinates({"x": [10, 11], "z": [-7.5, -7.5]})

        plot_1 = eq_analysis_1.plot()
        plot_1b = eq_analysis_2.plot()
        plot_2 = eq_analysis_1.plot_field()
        plot_2b = eq_analysis_2.plot_field()
        plot_3 = eq_analysis_1.plot_profiles()
        plot_3b = eq_analysis_2.plot_profiles()
        plot_4 = eq_analysis_1.plot_equilibria_with_profiles()
        plot_4b = eq_analysis_2.plot_equilibria_with_profiles()
        plot_5 = eq_analysis_1.plot_eq_core_mag_axis()
        plot_6 = eq_analysis_1.plot_compare_profiles()
        plot_6b = eq_analysis_2.plot_compare_profiles()
        plot_7_res, plot_7_ax = eq_analysis_1.plot_eq_core_analysis()
        plot_8 = eq_analysis_1.physics_info_table()
        plot_9 = eq_analysis_1.plot_compare_separatrix()
        plot_9b = eq_analysis_2.plot_compare_separatrix()
        plot_10 = eq_analysis_1.plot_compare_psi()
        plot_10b = eq_analysis_2.plot_compare_psi()
        plot_11 = eq_analysis_3.plot_compare_psi()
        plot_12 = eq_analysis_1.plot_target_flux(target, target_coords)

        assert plot_1 is None
        assert isinstance(plot_2[0], Axes)
        assert len(plot_2) == 2
        assert plot_3 is None
        assert isinstance(plot_4[0], Axes)
        assert len(plot_4) == 2
        assert plot_5 is None
        assert plot_6 is None
        assert isinstance(plot_7_ax[0], Axes)
        assert isinstance(plot_7_res, CoreResults)
        assert len(plot_7_ax) == 18
        assert len(plot_7_res.__dict__.items()) == 17
        assert isinstance(plot_8, DataFrame)
        assert isinstance(plot_9, Axes)
        assert plot_10 is None
        assert plot_11 is None
        assert isinstance(plot_12, Axes)
        assert plot_1b is None
        assert isinstance(plot_2b[0], Axes)
        assert len(plot_2b) == 2
        assert plot_3b is None
        assert isinstance(plot_4b[0], Axes)
        assert len(plot_4b) == 2
        assert plot_6b is None
        assert isinstance(plot_9b, Axes)
        assert plot_10b is None
