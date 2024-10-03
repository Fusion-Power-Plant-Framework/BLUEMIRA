# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

from bluemira.equilibria.analysis import EqAnalysis, MultiEqAnalysis, select_eq
from bluemira.equilibria.diagnostics import EqDiagnosticOptions

# %pdb

# %%
ref_path = Path("../../tests/equilibria/test_data/SH_test_file.json")
ref_eq = select_eq(ref_path)
eq_path = Path("../../tests/equilibria/test_data/eqref_OOB.json")
eq = select_eq(eq_path, from_cocos=7)

# %%
diag_ops = EqDiagnosticOptions(
    psi_diff=True,
    split_psi_plots=False,
    reference_eq=ref_eq,
)

analysis = EqAnalysis(diag_ops, eq)

# %%
analysis.plot_equilibria_with_profiles()

# %%
analysis.plot_compare_psi()

# %%
analysis.plot_compare_profiles()

# %%
p1 = ref_path
p2 = eq_path

equilibrium_names = ["MASTy Eq", "DEMOish Eq"]

multi_analysis = MultiEqAnalysis(
    [p1, p2], equilibrium_names=equilibrium_names, from_cocos=[3, 7]
)

# %%
pdf = multi_analysis.coilset_info_table()

# %%
pdf.style.set_caption("Current (MA)")

# %%
multi_analysis.plot_compare_profiles()
