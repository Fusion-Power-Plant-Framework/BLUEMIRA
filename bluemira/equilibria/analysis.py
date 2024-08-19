# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Equilibria and Equilibria optimisation analysis tools"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bluemira.equilibria.diagnostics import EqDiagnosticOptions
    from bluemira.equilibria.equilibrium import MHDState
    from bluemira.equilibria.optimisation.problem.base import CoilsetOptimisationProblem

from bluemira.base.error import BluemiraError
from bluemira.equilibria.diagnostics import FixedOrFree, CSData
from bluemira.equilibria.equilibrium import Equilibrium, FixedPlasmaEquilibrium
from bluemira.equilibria.plotting import EquilibriumComparisonPostOptPlotter
from bluemira.equilibria.profiles import CustomProfile
from bluemira.utilities.tools import is_num


### Functions used in multiple toolboxes ###
def select_eq(
        file_path,
        fixed_or_free=FixedOrFree.FREE,
        dummy_coils=None,
        from_cocos=3,
        to_cocos=3,
        qpsi_sign=-1
        ):
    """
    Return an Equilibrium or FixedPlasmaEquilibrium object given a particular file name.

    Cocos indecis and qpsi sign are set to Bluemira Defaults unless specified.

    Parameters
    ----------
    file_path:
        file path to chosen equilibria
    fixed_or_free:
        wheather or not it is for a fixed plasma boundary
    dummy_coils:
        coilset if none in equilibria file
        (a default coilset is used, and a warning message prints if none is provided)
    from_cocos:
        The COCOS index of the EQDSK file. Used when the determined
        COCOS is ambiguous. Will raise if given and not one of
        the determined COCOS indices.
    to_cocos:
        The COCOS index to convert the EQDSK file to.
    qpsi_sign:
        The sign of the qpsi, required for identification
        when qpsi is not present in the file.

    Returns
    -------
        Equilibrium or FixedPlasmaEquilibrium
    """
    if fixed_or_free == FixedOrFree.FREE:
        return Equilibrium.from_eqdsk(
            file_path, from_cocos=from_cocos, to_cocos=to_cocos, qpsi_sign=qpsi_sign
            )
    if dummy_coils is not None:
        return FixedPlasmaEquilibrium.from_eqdsk(
            file_path,
            user_coils=dummy_coils,
            from_cocos=from_cocos,
            to_cocos=to_cocos,
            qpsi_sign=qpsi_sign,
            )
    return FixedPlasmaEquilibrium.from_eqdsk(
        file_path,
        from_cocos=from_cocos,
        to_cocos=to_cocos,
        qpsi_sign=qpsi_sign,
        )


class EqAnalysis:
    """
    Equilibria analysis toolbox for selected Equilibrium.

    Can  compare selected equilibrium to a refernce equilibrium.
    The input and reference equilibria be different types (i.e, fixed or free), 
    and have differentgrid sizes and grid resolutions.

    Cocos indecis and qpsi sign are set to Bluemira Defaults unless specified.

    Parameters
    ----------
    diag_ops:
        Diagnostic plotting options, containg reference equilibria information.
    eq: 
        chosen Equilibrium 
    file_path:
        file path to chosen equilibrium
    fixed_or_free:
        fixed or free plasma boundary
    dummy_coils:
        coilset if none in equilibria file
        (a default coilset is used, and a warning message prints if none is provided)
    from_cocos:
        The COCOS index of the EQDSK file. Used when the determined
        COCOS is ambiguous. Will raise if given and not one of
        the determined COCOS indices.
    to_cocos:
        The COCOS index to convert the EQDSK file to.
    qpsi_sign:
        The sign of the qpsi, required for identification
        when qpsi is not present in the file.

    """
    def __init__(
            self, 
            diag_ops: EqDiagnosticOptions,
            eq: MHDState | None = None,
            file_path: str | None = None,
            fixed_or_free = FixedOrFree.FREE, 
            dummy_coils=None,
            from_cocos=3,
            to_cocos=3,
            qpsi_sign=-1,
            ):
        
        self.diag_ops = diag_ops
        self.fixed_or_free = FixedOrFree.FREE, 
        self.dummy_coils=None,
        self.from_cocos=3,
        self.to_cocos=3,
        self.qpsi_sign=-1
        
        if eq:
            self._eq = eq
            self._profiles = eq.profiles
        elif file_path:
            self.file_path = file_path
            self._eq = select_eq(
                file_path,
                fixed_or_free=fixed_or_free,
                dummy_coils=dummy_coils,
                from_cocos=from_cocos,
                to_cocos=to_cocos,
                qpsi_sign=qpsi_sign
                )
            self._profiles = CustomProfile.from_eqdsk_file(
                file_path,
                from_cocos=from_cocos,
                to_cocos=to_cocos,
                qpsi_sign=qpsi_sign
                )
        else:
            BluemiraError("Please provide either an Equilibrium object or eqdsk file path as an input.")  

        if diag_ops.reference_eq:
            self.reference_profiles = self.reference_eq.profiles

    def plot(self):
        """Plot equilibria"""
        return self._eq.plot()
    
    def plot_profiles(self):
        """Plot profiles"""
        #FIXME
        pass
    
    def plot_equilibria_with_profiles(
            self,
            title=None,
            ax=None,
            show=True
            ):
        """
        Plot equilibria alongside profiles.
        
        Parameters
        ----------
        title:
            Title to be added at top of figure
        ax:
            List of Matplotlib Axes objects set by user
        show:
            Whether or not to display the plot

        Returns
        -------
        ax1, ax2:
            The Matplotlib Axes objects for each subplot.

        """
        n_ax = 2
        if ax is not None:
            if len(ax) != n_ax:
                raise BluemiraError(
                    f"There are 2 subplots, you have provided settings for {len(ax)}."
                )
            ax1, ax2 = ax[0], ax[1]
        else:
            _, (ax1, ax2) = plt.subplots(1, 2)

        self._eq.plot(ax=ax1)
        ax1.set_title(f"R_0 = {self._profiles.R_0} m \n B_0 = {self._profiles._B_0} T \n I_p = {self._profiles.I_p / 1e6:.2f} MA \n")
        self._profiles.plot(ax=ax2)
        ax2.set_title("Profiles")
        if title is not None:
            plt.suptitle(title)
        if show:
            plt.show()
        return ax1, ax2
    
    def plot_compare_psi(self, ax=None, mask_type=None):
        """FIXME"""
        if self.diag_ops.reference_eq is None:
            BluemiraError("Please provide a reference Equilibrium object or Reference eqdsk file path in EqDiagnosticOptions.")
            return 
        return EquilibriumComparisonPostOptPlotter(
            equilibrium=self.eq,
            diag_ops=self.diag_ops,
            ax=ax
            ).plot_compare_psi(mask_type=mask_type)
    
    def plot_compare_profiles(self, equilibrium_names=None, reference_profile_sign=None, ax=None, diff=True):
        """
        Plot equilibria reference and input profiles. The diff option can be used to plot
        the difference beween the reference and input profiles on the same axis.

        Parameters
        ----------
        equilibrium_names:
            Names of the equilibia to be used as plot labels.
        reference_profile_sign:
            To be used with the diff option if the profile convention of the compared
            equilibria is different.
        ax:
            Matplotlib axes object.
        diff:
            If two equilibria are being compared then we have the option of also
            plotting the difference between them.
        """
        if self.diag_ops.reference_eq is None:
            BluemiraError("Please provide a reference Equilibrium object or Reference eqdsk file path in EqDiagnosticOptions.")
            return
        if equilibrium_names is None:
            equilibrium_names = ["Eq_reference", "Eq_input"]

        n_prof = 5
        if reference_profile_sign is None:
            reference_profile_sign = np.ones(n_prof)
        elif is_num(reference_profile_sign):
            reference_profile_sign *= np.ones(n_prof)
        elif len(reference_profile_sign) != n_prof:
            raise ValueError("profile_sign length not equal to 5.")

        shape_ax = (2, 3)
        if ax is not None:
            if np.shape(ax) != shape_ax:
                raise BluemiraError(
                    f"Subplot shape is (2,3), you have provided settings for {
                        np.shape(ax)
                        }."
                )
        else:
            _, ax = plt.subplots(2, 3)

        ref_profs = [
            self.reference_profiles.pprime, 
            self.reference_profiles.ffprime, 
            self.reference_profiles.fRBpol,
            self.reference_profiles.pressure,
            self.reference_profiles.shape,
            ]
        profs = [
            self._profiles.pprime, 
            self._profiles.ffprime, 
            self._profiles.fRBpol,
            self._profiles.pressure,
            self._profiles.shape,
        ]
        ax_titles = ["pprime", "ffprime", "fRBpol","pressure","shape"]
        axs = [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1], ax[0, 2]]

        x = np.linspace(0, 1, 50)

        for ref_prof, prof, sign, a, a_title in (
            ref_profs, 
            profs,
            reference_profile_sign,
            ax_titles, 
            axs,
        ):
            a.plot(x, sign * ref_prof,marker=".", label=self.equilibrium_names[0])
            a.plot(x, prof.pprime, marker=".", label=self.equilibrium_names[1])
            if diff:
                a.plot(x, sign * ref_prof - prof, marker=".", label=self.equilibrium_names[0] + " - " + self.equilibrium_names[1])
            a.set_title(a_title)
            a.legend(loc="best")
        
        ax[1, 2].axis("off")
        plt.suptitle("Profile Comparison")
        plt.show()

            
class COPAnalysis(EqAnalysis):
    """Coilset Optimisation Problem analysis toolbox"""

    def __init__(self, cop: CoilsetOptimisationProblem):
        super().__init__(cop.eq)
        self._cop = cop


class MultiEqAnalysis:
    """
    Equilibria analysis toolbox for multiple Equilibrium.
    
    Parameters
    ----------
    equilibrium_paths:
        List of file paths for chosen equilibria.
    fixed_or_free:
        wheather or not it is for a fixed plasma boundary
    equilibrium_names:
        Text to be used in table and plot labels. Default to Eq_1, Eq_2 ... if not chosen.
    dummy_coils:
        coilset if none in equilibria file
        (a default coilset is used, and a warning message prints if none is provided.
    from_cocos:
        The COCOS index of the EQDSK file. Used when the determined
        COCOS is ambiguous. Will raise if given and not one of
        the determined COCOS indices.
    to_cocos:
        The COCOS index to convert the EQDSK file to.
    qpsi_sign:
        The sign of the qpsi, required for identification
        when qpsi is not present in the file.
    """

    def __init__(
            self,
            equilibrium_paths,
            fixed_or_free=FixedOrFree.FREE,
            equilibrium_names=None,
            dummy_coils=None,
            from_cocos=3,
            to_cocos=3,
            qpsi_sign=-1
    ):
        self.equilibrium_paths = equilibrium_paths

        if not isinstance(fixed_or_free, Iterable):
            self.fixed_or_free = [fixed_or_free] * len(equilibrium_paths)
        elif len(fixed_or_free) != len(equilibrium_paths):
            raise ValueError(
                "FixedOrFree list length not equal to the number of equilibria."
                )
        else:
            self.fixed_or_free = fixed_or_free

        if equilibrium_names is None:
            self.equilibrium_names = [
                "Eq_" + str(x) for x in range(1, len(equilibrium_paths) + 1)
                ]
        elif len(equilibrium_names) != len(equilibrium_paths):
            raise ValueError(
                "equilibrium_names length not equal to the number of equilibria."
                )
        else:
            self.equilibrium_names = equilibrium_names

        if not isinstance(dummy_coils, Iterable):
            self.dummy_coils = [dummy_coils] * len(equilibrium_paths)
        elif len(dummy_coils) != len(equilibrium_paths):
            raise ValueError(
                "dummy_coils list length not equal to the number of equilibria."
                )
        else:
            self.dummy_coils = dummy_coils

        if is_num(from_cocos):
            from_cocos = np.ones(len(equilibrium_paths)) * from_cocos
        if is_num(to_cocos):
            to_cocos = np.ones(len(equilibrium_paths)) * to_cocos
        if is_num(qpsi_sign):
            qpsi_sign = np.ones(len(equilibrium_paths)) * qpsi_sign

        self.from_cocos = from_cocos
        self.to_cocos = to_cocos
        self.qpsi_sign = qpsi_sign

        self.profiles = self.profile_dictionary()

        self.equilibrium = []
        for (file, eq_type, dummy, fc, tc, qs) in zip(
            self.equilibrium_paths,
            self.fixed_or_free,
            self.dummy_coils,
            self.from_cocos,
            self.to_cocos,
            self.qpsi_sign,
            strict=False
            ):
            self.equilibrium.append(select_eq(
                file,
                fixed_or_free=eq_type,
                dummy_coils=dummy,
                from_cocos=fc,
                to_cocos=tc,
                qpsi_sign=qs
                ))
            
    def profile_dictionary(self):
        """
        Create a dictionary of profile information.
        To be used when comparing multiple equilibria but will work for a signle equilibrium.

        Cocos indecis and qpsi sign are set to Bluemira Defaults unless specified.

        The user can spesify from_cocos, to_cocos and qpsi_sign using:
            - lists if the equilibria have different values
            - single values if equilibria all have the same convention

        Return:
            Profile Dictionary (profile type: list of profile objects for each equilibria)
        """
        if not isinstance(self.equilibrium_paths, Iterable):
            prof = CustomProfile.from_eqdsk_file(
                self.equilibrium_paths,
                from_cocos=self.from_cocos,
                to_cocos=self.to_cocos,
                qpsi_sign=self.qpsi_sign
            )
            return {
                "pprime": prof.pprime,
                "ffprime": prof.ffprime,
                "fRBpol": prof.fRBpol,
                "pressure": prof.pressure,
                "shape": prof.shape,
            }

        if is_num(from_cocos):
            from_cocos = np.ones(len(self.equilibrium_paths)) * self.from_cocos
        if is_num(to_cocos):
            to_cocos = np.ones(len(self.equilibrium_paths)) * self.to_cocos
        if is_num(qpsi_sign):
            qpsi_sign = np.ones(len(self.equilibrium_paths)) * self.qpsi_sign

        prof_dict = {
            "pprime": [],
            "ffprime": [],
            "fRBpol": [],
            "pressure": [],
            "shape": []
        }
        for eq_path, fc, tc, q in zip(
            self.equilibrium_paths, self.from_cocos, self.to_cocos, self.qpsi_sign, strict=False
            ):
            prof = CustomProfile.from_eqdsk_file(
                eq_path,
                from_cocos=fc,
                to_cocos=tc,
                qpsi_sign=q
                )
            prof_dict["pprime"].append(prof.pprime)
            prof_dict["ffprime"].append(prof.ffprime)
            prof_dict["fRBpol"].append(prof.fRBpol)
            prof_dict["pressure"].append(prof.pressure)
            prof_dict["shape"].append(prof.shape)
        return prof_dict
    
    def physics_dictionary(self):
        """
        Create a list of dictionaries with the physics information
        from all listed Equilbria.

        Returns
        -------
        dict_list:
            Pandas dataframe with equilibria physics information.

        """
        dict_list = []
        for eq in self.equilibrium:
            plasma_dict = eq.analyse_plasma()
            dict_list.append(plasma_dict)
        return dict_list

    def physics_info_table(self):
        """
        Create a Pandas dataframe with the the physics information
        from all listed Equilbria.
        Not for use with FixedPlasmaEquilibrium.

        Returns
        -------
        dataframe:
            Pandas dataframe with equilibria physics information.

        """
        dict_list = self.physics_dictionary()
        pd.set_option("display.float_format", "{:.2f}".format)
        dataframe = pd.DataFrame(dict_list).T
        dataframe.columns = self.equilibrium_names
        return dataframe

    def coilset_dictionary(self, value=CSData.CURRENT):
        """
        Create a list of dictionaries with the coilset information
        from all listed Equilbria.
        Not for use with FixedPlasmaEquilibrium.

        Returns
        -------
        dict_list:
            Pandas dataframe with equilibria physics information.

        """
        dict_list = []
        for eq, fx in zip(self.equilibrium, self.fixed_or_free, strict=False):
            if fx != FixedOrFree.FIXED:
                coilset_dict = {}
                for coil in eq.coilset._coils:
                    if value == CSData.CURRENT:
                        coilset_dict[coil.name] = coil.current
                    elif value == CSData.XLOC:
                        coilset_dict[coil.name] = coil.x
                    elif value == CSData.ZLOC:
                        coilset_dict[coil.name] = coil.z
            dict_list.append(coilset_dict)
        return dict_list

    def coilset_info_table(self, value=CSData.CURRENT):
        """
        Create a Pandas dataframe with the the coilset information
        from all listed Equilbria.
        Not for use with FixedPlasmaEquilibrium.

        Returns
        -------
        dataframe:
            Pandas dataframe with equilibria coilset information.

        """
        dict_list = self.coilset_dictionary(value)
        pd.set_option("display.float_format", "{:.2f}".format)
        dataframe = pd.DataFrame(dict_list).T
        dataframe.columns = self.equilibrium_names
        return dataframe
    
    def plot_prof(
            ax,
            profiles,
            title,
            equilibrium_names,
            ):
        """
        Plot equilibria profiles from list.

        Parameters
        ----------
        ax:
            Matplotlib axes object.
        profiles:
            List of profile objects.
        title:
            Text to be printed above the axis.
        equilibrium_names:
            Names of the equilibia to be used as plot labels.
        """
    
        x = np.linspace(0, 1, 50)
        for p, name in zip(profiles, equilibrium_names, strict=False):
            ax.plot(x, p(x), marker=".", label=name)
            ax.legend(loc="best")
            ax.set_title(title)

    def plot_compare_profiles(
            self,
            ax=None,
            title=None,
            ):
        """
        Plot the profiles of all the listed equilibria.

        Parameters
        ----------
        ax:
            List of Matplotlib Axes objects set by user
        title:
            Tesxt to be added at the top of the figure
        show:
            Whether or not to display the plot

        Returns
        -------
        ax1, ax2, ax3, ax4, ax5, ax6:
            The Matplotlib Axes objects for each subplot.

        """
        shape_ax = (2, 3)
        if ax is not None:
            if np.shape(ax) != shape_ax:
                raise BluemiraError(
                    f"Subplot shape is (2,3), you have provided settings for {
                        np.shape(ax)
                        }."
                )
        else:
            _, ax = plt.subplots(2, 3)

        ax_list = [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1], ax[0, 2]]

        for a, (key, profs) in zip(ax_list, self.profiles.items(), strict=False):
            self.plot_prof(a, profs, title=key, equilibrium_names=self.equilibrium_names)
        ax[1, 2].axis("off")

        if title is not None:
            plt.suptitle(title)
        plt.show()
        return ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1], ax[0, 2], ax[1, 2]