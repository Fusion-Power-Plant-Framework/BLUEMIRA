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
from matplotlib.gridspec import GridSpec

if TYPE_CHECKING:
    from bluemira.equilibria.diagnostics import EqDiagnosticOptions
    from bluemira.equilibria.equilibrium import MHDState
    from bluemira.equilibria.flux_surfaces import CoreResults
    from bluemira.equilibria.optimisation.problem.base import CoilsetOptimisationProblem
from itertools import cycle

from bluemira.base.error import BluemiraError
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.diagnostics import (
    CSData,
    DivLegsToPlot,
    FixedOrFree,
    FluxSurfaceType,
)
from bluemira.equilibria.equilibrium import Equilibrium, FixedPlasmaEquilibrium
from bluemira.equilibria.find import find_flux_surface_through_point, find_flux_surfs
from bluemira.equilibria.find_legs import (
    LegFlux,
    get_legs_length_and_angle,
)
from bluemira.equilibria.flux_surfaces import analyse_plasma_core
from bluemira.equilibria.plotting import CorePlotter, EquilibriumComparisonPostOptPlotter
from bluemira.equilibria.profiles import CustomProfile
from bluemira.geometry.coordinates import Coordinates
from bluemira.utilities.tools import is_num


# Functions used in multiple toolboxes ###
def select_eq(
    file_path,
    fixed_or_free=FixedOrFree.FREE,
    dummy_coils=None,
    from_cocos=3,
    to_cocos=3,
    qpsi_sign=-1,
) -> FixedPlasmaEquilibrium | Equilibrium:
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
            file_path,
            from_cocos=from_cocos,
            user_coils=dummy_coils,
            to_cocos=to_cocos,
            qpsi_sign=qpsi_sign,
        )
    return FixedPlasmaEquilibrium.from_eqdsk(
        file_path,
        from_cocos=from_cocos,
        to_cocos=to_cocos,
        qpsi_sign=qpsi_sign,
    )


def get_leg_flux_info(
    eq_list,
    n_layers=10,
    dx_off=0.1,
    plasma_facing_boundary_list=None,
    legs_to_plot=DivLegsToPlot.ALL,
) -> tuple[dict[float | np.ndarray], dict[float | np.ndarray], dict[float | np.ndarray]]:
    """
    Get the divertor leg length and grazing angle (used in divertor comparison plotting).

    Parameters
    ----------
    eq_list:
        List of Equilibrium objects to compare
    n_layers:
        Number of flux surfaces to extract for each leg
    dx_off:
        Total span in radial space of the flux surfaces to extract
    plasma_facing_boundary_list:
        List of associated plasma facing boundary coordinates
    legs_to_plot:
        Which legs to plot i.e, upper, lower or all.

    Returns
    -------
    lengths:
        Dictionary of connection length values for each named leg.
    angles:
        Dictionary of grazing values for each named leg.

    Raises
    ------
    BluemiraError:
        If leg pair is chosen that does not exist in chosen eq.

    """
    lengths = []
    angles = []
    if plasma_facing_boundary_list is None:
        plasma_facing_boundary_list = [None] * len(eq_list)
    for eq, pfb in zip(eq_list, plasma_facing_boundary_list, strict=False):
        legflux = LegFlux(eq)
        legs = legflux.get_legs(n_layers=n_layers, dx_off=dx_off)
        lgth, ang = get_legs_length_and_angle(eq, legs, pfb)
        if legs_to_plot in DivLegsToPlot.PAIR:
            location = "lower" if legs_to_plot is DivLegsToPlot.LW else "upper"
            if f"{location}_inner" not in lgth:
                raise BluemiraError(  # noqa: DOC501
                    f"One of your chosen equilibria does not have {location}_inner legs."
                )
            if f"{location}_outer" not in lgth:
                raise BluemiraError(
                    f"One of your chosen equilibria does not have {location}_outer legs."
                )
            lgth = {k: lgth[k] for k in (f"{location}_inner", f"{location}_outer")}
            ang = {k: ang[k] for k in (f"{location}_inner", f"{location}_outer")}
        else:
            # sort alphabetically
            lgth = {k: lgth[k] for k in sorted(lgth)}
            ang = {k: ang[k] for k in sorted(ang)}
        lengths.append(lgth)
        angles.append(ang)
    return lengths, angles


def get_target_flux(eq, target, target_coords, n_layers, vertical=False):  # noqa: FBT002
    """
    Get a selection flux surfaces that cross the divertor target.

    eq:
        chosen Equilibrium
    target:
        Location of divertor target, lower_outer, lower_inner, upper_outer or upper_inner
    target_coords:
        Corrinates of divertor target
    n_layers:
        Number of flux surfaces to extract for each leg
    vertical:
        Set to True if using a vertical target.
        This boolean determines whether to get flux surfaces across
        the target x-range (horizontal) or z-range (vertical).

    Raises
    ------
    BluemiraError:
        If the target is set to vertical or horizontal incorectly.

    """
    if eq._o_points is None:
        _, _ = eq.get_OX_points()

    if not vertical:
        x = np.min(target_coords.x)
        z = target_coords.z[np.argmin(target_coords.x)]
        target_size = np.abs(np.max(target_coords.x) - np.min(target_coords.x))
        if target_size == 0:
            raise BluemiraError(  # noqa: DOC501
                "No x-range found for target coords,"
                " perhaps you are using a vertical target (set vertical=True)."
            )
        target_offsets = np.linspace(0, target_size, n_layers)[1:]
        dx = x + target_offsets
        dz = np.ones(n_layers) * z
    else:
        x = target_coords.z[np.argmin(target_coords.z)]
        z = np.min(target_coords.z)
        target_size = np.abs(np.max(target_coords.z) - np.min(target_coords.z))
        if target_size == 0:
            raise BluemiraError(
                "No z-range found for target coords,"
                " perhaps you are using a horizontal target (set vertical=False)."
            )
        target_offsets = np.linspace(0, target_size, n_layers)[1:]
        dx = np.ones(n_layers) * x
        dz = z + target_offsets

    fs_list = []
    for x, z in zip(dx, dz, strict=False):
        fs_x, fs_z = find_flux_surface_through_point(
            eq.x,
            eq.z,
            eq.psi(),
            x,
            z,
            eq.psi(x, z),
        )
        # Only need to plot from midpoint
        select_idx = (fs_z >= 0) if target.find("lower") == -1 else (fs_z <= 0)
        if all(fs_z >= 0) or all(fs_z <= 0):
            select_idx_x = (
                (fs_x <= eq._x_points[0].x)
                if target.find("inner") != -1
                else (fs_x >= eq._x_points[0].x)
            )
            select_idx &= select_idx_x
        fs = Coordinates({"x": fs_x[select_idx], "z": fs_z[select_idx]})
        fs_list.append(fs)

    return fs_list


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

    Raises
    ------
    BluemiraError:
        If no equilibrium is chosen as input.

    """

    def __init__(
        self,
        diag_ops: EqDiagnosticOptions,
        eq: MHDState | None = None,
        file_path: str | None = None,
        fixed_or_free=FixedOrFree.FREE,
        dummy_coils=None,
        from_cocos=3,
        to_cocos=3,
        qpsi_sign=-1,
    ):
        self.diag_ops = diag_ops
        self.fixed_or_free = FixedOrFree.FREE
        self.dummy_coils = (None,)
        self.from_cocos = (3,)
        self.to_cocos = (3,)
        self.qpsi_sign = -1

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
                qpsi_sign=qpsi_sign,
            )
            self._profiles = CustomProfile.from_eqdsk_file(
                file_path, from_cocos=from_cocos, to_cocos=to_cocos, qpsi_sign=qpsi_sign
            )
        else:
            BluemiraError(
                "Please provide either an Equilibrium object "
                "or eqdsk file path as an input."
            )

        if diag_ops.reference_eq:
            self.reference_profiles = self.diag_ops.reference_eq.profiles

    def plot(self):
        """Plot equilibria"""
        self._eq.plot()
        plt.show()

    def plot_profiles(self):
        """Plot profiles"""
        self._profiles.plot()
        plt.show()

    def plot_eq_core_analysis(self) -> CoreResults:
        """
        Plot characteristics of the plasma core and return results.
        Currently only works for free boundary equilibria.

        Returns
        -------
        :
            Dataclass for core results.

        Raises
        ------
        BluemiraError:
            If the equilibrium is fixed boundary.

        """
        if self.fixed_or_free is FixedOrFree.FIXED:
            raise BluemiraError(  # noqa: DOC501
                "This function can only be used for Free Boundary Equilbria."
            )
        return self._eq.analyse_core()

    def plot_eq_core_mag_axis(self):
        """
        Plot a 1-D section through the magnetic axis.
        Currently only works for free boundary equilibria.

        Raises
        ------
        BluemiraError:
            If the equilibrium is fixed boundary.

        """
        if self.fixed_or_free is FixedOrFree.FIXED:
            raise BluemiraError(  # noqa: DOC501
                "This function can only be used for Free Boundary Equilbria."
            )
        self._eq.plot_core()

    def physics_info_table(self, equilibrium_name="Eq_input"):
        """
        Create a Pandas dataframe with the physics information
        from the Equilbria of interest.
        Not for use with FixedPlasmaEquilibrium.

        Returns
        -------
        dataframe:
            Pandas dataframe with summary of physics information.

        Raises
        ------
        BluemiraError:
            If the equilibrium is fixed boundary.

        """
        if self.fixed_or_free is FixedOrFree.FIXED:
            raise BluemiraError(  # noqa: DOC501
                "This function can only be used for Free Boundary Equilbria."
            )

        summary_dict = [self._eq.analyse_plasma()]
        pd.set_option("display.float_format", "{:.2f}".format)
        dataframe = pd.DataFrame(summary_dict).T
        dataframe.columns = [equilibrium_name]
        return dataframe

    def plot_equilibria_with_profiles(self, title=None, ax=None, show=True):  # noqa: FBT002
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

        Raises
        ------
        BluemiraError:
            if the wrong number of axes is provided

        """
        n_ax = 2
        if ax is not None:
            if len(ax) != n_ax:
                raise BluemiraError(  # noqa: DOC501
                    f"There are 2 subplots, you have provided settings for {len(ax)}."
                )
            ax1, ax2 = ax[0], ax[1]
        else:
            _, (ax1, ax2) = plt.subplots(1, 2)

        self._eq.plot(ax=ax1)
        ax1.set_title(
            f"R_0 = {self._profiles.R_0} m \n "
            f"B_0 = {self._profiles._B_0} T \n "
            f"I_p = {self._profiles.I_p / 1e6:.2f} MA \n"
        )
        self._profiles.plot(ax=ax2)
        ax2.set_title("Profiles")
        if title is not None:
            plt.suptitle(title)
        if show:
            plt.show()
        return ax1, ax2

    def plot_compare_separatrix(
        self,
        title=None,
        ax=None,
        show=True,  # noqa: FBT002
        equilibrium_name="Eq_input",
    ):
        """
        Plot separatrices.
        N.B. Plots LCFS if a fixed boundary is used.

        Parameters
        ----------
        title:
            Title to be added at top of figure
        ax:
            Matplotlib Axes objects set by user
        show:
            Whether or not to display the plot
        equilibrium_name:
            Name used in plot label

        Returns
        -------
        ax:
            Matplotlib Axes object

        """
        label = equilibrium_name + " LCFS"
        if ax is None:
            _, ax = plt.subplots()

        eq_fs = (
            self._eq.get_separatrix()
            if self.fixed_or_free == FixedOrFree.FREE
            else self._eq.get_LCFS()
        )

        if self.diag_ops.reference_eq == FixedOrFree.FREE:
            ref_eq_fs = self.diag_ops.reference_eq.get_separatrix()
        else:
            ref_eq_fs = self.diag_ops.reference_eq.get_LCFS()

        if isinstance(eq_fs, list):
            ax.plot(eq_fs[0].x, eq_fs[0].z, color="red", label=label)
            ax.plot(eq_fs[1].x, eq_fs[1].z, color="red")
        else:
            ax.plot(eq_fs.x, eq_fs.z, color="red", label=label)

        if isinstance(ref_eq_fs, list):
            ax.plot(
                ref_eq_fs[0].x,
                ref_eq_fs[0].z,
                color="blue",
                linestyle="--",
                label="Reference LCFS",
            )
            ax.plot(ref_eq_fs[1].x, ref_eq_fs[1].z, color="blue", linestyle="--")
        else:
            ax.plot(
                ref_eq_fs.x,
                ref_eq_fs.z,
                color="blue",
                linestyle="--",
                label="Reference LCFS",
            )

        ax.legend(loc="best")
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$z$ [m]")

        if title is not None:
            plt.suptitle(title)
        if show:
            plt.show()
        return ax

    def plot_compare_psi(self, ax=None):
        """
        Plot Psi comparision.

        Parameters
        ----------
        ax:
            List of Matplotlib Axes objects set by user
        mask_type:
            Type of masking to be used on plot

        Raises
        ------
        BluemiraError:
            if no refernce equilibrium is provided

        """
        if self.diag_ops.reference_eq is None:
            BluemiraError(
                "Please provide a reference Equilibrium object "
                "or Reference eqdsk file path in EqDiagnosticOptions."
            )
            return None
        return EquilibriumComparisonPostOptPlotter(
            equilibrium=self._eq, diag_ops=self.diag_ops, ax=ax
        ).plot_compare_psi()

    def plot_compare_profiles(
        self,
        equilibrium_names=None,
        reference_profile_sign=None,
        ax=None,
        diff=True,  # noqa: FBT002
    ):
        """
        Plot equilibria reference and input profiles. The diff option can be used to plot
        the difference beween the reference and input profiles on the same axis.

        Parameters
        ----------
        equilibrium_names:
            Names of the equilibia to be used as plot labels.
            Reference is listed first.
        reference_profile_sign:
            To be used with the diff option if the profile convention of the compared
            equilibria is different.
        ax:
            Matplotlib axes object.
        diff:
            If two equilibria are being compared then we have the option of also
            plotting the difference between them.

        Raises
        ------
        BluemiraError:
            if no reference equilibrium or equilibrium file path is provided
        ValueError:
            if the profile sign array provided is an incorrect length
        """
        if self.diag_ops.reference_eq is None:
            BluemiraError(
                "Please provide a reference Equilibrium object "
                "or Reference eqdsk file path in EqDiagnosticOptions."
            )
            return
        if equilibrium_names is None:
            equilibrium_names = ["Eq_reference", "Eq_input"]

        n_prof = 5
        if reference_profile_sign is None:
            reference_profile_sign = np.ones(n_prof)
        elif is_num(reference_profile_sign):
            reference_profile_sign *= np.ones(n_prof)
        elif len(reference_profile_sign) != n_prof:
            raise ValueError("profile_sign length not equal to 5.")  # noqa: DOC501

        shape_ax = (2, 3)
        if ax is not None:
            if np.shape(ax) != shape_ax:
                raise BluemiraError(  # noqa: DOC501
                    "Subplot shape is (2,3), "
                    f"you have provided settings for {np.shape(ax)}."
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
        ax_titles = ["pprime", "ffprime", "fRBpol", "pressure", "shape"]
        axes = [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1], ax[0, 2]]

        x = np.linspace(0, 1, 50)

        for ref_prof, prof, sign, axs, a_title in zip(
            ref_profs, profs, reference_profile_sign, axes, ax_titles, strict=False
        ):
            axs.plot(x, sign * ref_prof(x), marker=".", label=equilibrium_names[0])
            axs.plot(x, prof(x), marker=".", label=equilibrium_names[1])
            if diff:
                axs.plot(
                    x,
                    sign * ref_prof(x) - prof(x),
                    marker=".",
                    label=equilibrium_names[0] + " - " + equilibrium_names[1],
                )
            axs.set_title(a_title)
            axs.legend(loc="best")

        ax[1, 2].axis("off")
        plt.suptitle("Profile Comparison")
        plt.show()

    def plot_target_flux(
        self,
        target: str,
        target_coords: Coordinates,
        n_layers=10,
        vertical=False,  # noqa: FBT002
        equilibrium_name="Eq_input",
        ax=None,
        show=True,  # noqa: FBT002
    ):
        """
        Plot the divertor leg flux. Will find flux suraces at evenely spaced points
        in x- or z- direction for a set of target coordinates.

        Parameters
        ----------
        target:
            Location of divertor target, i.e.,
            lower_outer, lower_inner, upper_outer or upper_inner
        target_coords:
            Corrinates of divertor target
        n_layers:
            Number of flux surfaces to extract for each leg
        vertical:
            Set to True if using a vertical target.
            This boolean determines whether to get flux surfaces across
            the target x-range (horizontal) or z-range (vertical).
        ax:
            Matplotlib Axes objects set by user
        show:
            Whether or not to display the plot

        Returns
        -------
        ax:
            Matplotlib Axes object

        """
        label = equilibrium_name + " LCFS"
        if ax is None:
            _, ax = plt.subplots()

        if self.diag_ops.reference_eq is not None:
            ref_target_flux = get_target_flux(
                self.diag_ops.reference_eq, target, target_coords, n_layers, vertical
            )

            if isinstance(ref_target_flux, list):
                for i in np.arange(len(ref_target_flux)):
                    if i == 0:
                        ax.plot(
                            ref_target_flux[i].x,
                            ref_target_flux[i].z,
                            color="blue",
                            linestyle="--",
                            label="Reference LCFS",
                        )
                    else:
                        ax.plot(
                            ref_target_flux[i].x,
                            ref_target_flux[i].z,
                            color="blue",
                            linestyle="--",
                        )
            elif ref_target_flux is not None:
                ax.plot(
                    ref_target_flux.x,
                    ref_target_flux.z,
                    color="blue",
                    linestyle="--",
                    label="Reference LCFS",
                )

        target_flux = get_target_flux(
            self._eq, target, target_coords, n_layers, vertical
        )

        if isinstance(target_flux, list):
            for i in np.arange(len(target_flux)):
                if i == 0:
                    ax.plot(target_flux[i].x, target_flux[i].z, color="red", label=label)
                else:
                    ax.plot(target_flux[i].x, target_flux[i].z, color="red")
        elif target_flux is not None:
            ax.plot(target_flux.x, target_flux.z, color="red", label=label)
        else:
            bluemira_warn("No flux found crossing target coordinates.")

        ax.plot(target_coords.x, target_coords.z, color="black", linewidth=5)

        ax.legend(loc="best")
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$z$ [m]")
        if show:
            plt.show()
        return ax


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
        Text to be used in table and plot labels.
        Default to Eq_1, Eq_2 ... if not chosen.
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
        qpsi_sign=-1,
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
        for file, eq_type, dummy, fc, tc, qs in zip(
            self.equilibrium_paths,
            self.fixed_or_free,
            self.dummy_coils,
            self.from_cocos,
            self.to_cocos,
            self.qpsi_sign,
            strict=False,
        ):
            self.equilibrium.append(
                select_eq(
                    file,
                    fixed_or_free=eq_type,
                    dummy_coils=dummy,
                    from_cocos=fc,
                    to_cocos=tc,
                    qpsi_sign=qs,
                )
            )

    def profile_dictionary(self):
        """
        Create a dictionary of profile information.
        To be used when comparing multiple equilibria
        but will work for a signle equilibrium.

        Cocos indecis and qpsi sign are set to Bluemira Defaults unless specified.

        The user can spesify from_cocos, to_cocos and qpsi_sign using:
            - lists if the equilibria have different values
            - single values if equilibria all have the same convention

        Return:
            Profile Dictionary
            (profile type: list of profile objects for each equilibria)
        """
        if not isinstance(self.equilibrium_paths, Iterable):
            prof = CustomProfile.from_eqdsk_file(
                self.equilibrium_paths,
                from_cocos=self.from_cocos,
                to_cocos=self.to_cocos,
                qpsi_sign=self.qpsi_sign,
            )
            return {
                "pprime": prof.pprime,
                "ffprime": prof.ffprime,
                "fRBpol": prof.fRBpol,
                "pressure": prof.pressure,
                "shape": prof.shape,
            }

        prof_dict = {
            "pprime": [],
            "ffprime": [],
            "fRBpol": [],
            "pressure": [],
            "shape": [],
        }
        for eq_path, fc, tc, q in zip(
            self.equilibrium_paths,
            self.from_cocos,
            self.to_cocos,
            self.qpsi_sign,
            strict=False,
        ):
            prof = CustomProfile.from_eqdsk_file(
                eq_path, from_cocos=fc, to_cocos=tc, qpsi_sign=q
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

    def plot_physics(self, title="Physics Parmeters", n_points=50, ax=None, show=True):  # noqa: FBT002
        """
        Plot physics parameters for the core plasma of each equilibria
        (i.e., plot across the normalised 1-D flux coordinate).

        Parameters
        ----------
        title:
            Title to be added at top of figure
        n_points:
            number of normalised psi points
        ax:
            Matplotlib Axes objects set by user
        show:
            Whether or not to display the plot

        Returns
        -------
        ax:
            Matplotlib Axes object

        """
        results = []
        [
            results.append(analyse_plasma_core(eq, n_points=n_points))
            for eq in self.equilibrium
        ]

        if ax is None:
            r, c = int((len(results[0].__dict__) - 1) / 2) + 1, 2
            gs = GridSpec(r, c)
            ax = [plt.subplot(gs[i]) for i in range(r * c)]

        for res, name in zip(results, self.equilibrium_names, strict=False):
            CorePlotter(res, ax, eq_name=name)

        plt.suptitle(title)
        if show:
            plt.show()
        return ax

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
                        coilset_dict[coil.name] = coil.current / 1e6
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
        if value is CSData.CURRENT:
            dataframe.style.set_caption("Current (MA)")
        elif value is CSData.XLOC:
            dataframe.style.set_caption("X-position (m)")
        elif value is CSData.ZLOC:
            dataframe.style.set_caption("Z-position (m)")
        return dataframe

    def plot_prof(
        self,
        ax,
        profiles,
        ax_title,
    ):
        """
        Plot equilibria profiles from list.

        Parameters
        ----------
        ax:
            Matplotlib axes object.
        profiles:
            List of profile objects.
        ax_title:
            Text to be printed above the axis.
        equilibrium_names:
            Names of the equilibia to be used as plot labels.
        """
        x = np.linspace(0, 1, 50)
        for p, name in zip(profiles, self.equilibrium_names, strict=False):
            ax.plot(x, p(x), marker=".", label=name)
            ax.legend(loc="best")
            ax.set_title(ax_title)

    def plot_compare_profiles(
        self,
        ax=None,
        header=None,
    ):
        """
        Plot the profiles of all the listed equilibria.

        Parameters
        ----------
        ax:
            List of Matplotlib Axes objects set by user
        header:
            Tesxt to be added at the top of the figure
        show:
            Whether or not to display the plot

        Returns
        -------
        ax1, ax2, ax3, ax4, ax5, ax6:
            The Matplotlib Axes objects for each subplot.

        Raises
        ------
        BluemiraError:
            if the axes provided are the incorrect shape

        """
        shape_ax = (2, 3)
        if ax is not None:
            if np.shape(ax) != shape_ax:
                raise BluemiraError(  # noqa: DOC501
                    "Subplot shape is (2,3), you have provided "
                    f"settings for {np.shape(ax)}."
                )
        else:
            _, ax = plt.subplots(2, 3)

        ax_list = [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1], ax[0, 2]]

        for axs, (key, profs) in zip(ax_list, self.profiles.items(), strict=False):
            self.plot_prof(axs, profs, ax_title=key)
        ax[1, 2].axis("off")

        if header is not None:
            plt.suptitle(header)
        plt.show()
        return ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1], ax[0, 2], ax[1, 2]

    def plot_compare_flux_surfaces(
        self,
        flux_surface=FluxSurfaceType.LCFS,
        plot_fixed=True,  # noqa: FBT002
        psi_norm=0.98,
        title=None,
        ax=None,
        show=True,  # noqa: FBT002
    ):
        """
        Plot a flux surface of choice for each equilibrium on the same axes.
        User can select either the LCFS, seperatrix or a flux surface with
        a given psi_norm.

        Always plots a LCFS for a fixed boundary equilibrium,
        unless plot_fixed is set to false, in which case no flux surfaces
        are plotted for any fixed boundary equilibria.

        Parameters
        ----------
        flux_surface:
            Type of flux surface to be plotted
        plot_fixed:
            Weather or not to plot the LCFS for any fixed boundary equilbria.
        psi_norm:
            The normalised psi value to use, if the user selects
            the psi norm comparison option.
        title:
            Title to be added at top of figure
        ax:
            Matplotlib Axes objects set by user
        show:
            Whether or not to display the plot

        Returns
        -------
        ax:
            Matplotlib Axes object

        """
        if ax is None:
            _, ax = plt.subplots()

        fs_list = []
        ccycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        for eq, fx, label in zip(
            self.equilibrium, self.fixed_or_free, self.equilibrium_names, strict=False
        ):
            if (flux_surface is FluxSurfaceType.SEPARATRIX) and (
                fx is not FixedOrFree.FIXED
            ):
                fs = eq.get_separatrix()
            elif (flux_surface is FluxSurfaceType.PSI_NORM) and (
                fx is not FixedOrFree.FIXED
            ):
                if psi_norm <= 1.0:
                    fs = eq.get_flux_surface(psi_norm)
                else:
                    coords = find_flux_surfs(
                        eq.x,
                        eq.z,
                        eq.psi(),
                        psi_norm,
                        o_points=eq._o_points,
                        x_points=eq._x_points,
                    )
                    loops = [Coordinates({"x": c.T[0], "z": c.T[1]}) for c in coords]
                    loops.sort(key=lambda loop: -loop.length)
                    fs = loops[:2]
            else:
                fs = eq.get_LCFS()
            fs_list.append(fs)

            color = next(ccycle)
            if (fx is not FixedOrFree.FIXED) or plot_fixed:
                if isinstance(fs, list):
                    ax.plot(fs[0].x, fs[0].z, label=label, color=color)
                    ax.plot(fs[1].x, fs[1].z, color=color)
                else:
                    ax.plot(fs.x, fs.z, label=label, color=color)

        ax.legend(loc="best")

        if title is not None:
            plt.suptitle(title)
        if show:
            plt.show()
        return ax

    def plot_divertor_length_angle(
        self,
        n_layers=10,
        dx_off=0.10,
        plasma_facing_boundary_list=None,
        legs_to_plot=DivLegsToPlot.ALL,
        ax=None,
        show=True,  # noqa: FBT002
    ):
        """
        Plot the divertor leg length and grazing angle for the equilibria
        divertor legs.

        Parameters
        ----------
        n_layers:
            Number of flux surfaces to extract for each leg
        dx_off:
            Total span in radial space of the flux surfaces to extract
        plasma_facing_boundary_list:
            List of associated plasma facing boundary coordinates
        legs_to_plot:
            Which legs to plot i.e, upper, lower or all.
        ax:
            Matplotlib Axes objects set by user
        show:
            Whether or not to display the plot

        Returns
        -------
        ax:
            Matplotlib Axes object

        """
        lengths, angles = get_leg_flux_info(
            self.equilibrium,
            n_layers=n_layers,
            dx_off=dx_off,
            plasma_facing_boundary_list=plasma_facing_boundary_list,
            legs_to_plot=legs_to_plot,
        )

        if ax is None:
            if legs_to_plot is DivLegsToPlot.ALL:
                _, ax = plt.subplots(2, 4, sharex="col")
            elif legs_to_plot in DivLegsToPlot.PAIR:
                _, ax = plt.subplots(2, 2, sharex="col")

        def plot_div_value(i, value, ax, name, n_layers=n_layers, dx_off=dx_off):
            for j, (k, v) in enumerate(value.items()):
                ax[i, j].plot(np.arange(n_layers) * (dx_off / n_layers), v, label=name)
                ax[i, j].set_title(k)
                ax[i, j].legend(name)

        for lngth, ang, name in zip(
            lengths, angles, self.equilibrium_names, strict=False
        ):
            plot_div_value(0, lngth, ax, name)
            plot_div_value(1, ang, ax, name)

        ax[0, 0].set_ylabel("length [m]")
        ax[1, 0].set_ylabel("angle [rad]")
        if show:
            plt.show()
        return ax
