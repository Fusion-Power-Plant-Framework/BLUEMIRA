# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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

"""
Database and DOE tools wrapping pypet (HDF5 files)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from pypet import Environment
from pypet import Parameter as pypetParameter
from pypet import pypetconstants
from pypet.trajectory import Trajectory
from pypet.utils.explore import cartesian_product
from pypet.utils.hdf5compression import compact_hdf5_file

from bluemira.base.logs import LogLevel, get_log_level
from bluemira.base.look_and_feel import bluemira_print


class DataBase:
    """
    DataBase object to build design studies using the pypet package. Compiles
    the results in an HDF5 file.

    Parameters
    ----------
    directory: str
        The directory in which to load/build the DataBase
    name: str
        The name of the DataBase (also the filename - no extension)
    function: callable
        The evalution function for the DataBase
    variables: List[Parameter]
        The variables to use in the exploration
    results

    Other Parameters
    ----------------
    ncpu: int
        The number of CPUs to use when building the DataBase
    """

    __loaded__ = False

    def __init__(self, directory, name, function, variables, results, **kwargs):
        self.name = name
        self.ncpu = kwargs.get("ncpu", os.cpu_count() - 2)
        self._multiproc = self.ncpu != 1

        self.filename = Path(directory, f"{self.name}.hdf5")
        self.function = function
        self.variables = variables
        self.results = results

        self.df = None
        self.env = None
        self.traj = None
        self.lock = False
        self.env0 = None
        self.traj_old = None

        if self.filename.exists():
            self.load_existing()
            self.runs = DataFrame.from_dict(self.get_run_inputs(self.traj_old))
            name += "_temp"

        self.initialise(name=name)

    def initialise(self, name):
        """
        Set up the pypet Environment and Trajectory, with the DataBase Parameters.
        """
        self.env = self.set_env(name)
        self.traj = self.env.trajectory
        for p in self.variables:
            if isinstance(p, pypetParameter):
                self.traj.f_add_parameter(
                    p.var, float(p.value), comment=f"{p.name} {p.unit}"
                )
            else:
                raise ValueError(f"Need a Parameter, not a {type(p)}")

    def set_env(self, name):
        """
        Set up the pypet Environment.
        """
        return Environment(
            trajectory=name,
            filename=self.filename,
            add_time=False,
            log_stdout=True,
            log_level=LogLevel(get_log_level()),
            overwrite_file=False,
            multiproc=self._multiproc,
            ncores=self.ncpu,
            large_overview_tables=False,
        )

    def add_ranges(self, parameter_dict):
        """
        Add ranges of variables to the DataBase. Builds a Cartesian product
        exploration.

        Parameters
        ----------
        parameter_dict: dict
            The dictionary of ranges for variables to add to the DataBase
        """
        new_runs = self.check_for_duplicates(cartesian_product(parameter_dict))
        if new_runs is None:
            bluemira_print("Already fully explored this parameter set.")
            self.lock = True
        else:
            self.traj.f_explore(new_runs.to_dict(orient="list"))
            self.lock = False

    def check_for_duplicates(self, exploration):
        """
        Check the exploration for duplicates in the DataBase

        Parameters
        ----------
        exploration: dict
            The dictionary of parameter ranges to explore

        Returns
        -------
        exploration: dict
            The dictionary of unique parameter ranges to explore
        """
        exploration = DataFrame.from_dict(self.pad_exploration(exploration))
        if hasattr(self, "runs"):
            c = pd.merge(self.runs, exploration, how="outer", indicator=True)
            new = c[c["_merge"] == "right_only"]
            if len(new) != 0:
                bluemira_print("Dropping duplicate parameter entries.")
                return new.drop("_merge", axis=1)
        else:
            return exploration

    def pad_exploration(self, exploration):
        """
        Adds default values to varied values so that check duplicates
        finds proper duplicates
        """
        length = np.ones(len(next(iter(exploration.items()))[-1]))  # Length of expl
        return {
            **{k: v[0] * length for k, v in self.get_run_inputs(self.traj).items()},
            **exploration,
        }  # Latter ** overwrites!

    def run(self):
        """
        Build the DataBase with the specified ranges of variables.
        """
        if self.lock:
            bluemira_print("Add new parameter explorations before running.")
            return None
        self.env.run(self.function)
        if self.traj_old:
            self.traj_old.f_merge(
                self.traj,
                backup_filename=False,
                delete_other_trajectory=True,
                move_data=True,
            )

    def load(self):
        """
        Load up results from the existing DataBase file.
        """
        if self.traj is None:
            self.traj = Trajectory(self.name)
        self.traj_old.f_load(load_results=pypetconstants.LOAD_DATA, force=True)
        self.__loaded__ = True

    def load_existing(self):
        """
        Load up the existing DataBase from a file and populate the old Trajectory.
        """
        self.env0 = self.set_env(self.name)
        self.traj_old = self.env0.traj
        # Just load explored parameters
        self.traj_old.f_load(
            load_parameters=2,
            load_derived_parameters=0,
            load_results=0,
            load_other_data=0,
            force=True,
        )

    def build_results_summary(self):
        """
        Build a summary DataFrame of the results.
        """
        res = []
        for i, run in enumerate(self.traj_old.f_get_run_names()):
            self.traj_old.f_set_crun(run)
            res += [
                [self.traj_old.crun[self.results[k]] for k in range(len(self.results))]
            ]
        self.traj_old.f_restore_default()  # Clears the run

        # Build dictionaries, then DataFrame
        self.df = DataFrame.from_dict(
            {
                **self.get_run_inputs(self.traj_old),
                **{r: res[i] for i, r in enumerate(self.results)},
            }
        )

    def get_run_inputs(self, traj):
        """
        Get the inputs of a run.

        Parameters
        ----------
        traj: Trajectory
            The trajectory for which to get the inputs

        Returns
        -------
        var_dict: dict
            The dictionary of variables
        """
        var = np.zeros((len(self.variables), len(traj)))
        for i, run in enumerate(traj.f_get_run_names()):
            traj.f_set_crun(run)
            for j in range(var.shape[0]):
                var[j, i] = traj[self.variables[j].var]
        return {v.var: var[j] for j, v in enumerate(self.variables)}

    def get_result(self, input_set):
        """
        Loads an individual result dynamically
        """
        self.traj_old.f_load(load_results=1)  # Load results skeleton
        self.traj_old.v_auto_load = True  # Dynamic loading (noisy in Ipython)
        df = self.runs
        for k, v in input_set.items():
            df = df[np.isclose(df[k], v)]
        result = self.traj_old.results.f_get(f"run_{df.index[0]:0>8}")

        return {k: result[k] for k in self.results}

    def compact_file(self):
        """
        Compress the HDF5 file. Takes a long time.
        """
        # NOTE: You tried this once on a 3.7 GB file (with naive storage style)
        # It took several minutes and reduced the file to 3.7 GB...
        compact_hdf5_file(self.filename)

    def frame_data(self):
        """
        Build a total DataFrame of the results.

        Returns
        -------
        df: DataFrame
            The DataFrame of results
        """
        if not self.__loaded__:
            self.load()
        var = [i.var for i in self.variables]
        row_list = []
        for run in self.traj_old.f_get_run_names():
            self.traj_old.f_set_crun(run)
            row = {i: self.traj_old[i] for i in var}
            for k in range(len(self.traj_old.results[0][self.results[0]])):
                newrow = row.copy()
                for j in self.results:
                    try:
                        newrow[j] = self.traj_old.results[run][j][k]
                    except AttributeError:  # Run failed
                        row[j] = None
                        newrow = row
                row_list.append(newrow)
            self.traj_old.f_restore_default()
        return DataFrame(row_list)
