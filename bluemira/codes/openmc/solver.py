# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""OpenMC designer"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, fields
from operator import attrgetter
from pathlib import Path
from typing import Literal

import numpy as np
import openmc

from bluemira.base.constants import raw_uc
from bluemira.base.look_and_feel import bluemira_debug
from bluemira.base.parameter_frame import ParameterFrame, make_parameter_frame
from bluemira.base.tools import _timing
from bluemira.codes.interface import (
    BaseRunMode,
    CodesSetup,
    CodesSolver,
    CodesTask,
    CodesTeardown,
)
from bluemira.codes.openmc.csg_reactor import OpenMCReactor
from bluemira.codes.openmc.make_cell import BlanketCellArray, DivertorCellArray
from bluemira.codes.openmc.material import MaterialsLibrary
from bluemira.codes.openmc.output import OpenMCResult
from bluemira.codes.openmc.params import (
    OpenMCNeutronicsSolverParams,
    PlasmaSourceParameters,
)
from bluemira.codes.openmc.tallying import filter_cells
from bluemira.plasma_physics.reactions import n_DT_reactions


class OpenMCRunModes(BaseRunMode):
    """OpenMC run modes"""

    RUN = openmc.settings.RunMode.FIXED_SOURCE.value
    RUN_AND_PLOT = "run and plot"
    PLOT = openmc.settings.RunMode.PLOT.value
    VOLUME = openmc.settings.RunMode.VOLUME.value


OPENMC_NAME = "OpenMC"


@dataclass
class OpenMCSimulationRuntimeParameters:
    """Parameters used in the actual simulation

    Parameters
    ----------
    particles:
        Number of neutrons emitted by the plasma source per batch.
    batches:
        How many batches to simulate.
    photon_transport:
        Whether to simulate the transport of photons (i.e. gamma-rays created) or not.
    electron_treatment:
        The way in which OpenMC handles secondary charged particles.
        'thick-target bremsstrahlung' or 'local energy deposition'
        'thick-target bremsstrahlung' accounts for the energy carried away by
        bremsstrahlung photons and deposited elsewhere, whereas 'local energy
        deposition' assumes electrons deposit all energies locally.
        (the latter is expected to be computationally faster.)
    run_mode:
        see below for details:
        https://docs.openmc.org/en/stable/usersguide/settings.html#run-modes
    openmc_write_summary:
        whether openmc should write a 'summary.h5' file or not.
    cross_section_xml:
        Where the xml file for cross-section is stored locally.
    """

    particles: int  # number of particles used in the neutronics simulation
    cross_section_xml: str | Path
    batches: int = 2
    photon_transport: bool = True
    # Bremsstrahlung only matters for very thin objects
    electron_treatment: Literal["ttb", "led"] = "led"
    run_mode: str = OpenMCRunModes.RUN.value
    openmc_write_summary: bool = False
    parametric_source: bool = True
    plot_axis: str = "xz"
    plot_pixel_per_metre: int = 100


class Setup(CodesSetup):
    """Setup task for OpenMC solver"""

    def __init__(
        self,
        out_path: str,
        codes_name: str,
        cross_section_xml: str,
        source,
        cell_arrays,
        pre_cell_model,
        materials,
    ):
        super().__init__(None, codes_name)

        self.out_path = out_path
        self.cells = cell_arrays.cells
        self.cross_section_xml = cross_section_xml
        self.source = source
        self.blanket_cell_array = cell_arrays.blanket
        self.divertor_cell_array = cell_arrays.divertor
        self.pre_cell_model = pre_cell_model
        self.materials = materials
        self.matlist = attrgetter(
            "outb_sf_mat",
            "outb_fw_mat",
            "outb_bz_mat",
            "outb_mani_mat",
            "outb_vv_mat",
            "divertor_mat",
            "div_fw_mat",
            "tf_coil_mat",
        )

    @contextmanager
    def _base_setup(self, run_mode, *, debug: bool = False):
        from openmc.config import config  # noqa: PLC0415

        self.files_created = set()
        folder = run_mode.name.lower()
        cwd = Path(self.out_path, folder)
        cwd.mkdir(parents=True, exist_ok=True)

        config["cross_sections"] = self.cross_section_xml

        self.settings = openmc.Settings(
            run_mode=run_mode.value, output={"summary": False}
        )
        self.universe = openmc.Universe(cells=self.cells)
        self.geometry = openmc.Geometry(self.universe)
        self.settings.verbosity = 10 if debug else 6
        try:
            yield
        finally:
            for obj, pth in (
                (self.settings, Path(self.out_path, folder, "settings.xml")),
                (self.geometry, Path(self.out_path, folder, "geometry.xml")),
                (self.materials, Path(self.out_path, folder, "materials.xml")),
            ):
                obj.export_to_xml(pth)
                self.files_created.add(pth)

    def _set_tallies(
        self,
        run_mode,
        tally_function: TALLY_FUNCTION_TYPE,
        blanket_cell_array: BlanketCellArray,
        divertor_cell_array: DivertorCellArray,
        material_list: list[openmc.Material],
    ):
        out_path = Path(self.out_path, run_mode.name.lower(), "tallies.xml")
        tallies_list = []
        for name, scores, filters in tally_function(
            material_list, blanket_cell_array, divertor_cell_array
        ):
            tally = openmc.Tally(name=name)
            tally.scores = [scores]
            tally.filters = filters
            tallies_list.append(tally)

        openmc.Tallies(tallies_list).export_to_xml(out_path)
        self.files_created.add(out_path)

    def run(
        self,
        run_mode,
        runtime_params,
        source_params,
        tally_function,
        *,
        debug: bool = False,
    ):
        """Run stage for setup openmc"""
        with self._base_setup(run_mode, debug=debug):
            self.settings.particles = runtime_params.particles
            self.settings.source = self.source(source_params)
            self.settings.batches = int(runtime_params.batches)
            self.settings.photon_transport = runtime_params.photon_transport
            self.settings.electron_treatment = runtime_params.electron_treatment

            self._set_tallies(
                run_mode,
                tally_function,
                self.blanket_cell_array,
                self.divertor_cell_array,
                self.matlist(self.materials),
            )
        self.files_created.add(f"statepoint.{runtime_params.batches}.h5")
        self.files_created.add("tallies.out")

    def plot(
        self,
        run_mode,
        runtime_params,
        _source_params,
        _tally_function,
        *,
        debug: bool = False,
    ):
        """Plot stage for setup openmc"""
        with self._base_setup(run_mode, debug=debug):
            z_max, _z_min, r_max, _r_min = self.pre_cell_model.bounding_box
            plot_width_0 = r_max * 2.1
            plot_width_1 = z_max * 3.1
            plot = openmc.Plot()
            plot.basis = runtime_params.plot_axis
            plot.pixels = [
                int(plot_width_0 * runtime_params.plot_pixel_per_metre),
                int(plot_width_1 * runtime_params.plot_pixel_per_metre),
            ]
            plot.width = raw_uc([plot_width_0, plot_width_1], "m", "cm")
            plot.show_overlaps = True

            plot_pth = Path(self.out_path, run_mode.name.lower(), "plots.xml")
            openmc.Plots([plot]).export_to_xml(plot_pth)
            self.files_created.add(plot_pth)

    def volume(
        self,
        run_mode,
        runtime_params,
        _source_params,
        _tally_function,
        *,
        debug: bool = False,
    ):
        """Stochastic volume stage for setup openmc"""
        z_max, z_min, r_max, r_min = self.pre_cell_model.bounding_box

        min_xyz = (r_min, r_min, z_min)
        max_xyz = (r_max, r_max, z_max)

        with self._base_setup(run_mode, debug=debug):
            self.settings.volume_calculations = openmc.VolumeCalculation(
                self.cells,
                runtime_params.particles,
                raw_uc(min_xyz, "m", "cm"),
                raw_uc(max_xyz, "m", "cm"),
            )
        self.files_created.add(Path(self.out_path, run_mode.name.lower(), "summary.h5"))
        # single batch
        self.files_created.add(
            Path(self.out_path, run_mode.name.lower(), "statepoint.1.h5")
        )


class Run(CodesTask):
    """Run task for OpenMC solver"""

    def __init__(self, out_path: Path, codes_name: str):
        super().__init__(None, codes_name)

        self.out_path = out_path

    def _run(self, run_mode, *, debug: bool = False):
        """Run openmc"""
        folder = run_mode.name.lower()
        cwd = Path(self.out_path, folder)
        cwd.mkdir(parents=True, exist_ok=True)
        _timing(
            openmc.run,
            "Executed in",
            f"Running OpenMC in {folder} mode",
            debug_info_str=False,
        )(
            output=debug,
            threads=None,
            geometry_debug=False,
            restart_file=None,
            tracks=False,
            cwd=cwd,
            openmc_exec="openmc",
            mpi_args=None,
            event_based=False,
            path_input=None,
        )

    def run(self, run_mode, *, debug: bool = False):
        """Run stage for run task"""
        self._run(run_mode, debug=debug)

    def plot(self, run_mode, *, debug: bool = False):
        """Plot stage for run task"""
        self._run(run_mode, debug=debug)

    def volume(self, run_mode, *, debug: bool = False):
        """Stochastic volume stage for run task"""
        self._run(run_mode, debug=debug)


class Teardown(CodesTeardown):
    """Teardown task for OpenMC solver"""

    def __init__(
        self,
        cells,
        out_path: str,
        codes_name: str,
    ):
        super().__init__(None, codes_name)

        self.out_path = out_path
        self.cells = cells

    @staticmethod
    def delete_files(files_created):
        """Remove files generated during the run (mainly .xml files.)"""
        removed_files, failed_to_remove_files = [], []
        for file_name in files_created:
            if (f := file_name).exists():
                f.unlink()
                removed_files.append(file_name)
            else:
                failed_to_remove_files.append(file_name)

        if removed_files:
            bluemira_debug(f"Removed files {removed_files}")
        if failed_to_remove_files:
            bluemira_debug(
                f"Attempted to remove files {failed_to_remove_files} but "
                "they don't exists."
            )

    def run(
        self,
        universe,
        files_created,
        source_params,
        statepoint_file,
        *,
        delete_files: bool = False,
    ):
        """Run stage for Teardown task"""
        result = OpenMCResult.from_run(
            universe,
            n_DT_reactions(source_params.reactor_power),
            statepoint_file,
        )
        if delete_files:
            self.delete_files(files_created)
        return result

    def plot(
        self,
        _universe,
        files_created,
        *_args,
        delete_files: bool = False,
    ):
        """Plot stage for Teardown task"""
        if delete_files:
            self.delete_files(files_created)

    def volume(
        self,
        _universe,
        files_created,
        _source_params,
        _statepoint_file,
        *,
        delete_files: bool = False,
    ) -> dict[int, float]:
        """Stochastic volume stage for teardown task"""
        if delete_files:
            self.delete_files(files_created)
        return {
            cell.id: raw_uc(
                np.nan if cell.volume is None else cell.volume, "cm^3", "m^3"
            )
            for cell in self.cells
        }


TALLY_FUNCTION_TYPE = Callable[
    [list[openmc.Material], BlanketCellArray, DivertorCellArray],
    tuple[
        str,
        str,
        list[openmc.CellFilter | openmc.MaterialFilter | openmc.ParticleFilter],
    ],
]


class OpenMCNeutronicsSolver(CodesSolver):
    """OpenMC 2D neutronics solver"""

    name: str = OPENMC_NAME
    param_cls: type[OpenMCNeutronicsSolverParams] = OpenMCNeutronicsSolverParams
    params: OpenMCNeutronicsSolverParams
    run_mode_cls: type[OpenMCRunModes] = OpenMCRunModes
    setup_cls: type[Setup] = Setup
    run_cls: type[Run] = Run
    teardown_cls: type[Teardown] = Teardown

    def __init__(
        self,
        params: dict | ParameterFrame,
        build_config: dict,
        neutronics_pre_cell_model,
        source: Callable[[PlasmaSourceParameters], openmc.source.SourceBase],
        tally_function: TALLY_FUNCTION_TYPE | None = None,
    ):
        self.params = make_parameter_frame(params, self.param_cls)
        self.build_config = build_config

        self.out_path = self.build_config.get("neutronics_output_path", Path.cwd())

        self.source = source

        self.pre_cell_model = neutronics_pre_cell_model
        self.materials = MaterialsLibrary.from_neutronics_materials(
            self.pre_cell_model.material_library
        )

        self.cell_arrays = OpenMCReactor.from_pre_cell_reactor(
            self.pre_cell_model, self.materials, control_id=True
        )

        self.tally_function = filter_cells if tally_function is None else tally_function

    @property
    def source(self) -> Callable[[PlasmaSourceParameters], openmc.Source]:
        """Source term for OpenMC"""
        return self._source

    @source.setter
    def source(self, value: Callable[[PlasmaSourceParameters], openmc.Source]):
        self._source = value

    @property
    def tally_function(self) -> TALLY_FUNCTION_TYPE:
        """Function used to set up tallies"""
        return self._tally_function

    @tally_function.setter
    def tally_function(self, value: TALLY_FUNCTION_TYPE):
        self._tally_function = value

    def execute(self, *, debug=False) -> OpenMCResult | dict[int, float]:
        """Execute the setup, run, and teardown tasks, in order."""
        run_mode = self.build_config.get("run_mode", self.run_mode_cls.RUN)
        if isinstance(run_mode, str):
            run_mode = self.run_mode_cls.from_string(run_mode)

        source_params = PlasmaSourceParameters.from_parameterframe(self.params)
        runtime_params = OpenMCSimulationRuntimeParameters(**{
            k.name: self.build_config[k.name]
            for k in fields(OpenMCSimulationRuntimeParameters)
            if k.name in self.build_config
        })
        if run_mode is OpenMCRunModes.RUN_AND_PLOT:
            for run_mode in (OpenMCRunModes.PLOT, OpenMCRunModes.RUN):
                result = self._single_run(
                    run_mode, source_params, runtime_params, debug=debug
                )
            return result
        return self._single_run(run_mode, source_params, runtime_params, debug=debug)

    def _single_run(
        self,
        run_mode: OpenMCRunModes,
        source_params: PlasmaSourceParameters,
        runtime_params: OpenMCSimulationRuntimeParameters,
        *,
        debug=False,
    ) -> OpenMCResult | dict[int, float]:
        self._setup = self.setup_cls(
            self.out_path,
            self.name,
            str(self.build_config["cross_section_xml"]),
            self.source,
            self.cell_arrays,
            self.pre_cell_model,
            self.materials,
        )
        self._run = self.run_cls(self.out_path, self.name)
        self._teardown = self.teardown_cls(
            self.cell_arrays.cells, self.out_path, self.name
        )

        result = None
        if setup := self._get_execution_method(self._setup, run_mode):
            result = setup(
                run_mode, runtime_params, source_params, self.tally_function, debug=debug
            )
        if run := self._get_execution_method(self._run, run_mode):
            result = run(run_mode, debug=debug)
        if teardown := self._get_execution_method(self._teardown, run_mode):
            result = teardown(
                self._setup.universe,
                self._setup.files_created,
                source_params,
                Path(
                    self.out_path,
                    run_mode.name.lower(),
                    f"statepoint.{runtime_params.batches}.h5",
                ),
            )
        return result
