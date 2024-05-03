# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
The EUDEMO reactor design routine.

1. Radial build (using PROCESS)
2. Perform equilibria optimisation
3. Build plasma
4. Design scaffold for the IVCs
5. Build vacuum vessel
6. Build TF coils
7. Build PF coils
8. Build cryo thermal shield
9. Build cryostat
10. Build radiation shield
11. Produce power cycle report
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.components import Component
from bluemira.base.constants import raw_uc
from bluemira.base.designer import run_designer
from bluemira.base.logs import set_log_level
from bluemira.base.parameter_frame import ParameterFrame
from bluemira.base.reactor import Reactor
from bluemira.base.reactor_config import ReactorConfig
from bluemira.builders.cryostat import CryostatBuilder, CryostatDesigner
from bluemira.builders.divertor import DivertorBuilder
from bluemira.builders.plasma import Plasma, PlasmaBuilder
from bluemira.builders.radiation_shield import RadiationShieldBuilder
from bluemira.builders.thermal_shield import CryostatTSBuilder, VVTSBuilder
from bluemira.display.displayer import show_cad
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.profiles import Profile
from bluemira.equilibria.run import Snapshot
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import distance_to, interpolate_bspline, offset_wire
from eudemo.blanket import Blanket, BlanketBuilder, BlanketDesigner
from eudemo.coil_structure import build_coil_structures_component
from eudemo.comp_managers import (
    CoilStructures,
    Cryostat,
    CryostatThermalShield,
    RadiationShield,
    ThermalShield,
    VacuumVesselThermalShield,
)
from eudemo.equilibria import (
    DummyFixedEquilibriumDesigner,
    FixedEquilibriumDesigner,
    ReferenceFreeBoundaryEquilibriumDesigner,
)
from eudemo.ivc import design_ivc
from eudemo.ivc.divertor_silhouette import Divertor
from eudemo.maintenance.duct_connection import (
    TSEquatorialPortDuctBuilder,
    TSUpperPortDuctBuilder,
    VVEquatorialPortDuctBuilder,
    VVUpperPortDuctBuilder,
)
from eudemo.maintenance.equatorial_port import EquatorialPortKOZDesigner
from eudemo.maintenance.lower_port import (
    LowerPortKOZDesigner,
    TSLowerPortDuctBuilder,
    VVLowerPortDuctBuilder,
)
from eudemo.maintenance.port_plug import (
    CryostatPortPlugBuilder,
    RadiationPortPlugBuilder,
)
from eudemo.maintenance.upper_port import UpperPortKOZDesigner
from eudemo.model_managers import EquilibriumManager
from eudemo.neutronics.run import run_neutronics
from eudemo.params import EUDEMOReactorParams
from eudemo.pf_coils import PFCoil, PFCoilsDesigner, build_pf_coils_component
from eudemo.power_cycle import SteadyStatePowerCycleSolver
from eudemo.radial_build import radial_build
from eudemo.tf_coils import TFCoil, TFCoilBuilder, TFCoilDesigner
from eudemo.vacuum_vessel import VacuumVessel, VacuumVesselBuilder

CONFIG_DIR = Path(__file__).parent.parent / "config"
BUILD_CONFIG_FILE_PATH = Path(CONFIG_DIR, "build_config.json").as_posix()


class EUDEMO(Reactor):
    """EUDEMO reactor definition."""

    # Components
    plasma: Plasma
    vacuum_vessel: VacuumVessel
    thermal_shield: ThermalShield
    divertor: Divertor
    blanket: Blanket
    tf_coils: TFCoil
    pf_coils: PFCoil
    coil_structures: CoilStructures
    cryostat: Cryostat
    radiation_shield: RadiationShield

    # Models
    equilibria: EquilibriumManager


def build_reference_equilibrium(
    params: dict | ParameterFrame,
    build_config: dict,
    equilibrium_manager: EquilibriumManager,
    lcfs_coords: Coordinates | None,
    profiles: Profile | None,
) -> Equilibrium:
    """
    Build the reference equilibrium for the tokamak and store in
    the equilibrium manager
    """
    designer = ReferenceFreeBoundaryEquilibriumDesigner(
        params,
        build_config,
        lcfs_coords,
        profiles,
    )
    reference_eq = designer.execute()
    constraints = None
    result = None
    if designer.opt_problem is not None:
        constraints = designer.opt_problem.targets
        result = designer._result
    ref_snapshot = Snapshot(
        reference_eq,
        reference_eq.coilset,
        constraints,
        reference_eq.profiles,
        result,
        reference_eq.limiter,
    )
    equilibrium_manager.add_state(equilibrium_manager.REFERENCE, ref_snapshot)
    return reference_eq


def build_plasma(params, build_config: dict, eq: Equilibrium) -> Plasma:
    """Build EUDEMO plasma from an equilibrium."""
    lcfs_loop = eq.get_LCFS()
    lcfs_wire = interpolate_bspline({"x": lcfs_loop.x, "z": lcfs_loop.z}, closed=True)
    builder = PlasmaBuilder(params, build_config, lcfs_wire)
    return Plasma(builder.build())


def build_vacuum_vessel(params, build_config, ivc_koz) -> VacuumVessel:
    """Build the vacuum vessel around the given IVC keep-out zone."""
    vv_builder = VacuumVesselBuilder(params, build_config, ivc_koz)
    return VacuumVessel(vv_builder.build())


def build_vacuum_vessel_thermal_shield(
    params, build_config, vv_koz
) -> VacuumVesselThermalShield:
    """Build the vacuum vessel thermal shield around the given  VV keep-out zone"""
    vvts_builder = VVTSBuilder(params, build_config, vv_koz)
    return VacuumVesselThermalShield(vvts_builder.build())


def build_cryots(params, build_config, pf_kozs, tf_koz) -> CryostatThermalShield:
    """
    Build the Cryostat thermal shield for the reactor.
    """
    cts_builder = CryostatTSBuilder(
        params,
        build_config,
        pf_kozs,
        tf_koz,
    )
    return CryostatThermalShield(cts_builder.build())


def assemble_thermal_shield(vv_thermal_shield, cryostat_thermal_shield):
    """
    Assemble the thermal shield component for the reactor.
    """
    component = Component(
        name="Thermal Shield",
        children=[vv_thermal_shield.component(), cryostat_thermal_shield.component()],
    )
    return ThermalShield(component)


def build_divertor(params, build_config, div_silhouette) -> Divertor:
    """Build the divertor given a silhouette of a sector."""
    builder = DivertorBuilder(params, build_config, div_silhouette)
    return Divertor(builder.build())


def build_blanket(
    params,
    build_config: dict,
    blanket_boundary,
    blanket_face,
    r_inner_cut: float,
    cut_angle: float,
) -> Blanket:
    """Build the blanket given a silhouette of a sector."""
    designer = BlanketDesigner(
        params, blanket_boundary, blanket_face, r_inner_cut, cut_angle
    )
    ib_silhouette, ob_silhouette, panel_points = designer.execute()
    builder = BlanketBuilder(params, build_config, ib_silhouette, ob_silhouette)
    return Blanket(builder.build(), panel_points)


def build_tf_coils(params, build_config, separatrix, vvts_cross_section) -> TFCoil:
    """Design and build the TF coils for the reactor."""
    centreline, wp_cross_section = run_designer(
        TFCoilDesigner,
        params,
        build_config,
        separatrix=separatrix,
        keep_out_zone=vvts_cross_section,
    )
    builder = TFCoilBuilder(
        params, build_config, centreline.create_shape(), wp_cross_section
    )
    return TFCoil(builder.build(), builder._make_field_solver())


def build_pf_coils(
    params,
    build_config,
    equilibrium_manager,
    tf_coil_boundary,
    pf_coil_keep_out_zones=(),
) -> PFCoil:
    """
    Design and build the PF coils for the reactor.
    """
    pf_coil_keep_out_zones_new = []
    # This is a very crude way of forcing PF coil centrepoints away from the KOZs
    # to stop clashes between ports and PF coil corners
    # TODO: Implement adjustable current bounds on sub-opt problems
    offset_value = np.sqrt(
        params.global_params.I_p.value / params.global_params.PF_jmax.value
    )
    for koz in pf_coil_keep_out_zones:
        new_wire = offset_wire(koz.boundary[0], offset_value, open_wire=False)
        new_face = BluemiraFace(new_wire)
        pf_coil_keep_out_zones_new.append(new_face)

    pf_designer = PFCoilsDesigner(
        params,
        build_config,
        equilibrium_manager,
        tf_coil_boundary,
        pf_coil_keep_out_zones_new,
    )

    coilset = pf_designer.execute()
    component = build_pf_coils_component(params, build_config, coilset)
    return PFCoil(component, coilset)


def build_coil_structures(
    params,
    build_config,
    tf_coil_xz_face,
    pf_coil_xz_wires,
    pf_coil_keep_out_zones,
) -> CoilStructures:
    """
    Design and build the coil structures for the reactor.
    """
    component = build_coil_structures_component(
        params, build_config, tf_coil_xz_face, pf_coil_xz_wires, pf_coil_keep_out_zones
    )
    return CoilStructures(component)


def build_upper_port(
    params,
    build_config,
    upper_port_koz: BluemiraFace,
    pf_coils,
    cryostat_ts_xz_boundary: BluemiraFace,
):
    """
    Build the upper port for the reactor.
    """
    ts_builder = TSUpperPortDuctBuilder(params, upper_port_koz, cryostat_ts_xz_boundary)
    ts_upper_port = ts_builder.build()
    vv_builder = VVUpperPortDuctBuilder(params, upper_port_koz, cryostat_ts_xz_boundary)
    vv_upper_port = vv_builder.build()
    return ts_upper_port, vv_upper_port


def build_equatorial_port(params, build_config, cryostat_ts_xz_boundary):
    """
    Build the equatorial port for the reactor.
    """
    builder = VVEquatorialPortDuctBuilder(params, cryostat_ts_xz_boundary)
    vv_eq_port = builder.build()
    builder = TSEquatorialPortDuctBuilder(params, cryostat_ts_xz_boundary)
    ts_eq_port = builder.build()
    return ts_eq_port, vv_eq_port


def build_lower_port(
    params,
    build_config,
    lp_duct_angled_nowall_extrude_boundary,
    lp_duct_straight_nowall_extrude_boundary,
    cryostat_xz_boundary,
):
    """Builder for the Lower Port and Duct"""
    offset = params.global_params.tk_cr_vv.value + params.global_params.g_cr_ts.value
    x_straight_end = cryostat_xz_boundary.bounding_box.x_max - offset
    builder = TSLowerPortDuctBuilder(
        params,
        build_config,
        lp_duct_angled_nowall_extrude_boundary,
        lp_duct_straight_nowall_extrude_boundary,
        x_straight_end,
    )
    ts_lower_port = builder.build()

    builder = VVLowerPortDuctBuilder(
        params,
        build_config,
        lp_duct_angled_nowall_extrude_boundary,
        lp_duct_straight_nowall_extrude_boundary,
        x_straight_end,
    )
    vv_lower_port = builder.build()
    return ts_lower_port, vv_lower_port


def build_cryostat(params, build_config, cryostat_thermal_koz) -> Cryostat:
    """
    Design and build the Cryostat for the reactor.
    """
    cryod = CryostatDesigner(params, cryostat_thermal_koz)
    return Cryostat(CryostatBuilder(params, build_config, *cryod.execute()).build())


def build_radiation_shield(params, build_config, cryostat_koz) -> RadiationShield:
    """
    Design and build the Radiation shield for the reactor.
    """
    return RadiationShield(
        RadiationShieldBuilder(params, build_config, BluemiraFace(cryostat_koz)).build()
    )


def build_cryostat_plugs(
    params, build_config, ts_ports, cryostat_xz_boundary: BluemiraFace
):
    """
    Build the port plugs for the cryostat.
    """
    closest_faces = []
    for port in ts_ports:
        xyz = port.get_component("xyz")
        for child in xyz.children:
            if "voidspace" not in child.name:
                port_xyz = child.shape.deepcopy()
                port_xyz.rotate(degree=-180 / params.global_params.n_TF.value)
        faces = port_xyz.faces
        distances = [
            distance_to(f.center_of_mass, cryostat_xz_boundary)[0] for f in faces
        ]
        closest_face = faces[np.argmin(distances)]
        closest_faces.append(closest_face)

    outer_wires = [cf.boundary[0].deepcopy() for cf in closest_faces]

    builder = CryostatPortPlugBuilder(
        params, build_config, outer_wires, cryostat_xz_boundary
    )
    return builder.build()


def build_radiation_plugs(params, build_config, cr_ports, radiation_xz_boundary):
    """
    Build the port plugs for the radiation shield.
    """
    closest_faces = []
    xyz = cr_ports.get_component("xyz")
    for child in xyz.children:
        if "voidspace" not in child.name:
            port_xyz = child.shape.deepcopy()
            port_xyz.rotate(degree=-180 / params.global_params.n_TF.value)
            faces = port_xyz.faces
            distances = [
                distance_to(f.center_of_mass, radiation_xz_boundary)[0] for f in faces
            ]
            closest_face = faces[np.argmin(distances)]
            closest_faces.append(closest_face)
    outer_wires = [cf.boundary[0].deepcopy() for cf in closest_faces]

    builder = RadiationPortPlugBuilder(
        params, build_config, outer_wires, radiation_xz_boundary
    )
    return builder.build()


if __name__ == "__main__":
    set_log_level("INFO")
    reactor_config = ReactorConfig(BUILD_CONFIG_FILE_PATH, EUDEMOReactorParams)
    reactor = EUDEMO(
        "EUDEMO",
        n_sectors=reactor_config.global_params.n_TF.value,
    )

    radial_build(
        reactor_config.params_for("Radial build").global_params,
        reactor_config.config_for("Radial build"),
    )
    lcfs_coords, profiles = run_designer(
        FixedEquilibriumDesigner,
        reactor_config.params_for("Fixed boundary equilibrium"),
        reactor_config.config_for("Fixed boundary equilibrium"),
    )

    lcfs_coords, profiles = run_designer(
        DummyFixedEquilibriumDesigner,
        reactor_config.params_for("Dummy fixed boundary equilibrium"),
        reactor_config.config_for("Dummy fixed boundary equilibrium"),
    )

    reactor.equilibria = EquilibriumManager()

    reference_eq = build_reference_equilibrium(
        reactor_config.params_for("Free boundary equilibrium"),
        reactor_config.config_for("Free boundary equilibrium"),
        reactor.equilibria,
        lcfs_coords,
        profiles,
    )

    reactor.plasma = build_plasma(
        reactor_config.params_for("Plasma"),
        reactor_config.config_for("Plasma"),
        reference_eq,
    )

    ivc_shapes = design_ivc(
        reactor_config.params_for("IVC").global_params,
        reactor_config.config_for("IVC"),
        equilibrium=reference_eq,
    )

    reactor.vacuum_vessel = build_vacuum_vessel(
        reactor_config.params_for("Vacuum vessel"),
        reactor_config.config_for("Vacuum vessel"),
        ivc_shapes.outer_boundary,
    )

    reactor.divertor = build_divertor(
        reactor_config.params_for("Divertor"),
        reactor_config.config_for("Divertor"),
        ivc_shapes.divertor_face,
    )

    upper_port_designer = UpperPortKOZDesigner(
        reactor_config.params_for("Upper Port"),
        reactor_config.config_for("Upper Port"),
        ivc_shapes.blanket_face,
    )
    upper_port_koz_xz, r_inner_cut, cut_angle = upper_port_designer.execute()

    reactor.blanket = build_blanket(
        reactor_config.params_for("Blanket"),
        reactor_config.config_for("Blanket"),
        ivc_shapes.inner_boundary,
        ivc_shapes.blanket_face,
        r_inner_cut,
        cut_angle,
    )

    run_neutronics(
        {
            "major_radius": {"value": 8.938, "unit": "m"},
            "aspect_ratio": {"value": 8.938 / 2.8938, "unit": "m"},
            "elongation": {"value": 1.65, "unit": ""},
            "triangularity": {"value": 0.333, "unit": ""},
            "reactor_power": {"value": 1998, "unit": "MW"},
            "peaking_factor": {"value": 1.508, "unit": ""},
            "temperature": {"value": raw_uc(15.4, "keV", "K"), "unit": "K"},
            "shaf_shift": {"value": 0, "unit": "m"},
            "vertical_shift": {"value": 0, "unit": "m"},
        },
        {
            "cross_section_xml": Path(
                "/home/oliver/bluemira_openmc_data/cross_sections.xml"
            ).expanduser(),
            "particles": 16800,  # 16800 takes 5 seconds,  1000000 takes 280 seconds.
            "batches": 3,
            "photon_transport": True,
            "electron_treatment": "ttb",
            "run_mode": "run_and_plot",
            "openmc_write_summary": False,
            "parametric_source": True,
            "blanket_type": "HCPB",
            "plot_axis": "xz",
            "plot_pixel_per_metre": 100,
        },
        blanket_panel_points=reactor.blanket.panel_points(),
        divertor_internal_boundary=ivc_shapes.div_internal_boundary,
        ivc_outer_boundary=ivc_shapes.outer_boundary,
        vv_wire=reactor.vacuum_vessel.xz_boundary(),
    )

    vv_thermal_shield = build_vacuum_vessel_thermal_shield(
        reactor_config.params_for("Thermal shield"),
        reactor_config.config_for("Thermal shield", "VVTS"),
        reactor.vacuum_vessel.xz_boundary(),
    )

    reactor.tf_coils = build_tf_coils(
        reactor_config.params_for("TF coils"),
        reactor_config.config_for("TF coils"),
        reactor.plasma.lcfs(),
        vv_thermal_shield.xz_boundary(),
    )

    eq_port_designer = EquatorialPortKOZDesigner(
        reactor_config.params_for("Equatorial Port"),
        reactor_config.config_for("Equatorial Port"),
        x_ob=20.0,
    )

    eq_port_koz_xz = eq_port_designer.execute()

    (
        lp_duct_xz_void_space,
        lower_port_koz_xz,
        lp_duct_angled_nowall_extrude_boundary,
        lp_duct_straight_nowall_extrude_boundary,
    ) = LowerPortKOZDesigner(
        reactor_config.params_for("Lower Port"),
        reactor_config.config_for("Lower Port"),
        ivc_shapes.divertor_face,
        ivc_shapes.div_wall_join_pt,
        reactor.tf_coils.xz_outer_boundary(),
    ).execute()

    reactor.pf_coils = build_pf_coils(
        reactor_config.params_for("PF coils"),
        reactor_config.config_for("PF coils"),
        reactor.equilibria,
        reactor.tf_coils.xz_outer_boundary(),
        pf_coil_keep_out_zones=[
            upper_port_koz_xz,
            eq_port_koz_xz,
            lower_port_koz_xz,
        ],
    )

    cryostat_thermal_shield = build_cryots(
        reactor_config.params_for("Thermal shield"),
        reactor_config.config_for("Thermal shield", "Cryostat"),
        reactor.pf_coils.xz_boundary(),
        reactor.tf_coils.xz_outer_boundary(),
    )

    reactor.thermal_shield = assemble_thermal_shield(
        vv_thermal_shield, cryostat_thermal_shield
    )

    reactor.coil_structures = build_coil_structures(
        reactor_config.params_for("Coil structures"),
        reactor_config.config_for("Coil structures"),
        tf_coil_xz_face=reactor.tf_coils.xz_face(),
        pf_coil_xz_wires=reactor.pf_coils.PF_xz_boundary(),
        pf_coil_keep_out_zones=[
            upper_port_koz_xz,
            eq_port_koz_xz,
            lower_port_koz_xz,
        ],
    )

    reactor.cryostat = build_cryostat(
        reactor_config.params_for("Cryostat"),
        reactor_config.config_for("Cryostat"),
        cryostat_thermal_shield.xz_boundary(),
    )

    reactor.radiation_shield = build_radiation_shield(
        reactor_config.params_for("RadiationShield"),
        reactor_config.config_for("RadiationShield"),
        reactor.cryostat.xz_boundary(),
    )

    # Incorporate ports
    # TODO: Make potentially larger depending on where the PF
    # coils ended up. Warn if this isn't the case.

    ts_upper_port, vv_upper_port = build_upper_port(
        reactor_config.params_for("Upper Port"),
        reactor_config.config_for("Upper Port"),
        upper_port_koz_xz,
        reactor.pf_coils,
        cryostat_thermal_shield.xz_boundary(),
    )
    ts_eq_port, vv_eq_port = build_equatorial_port(
        reactor_config.params_for("Equatorial Port"),
        reactor_config.config_for("Equatorial Port"),
        cryostat_thermal_shield.xz_boundary(),
    )

    ts_lower_port, vv_lower_port = build_lower_port(
        reactor_config.params_for("Lower Port"),
        reactor_config.config_for("Lower Port"),
        lp_duct_angled_nowall_extrude_boundary,
        lp_duct_straight_nowall_extrude_boundary,
        reactor.cryostat.xz_boundary(),
    )

    reactor.vacuum_vessel.add_ports(
        [vv_upper_port, vv_eq_port, vv_lower_port],
        n_TF=reactor_config.global_params.n_TF.value,
    )

    reactor.thermal_shield.add_ports(
        [ts_upper_port, ts_eq_port, ts_lower_port],
        n_TF=reactor_config.global_params.n_TF.value,
    )

    cr_plugs = build_cryostat_plugs(
        reactor_config.params_for("Cryostat"),
        reactor_config.config_for("Cryostat"),
        [ts_upper_port, ts_eq_port, ts_lower_port],
        reactor.cryostat.xz_boundary(),
    )

    rs_plugs = build_radiation_plugs(
        reactor_config.params_for("RadiationShield"),
        reactor_config.config_for("RadiationShield"),
        cr_plugs,
        reactor.radiation_shield.xz_boundary(),
    )

    reactor.cryostat.add_plugs(
        cr_plugs,
        n_TF=reactor_config.global_params.n_TF.value,
    )

    reactor.radiation_shield.add_plugs(
        rs_plugs,
        n_TF=reactor_config.global_params.n_TF.value,
    )

    debug = [upper_port_koz_xz, eq_port_koz_xz, lower_port_koz_xz]
    debug.extend(reactor.pf_coils.xz_boundary())
    # I know there are clashes, I need to put in dynamic bounds on position opt to
    # include coil XS.
    show_cad(debug)

    reactor.show_cad("xz")
    reactor.show_cad(n_sectors=2)

    sspc_solver = SteadyStatePowerCycleSolver(reactor_config.global_params)
    sspc_result = sspc_solver.execute()
    sspc_solver.model.plot()
    plt.show()
