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

"""EUDEMO reactor build parameters."""

from dataclasses import dataclass

from bluemira.base.parameter_frame import Parameter, ParameterFrame


@dataclass
class EUDEMOReactorParams(ParameterFrame):
    """All parameters for the EUDEMO reactor."""

    # Common parameters
    A: Parameter[float]
    B_0: Parameter[float]
    B_premag_stray_max: Parameter[float]
    B_tf_peak: Parameter[float]
    beta_p: Parameter[float]
    beta: Parameter[float]
    bb_min_angle: Parameter[float]
    C_Ejima: Parameter[float]
    condrad_cryo_heat: Parameter[float]
    CS_bmax: Parameter[float]
    CS_jmax: Parameter[float]
    delta_95: Parameter[float]
    delta: Parameter[float]
    e_mult: Parameter[float]
    e_nbi: Parameter[float]
    eta_nb: Parameter[float]
    f_bs: Parameter[float]
    f_ni: Parameter[float]
    g_cr_rs: Parameter[float]
    g_cr_ts: Parameter[float]
    g_cs_mod: Parameter[float]
    g_cs_tf: Parameter[float]
    g_ts_pf: Parameter[float]
    g_ts_tf: Parameter[float]
    g_vv_bb: Parameter[float]
    g_vv_ts: Parameter[float]
    h_cp_top: Parameter[float]
    H_star: Parameter[float]
    h_tf_max_in: Parameter[float]
    I_p: Parameter[float]
    ib_offset_angle: Parameter[float]
    kappa_95: Parameter[float]
    kappa: Parameter[float]
    l_i: Parameter[float]
    n_CS: Parameter[int]
    n_PF: Parameter[int]
    n_TF: Parameter[int]
    ob_offset_angle: Parameter[float]
    P_bd_in: Parameter[float]
    P_brehms: Parameter[float]
    P_el_net_process: Parameter[float]
    P_el_net: Parameter[float]
    P_fus_DD: Parameter[float]
    P_fus_DT: Parameter[float]
    P_fus: Parameter[float]
    P_hcd_ss: Parameter[float]
    P_line: Parameter[float]
    P_rad_core: Parameter[float]
    P_rad_edge: Parameter[float]
    P_rad: Parameter[float]
    P_sep: Parameter[float]
    P_sync: Parameter[float]
    PF_bmax: Parameter[float]
    PF_jmax: Parameter[float]
    q_95: Parameter[float]
    R_0: Parameter[float]
    r_cs_corner: Parameter[float]
    r_pf_corner: Parameter[float]
    r_cp_top: Parameter[float]
    r_cs_in: Parameter[float]
    r_fw_ib_in: Parameter[float]
    r_fw_ob_in: Parameter[float]
    r_tf_in_centre: Parameter[float]
    r_tf_in: Parameter[float]
    r_tf_inboard_out: Parameter[float]
    r_tf_out_centre: Parameter[float]
    r_ts_ib_in: Parameter[float]
    r_vv_ib_in: Parameter[float]
    r_vv_ob_in: Parameter[float]
    sigma_tf_case_max: Parameter[float]
    sigma_tf_wp_max: Parameter[float]
    T_e: Parameter[float]
    tau_e: Parameter[float]
    tau_flattop: Parameter[float]
    TF_currpt_ob: Parameter[float]
    TF_E_stored: Parameter[float]
    TF_res_bus: Parameter[float]
    TF_res_tot: Parameter[float]
    TF_respc_ob: Parameter[float]
    TF_ripple_limit: Parameter[float]
    tf_wp_depth: Parameter[float]
    tf_wp_width: Parameter[float]
    tk_bb_ib: Parameter[float]
    tk_bb_ob: Parameter[float]
    tk_cr_vv: Parameter[float]
    tk_cs_casing: Parameter[float]
    tk_cs_insulation: Parameter[float]
    tk_cs: Parameter[float]
    tk_pf_casing: Parameter[float]
    tk_pf_insulation: Parameter[float]
    tk_fw_in: Parameter[float]
    tk_fw_out: Parameter[float]
    tk_rs: Parameter[float]
    tk_sh_bot: Parameter[float]
    tk_sh_in: Parameter[float]
    tk_sh_out: Parameter[float]
    tk_sh_top: Parameter[float]
    tk_sol_ib: Parameter[float]
    tk_sol_ob: Parameter[float]
    tk_tf_front_ib: Parameter[float]
    tk_tf_inboard: Parameter[float]
    tk_tf_ins: Parameter[float]
    tk_tf_insgap: Parameter[float]
    tk_tf_nose: Parameter[float]
    tk_tf_outboard: Parameter[float]
    tk_tf_side: Parameter[float]
    tk_ts: Parameter[float]
    tk_vv_bot: Parameter[float]
    tk_vv_in: Parameter[float]
    tk_vv_out: Parameter[float]
    tk_vv_top: Parameter[float]
    v_burn: Parameter[float]
    V_p: Parameter[float]
    well_depth: Parameter[float]
    Z_eff: Parameter[float]

    # PLASMOD
    T_e_ped: Parameter[float]
    q_control: Parameter[float]

    # Equilibrium
    div_L2D_ib: Parameter[float]
    div_L2D_ob: Parameter[float]
    shaf_shift: Parameter[float]

    # Wall designer
    fw_psi_n: Parameter[float]

    # Divertor silhouette
    div_type: Parameter[str]
    div_Ltarg: Parameter[float]  # noqa: N815
    div_open: Parameter[bool]

    # Plasma face
    c_rm: Parameter[float]

    # Vacuum vessel
    vv_in_off_deg: Parameter[float]
    vv_out_off_deg: Parameter[float]

    # Divertor
    n_div_cassettes: Parameter[int]

    # Blanket
    n_bb_inboard: Parameter[int]
    n_bb_outboard: Parameter[int]

    # TF Coils
    r_tf_current_ib: Parameter[float]
    tk_tf_wp_y: Parameter[float]
    tk_tf_wp: Parameter[float]
    z_0: Parameter[float]

    # OIS
    tk_ois: Parameter[float]
    g_ois_tf_edge: Parameter[float]
    min_OIS_length: Parameter[float]

    # PF Coils
    F_cs_sepmax: Parameter[float]
    F_cs_ztotmax: Parameter[float]
    F_pf_zmax: Parameter[float]

    # PF supports
    pf_s_tk_plate: Parameter[float]
    pf_s_n_plate: Parameter[int]
    pf_s_g: Parameter[float]

    # Gravity supports
    x_g_support: Parameter[float]
    x_gs_kink_diff: Parameter[float]
    z_gs: Parameter[float]
    tf_gs_tk_plate: Parameter[float]
    tf_gs_g_plate: Parameter[float]
    tf_gs_base_depth: Parameter[float]

    # Ports
    tk_vv_single_wall: Parameter[float]
    tk_vv_double_wall: Parameter[float]
    lower_port_angle: Parameter[float]

    # Powercycle
    bb_p_inlet: Parameter[float]
    bb_p_outlet: Parameter[float]
    bb_pump_eta_el: Parameter[float]
    bb_pump_eta_isen: Parameter[float]
    bb_t_inlet: Parameter[float]
    bb_t_outlet: Parameter[float]
    blanket_type: Parameter[str]
    div_pump_eta_el: Parameter[float]
    div_pump_eta_isen: Parameter[float]
    e_decay_mult: Parameter[float]
    f_core_rad_fw: Parameter[float]
    f_fw_aux: Parameter[float]
    f_sol_ch_fw: Parameter[float]
    f_sol_rad: Parameter[float]
    f_sol_rad_fw: Parameter[float]
    vvpfrac: Parameter[float]
    P_hcd_ss_el: Parameter[float]

    # First wall panelling
    fw_a_max: Parameter[float]
    fw_dL_min: Parameter[float]  # noqa: N815
