# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
PROCESS mappings
"""

from bluemira.codes.utilities import create_mapping

IN_mappings = {
    "n_TF": ("n_tf", "dimensionless"),
    "TF_ripple_limit": ("ripmax", "%"),
    "C_Ejima": ("gamma", "dimensionless"),
    "e_nbi": ("enbeam", "keV"),
    "P_hcd_ss": ("pinjalw", "MW"),
    "eta_nb": ("etanbi", "dimensionless"),
    "e_mult": ("emult", "dimensionless"),
    "tk_cr_vv": ("ddwex", "m"),
    "tk_tf_front_ib": ("casthi", "m"),
    "tk_tf_side": ("casths", "m"),
    "PsepB_qAR_max": ("psepbqarmax", "MW.T/m"),
    "q_0": ("q0", "dimensionless"),
    "m_s_limit": ("m_s_limit", "dimensionless"),
    "delta": ("triang", "dimensionless"),
    "sigma_tf_case_max": ("sig_tf_case_max", "Pa"),
    "sigma_tf_wp_max": ("sig_tf_wp_max", "Pa"),
    "sigma_cs_wp_max": ("alstroh", "Pa"),
    "H_star": ("hfact", "dimensionless"),
    "bb_pump_eta_el": ("etahtp", "dimensionless"),
    "bb_pump_eta_isen": ("etaiso", "dimensionless"),
    "bb_t_inlet": ("inlet_temp", "K"),
    "bb_t_outlet": ("outlet_temp", "K"),
    "eta_ecrh": ("etaech", "dimensionless"),
    "gamma_ecrh": ("gamma_ecrh", "1e20 A/W/m^2"),
}

OUT_mappings = {
    "R_0": ("rmajor", "m"),
    "B_0": ("bt", "T"),
    "kappa_95": ("kappa95", "dimensionless"),
    "kappa": ("kappa", "dimensionless"),
    "delta_95": ("triang95", "dimensionless"),
    "delta": ("triang", "dimensionless"),
    "I_p": ("plascur/1d6", "MA"),
    "P_fus": ("powfmw", "MW"),
    "P_fus_DT": ("pdt", "MW"),
    "P_fus_DD": ("pdd", "MW"),
    "H_star": ("hfact", "dimensionless"),
    "P_sep": ("pdivt", "MW"),
    "P_rad_core": ("pinnerzoneradmw", "MW"),
    "P_rad_edge": ("pouterzoneradmw", "MW"),
    "P_rad": ("pradmw", "MW"),
    "P_line": ("plinepv*vol", "MW"),
    "P_sync": ("psyncpv*vol", "MW"),
    "P_brehms": ("pbrempv*vol", "MW"),
    "f_bs": ("bootipf", "dimensionless"),
    "beta_p": ("betap", "dimensionless"),
    "beta": ("beta", "dimensionless"),
    "tau_e": ("taueff", "s"),
    "v_burn": ("vburn", "V"),
    "tk_fw_in": ("fwith", "m"),
    "tk_fw_out": ("fwoth", "m"),
    "tk_tf_inboard": ("tfcth", "m"),
    "tk_tf_nose": ("thkcas", "m"),
    "tf_wp_width": ("dr_tf_wp", "m"),
    "tf_wp_depth": ("wwp1", "m"),
    "tk_tf_ins": ("tinstf", "m"),
    "tk_tf_insgap": ("tfinsgap", "m"),
    "tk_cs": ("ohcth", "m"),
    "r_cp_top": ("r_cp_top", "m"),
    "r_cs_in": ("bore", "m"),
    "r_tf_in": ("rtfin", "m"),
    "r_tf_in_centre": ("r_tf_inboard_mid", "m"),
    "r_ts_ib_in": ("r_ts_ib_in", "m"),
    "r_vv_ib_in": ("r_vv_ib_in", "m"),
    "r_fw_ib_in": ("r_fw_ib_in", "m"),
    "r_fw_ob_in": ("r_fw_ob_in", "m"),
    "r_vv_ob_in": ("r_vv_ob_in", "m"),
    "r_tf_out_centre": ("r_tf_outboard_mid", "m"),
    "g_vv_ts": ("gapds", "m"),
    "TF_res_bus": ("tfbusres", "m"),
    "TF_res_tot": ("ztot", "ohm"),
    "TF_E_stored": ("estotftgj", "GJ"),
    "TF_respc_ob": ("tflegres", "ohm"),
    "TF_currpt_ob": ("cpttf", "A"),
    "P_bd_in": ("pinjmw", "MW"),
    "condrad_cryo_heat": ("qss/1.0d6", "MW"),
}

IO_mappings = {
    "A": ("aspect", "dimensionless"),
    "tau_flattop": (("tbrnmn", "tburn"), "s"),
    "P_el_net": (("pnetelin", "pnetelmw"), "MW"),
    "tk_bb_ib": ("blnkith", "m"),
    "tk_bb_ob": ("blnkoth", "m"),
    "tk_vv_in": ("d_vv_in", "m"),
    "tk_sol_ib": ("scrapli", "m"),
    "tk_sol_ob": ("scraplo", "m"),
    "g_cs_tf": ("gapoh", "m"),
    "g_ts_tf": ("tftsgap", "m"),
    "g_vv_bb": ("vvblgap", "m"),
}

NONE_mappings = {
    "B_tf_peak": ("bmaxtfrp", "T"),
    "T_e": ("te", "keV"),
    "Z_eff": ("zeff", "amu"),
    "V_p": ("vol", "m^3"),
    "l_i": ("rli", "dimensionless"),
    "f_ni": ("faccd", "dimensionless"),
    "tk_tf_outboard": ("tfthko", "m"),
    "h_cp_top": ("h_cp_top", "m"),
    "h_tf_max_in": ("hmax", "m"),
    "r_tf_inboard_out": ("r_tf_inboard_out", "m"),
    # The following mappings are not 1:1
    "tk_sh_in": ("shldith", "m"),
    "tk_sh_out": ("shldoth", "m"),
    "tk_sh_top": ("shldtth", "m"),
    "tk_sh_bot": ("shldlth", "m"),
    "tk_vv_out": ("d_vv_out", "m"),
    "tk_vv_top": ("d_vv_top", "m"),
    "tk_vv_bot": ("d_vv_bot", "m"),
    # Thermal shield thickness is a constant for us
    "tk_ts": ("thshield_ib", "m"),
    # "tk_ts": ("thshield_ob", "m"),
    # "tk_ts": ("thshield_vb", "m"),
    "q_95": ("q95", "dimensionless"),
}

mappings = create_mapping(IN_mappings, OUT_mappings, IO_mappings, NONE_mappings)
