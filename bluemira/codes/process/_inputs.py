# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Parameter classes/structures for Process
"""

from collections.abc import Iterator
from dataclasses import dataclass, fields

from bluemira.codes.process.api import _INVariable


@dataclass
class ProcessInputs:
    """
    Process Inputs dataclass

    Notes
    -----
    All entries get wrapped in an INVariable class to enable easy InDat writing.

    Units for these are available in bluemira.codes.process.mapping for mapped
    variables otherwise
    `process.io.python_fortran_dicts.get_dicts()["DICT_DESCRIPTIONS"]`
    """

    runtitle: str | None = None

    # Optimisation problem setup
    bounds: dict[str, dict[str, str]] | None = None
    icc: list[int] | None = None
    ixc: list[int] | None = None

    # Settings
    maxcal: int | None = None
    minmax: int | None = None
    epsvmc: float | None = None
    ioptimz: int | None = None
    output_costs: int | None = None
    isweep: int | None = None
    nsweep: int | None = None
    sweep: list[float] | None = None
    pulsetimings: int | None = None
    scan_dim: int | None = None
    # Top down of PROCESS variables list

    # Times
    t_burn: float | None = None
    t_between_pulse: float | None = None
    t_fusion_ramp: float | None = None
    t_current_ramp_up: float | None = None
    t_ramp_down: float | None = None
    t_precharge: float | None = None

    # FWBS
    ibkt_life: int | None = None
    denstl: float | None = None
    denw: float | None = None
    emult: float | None = None
    fblss: float | None = None
    fdiv: float | None = None
    fwbsshape: int | None = None
    fw_armour_thickness: float | None = None
    iblanket: int | None = None
    iblnkith: int | None = None
    li6enrich: float | None = None
    breeder_f: float | None = None
    breeder_multiplier: float | None = None
    vfcblkt: float | None = None
    vfpblkt: float | None = None
    blktmodel: int | None = None  # Listed as an output...
    # f_neut_shield: float = # -1.0 the documentation defaults cannot be right...
    breedmat: int | None = None
    fblbe: float | None = None
    fblbreed: float | None = None
    fblhebmi: float | None = None
    fblhebmo: float | None = None
    fblhebpi: float | None = None
    fblhebpo: float | None = None
    hcdportsize: int | None = None
    npdiv: int | None = None
    nphcdin: int | None = None
    nphcdout: int | None = None
    wallpf: float | None = None
    iblanket_thickness: int | None = None
    secondary_cycle: int | None = None  # Listed as an output...
    secondary_cycle_liq: int | None = None
    afwi: float | None = None
    afwo: float | None = None
    fw_wall: float | None = None
    afw: float | None = None
    pitch: float | None = None
    fwinlet: float | None = None
    fwoutlet: float | None = None
    fwpressure: float | None = None
    roughness: float | None = None
    fw_channel_length: float | None = None
    peaking_factor: float | None = None
    blpressure: float | None = None
    inlet_temp: float | None = None
    outlet_temp: float | None = None
    coolp: float | None = None
    coolwh: int | None = None
    nblktmodpo: int | None = None
    nblktmodpi: int | None = None
    nblktmodto: int | None = None
    nblktmodti: int | None = None
    tfwmatmax: float | None = None
    fw_th_conductivity: float | None = None
    fvoldw: float | None = None
    fvolsi: float | None = None
    fvolso: float | None = None
    fwclfr: float | None = None
    rpf2dewar: float | None = None
    vfshld: float | None = None
    irefprop: int | None = None
    fblli2o: float | None = None
    fbllipb: float | None = None
    vfblkt: float | None = None
    declblkt: float | None = None
    declfw: float | None = None
    declshld: float | None = None
    blkttype: int | None = None
    etaiso: float | None = None
    etahtp: float | None = None
    n_liq_recirc: int | None = None
    bz_channel_conduct_liq: float | None = None
    blpressure_liq: float | None = None
    inlet_temp_liq: float | None = None
    outlet_temp_liq: float | None = None
    f_nuc_pow_bz_struct: float | None = None
    pnuc_fw_ratio_dcll: float | None = None

    # TF coil
    sig_tf_case_max: float | None = None
    sig_tf_wp_max: float | None = None
    bcritsc: float | None = None
    casthi_fraction: float | None = None
    casths_fraction: float | None = None
    f_t_turn_tf: float | None = None
    t_turn_tf_max: float | None = None
    cpttf: float | None = None
    cpttf_max: float | None = None
    dcase: float | None = None
    dcond: list[float] | None = None
    dcondins: float | None = None
    dhecoil: float | None = None
    b_crit_upper_nbti: float | None = None
    t_crit_nbti: float | None = None
    fcutfsu: float | None = None
    fhts: float | None = None
    i_tf_stress_model: int | None = None
    i_tf_wp_geom: int | None = None
    i_tf_case_geom: int | None = None  # Listed as an output
    i_tf_turns_integer: int | None = None  # Listed as an output
    i_tf_sc_mat: int | None = None
    i_tf_sup: int | None = None
    i_tf_shape: int | None = None  # Listed as an output
    i_tf_cond_eyoung_trans: int | None = None
    i_r_cp_top: int | None = None
    i_tf_tresca: int | None = None
    n_pancake: int | None = None
    n_layer: int | None = None
    n_rad_per_layer: int | None = None
    i_tf_bucking: int | None = None
    n_tf_graded_layers: int | None = None
    jbus: float | None = None
    eyoung_ins: float | None = None
    eyoung_steel: float | None = None
    eyong_cond_axial: float | None = None
    eyoung_res_tf_buck: float | None = None
    # eyoung_al: Optional[float] = 69000000000.0 # defaults  cannot be right
    poisson_steel: float | None = None
    poisson_copper: float | None = None
    poisson_al: float | None = None
    str_cs_con_res: float | None = None
    str_pf_con_res: float | None = None
    str_tf_con_res: float | None = None
    str_wp_max: float | None = None
    i_str_wp: int | None = None
    quench_model: str = None
    tcritsc: float | None = None
    tdmptf: float | None = None
    tfinsgap: float | None = None
    # rhotfbus: Optional[float] = -1.0 # defaults cannot be right
    frhocp: float | None = None
    frholeg: float | None = None
    i_cp_joints: int | None = None
    rho_tf_joints: float | None = None
    n_tf_joints_contact: int | None = None
    n_tf_joints: int | None = None
    th_joint_contact: float | None = None
    # eff_tf_cryo: Optional[float] = -1.0 # defaults cannot be right
    n_tf: int | None = None
    tftmp: float | None = None
    thicndut: float | None = None
    thkcas: float | None = None
    thwcndut: float | None = None
    tinstf: float | None = None
    tmaxpro: float | None = None
    tmax_croco: float | None = None
    tmpcry: float | None = None
    vdalw: float | None = None
    f_vforce_inboard: float | None = None
    vftf: float | None = None
    etapump: float | None = None
    fcoolcp: float | None = None
    fcoolleg: float | None = None
    ptempalw: float | None = None
    rcool: float | None = None
    tcoolin: float | None = None
    tcpav: float | None = None
    vcool: float | None = None
    theta1_coil: float | None = None
    theta1_vv: float | None = None
    max_vv_stress: float | None = None
    inuclear: int | None = None
    qnuc: float | None = None
    ripmax: float | None = None
    tf_in_cs: int | None = None
    tfcth: float | None = None
    tftsgap: float | None = None
    casthi: float | None = None
    casths: float | None = None
    tmargmin: float | None = None
    tmargmin_cs: float | None = None
    oacdcp: float | None = None
    t_turn_tf: int | None = None

    # PF Power
    iscenr: int | None = None
    maxpoloidalpower: float | None = None

    # Cost variables
    abktflnc: float | None = None
    adivflnc: float | None = None
    cconfix: float | None = None
    cconshpf: float | None = None
    cconshtf: float | None = None
    cfactr: float | None = None
    cfind: list[float] | None = None
    cland: float | None = None
    costexp: float | None = None
    costexp_pebbles: float | None = None
    cost_factor_buildings: float | None = None
    cost_factor_land: float | None = None
    cost_factor_tf_coils: float | None = None
    cost_factor_fwbs: float | None = None
    cost_factor_tf_rh: float | None = None
    cost_factor_tf_vv: float | None = None
    cost_factor_tf_bop: float | None = None
    cost_factor_tf_misc: float | None = None
    maintenance_fwbs: float | None = None
    maintenance_gen: float | None = None
    amortization: float | None = None
    cost_model: int | None = None
    cowner: float | None = None
    cplife_input: float | None = None
    cpstflnc: float | None = None
    csi: float | None = None
    # cturbb: Optional[float] = 38.0 # defaults cannot be right
    decomf: float | None = None
    dintrt: float | None = None
    fcap0: float | None = None
    fcap0cp: float | None = None
    fcdfuel: float | None = None
    fcontng: float | None = None
    fcr0: float | None = None
    fkind: float | None = None
    iavail: int | None = None
    life_dpa: float | None = None
    avail_min: float | None = None
    favail: float | None = None
    num_rh_systems: int | None = None
    conf_mag: float | None = None
    div_prob_fail: float | None = None
    div_umain_time: float | None = None
    div_nref: float | None = None
    div_nu: float | None = None
    fwbs_nref: float | None = None
    fwbs_nu: float | None = None
    fwbs_prob_fail: float | None = None
    fwbs_umain_time: float | None = None
    redun_vacp: float | None = None
    tbktrepl: float | None = None
    tcomrepl: float | None = None
    tdivrepl: float | None = None
    uubop: float | None = None
    uucd: float | None = None
    uudiv: float | None = None
    uufuel: float | None = None
    uufw: float | None = None
    uumag: float | None = None
    uuves: float | None = None
    ifueltyp: int | None = None
    ucblvd: float | None = None
    ucdiv: float | None = None
    ucme: float | None = None
    ireactor: int | None = None
    lsa: int | None = None
    discount_rate: float | None = None
    startupratio: float | None = None
    tlife: float | None = None
    bkt_life_csf: int | None = None
    i_bldgs_size: int | None = None
    # ...

    # CS fatigue
    residual_sig_hoop: float | None = None
    n_cycle_min: int | None = None
    t_crack_vertical: float | None = None
    t_crack_radial: float | None = None
    t_structural_radial: float | None = None
    t_structural_vertical: float | None = None
    sf_vertical_crack: float | None = None
    sf_radial_crack: float | None = None
    sf_fast_fracture: float | None = None
    paris_coefficient: float | None = None
    paris_power_law: float | None = None
    walker_coefficient: float | None = None
    fracture_toughness: float | None = None

    # REBCO
    rebco_thickness: float | None = None
    copper_thick: float | None = None
    hastelloy_thickness: float | None = None
    tape_width: float | None = None
    tape_thickness: float | None = None
    croco_thick: float | None = None
    copper_rrr: float | None = None
    copper_m2_max: float | None = None
    f_coppera_m2: float | None = None
    copperaoh_m2_max: float | None = None
    f_copperaoh_m2: float | None = None

    # Primary pumping
    primary_pumping: int | None = None
    gamma_he: float | None = None
    t_in_bb: float | None = None
    t_out_bb: float | None = None
    p_he: float | None = None
    dp_he: float | None = None

    # Constraint variables
    auxmin: float | None = None
    beta_poloidal_max: float | None = None
    bigqmin: float | None = None
    bmxlim: float | None = None
    dr_tf_wp: float | None = None
    fauxmn: float | None = None
    fbeta: float | None = None
    fbeta_poloidal: float | None = None
    fbeta_max: float | None = None
    fbetatry_lower: float | None = None
    fcwr: float | None = None
    fdene: float | None = None
    fdivcol: float | None = None
    fdtmp: float | None = None
    fecrh_ignition: float | None = None
    fflutf: float | None = None
    ffuspow: float | None = None
    fgamcd: float | None = None
    fhldiv: float | None = None
    fiooic: float | None = None
    fipir: float | None = None
    fjohc: float | None = None
    fjohc0: float | None = None
    fjprot: float | None = None
    flhthresh: float | None = None
    fmva: float | None = None
    fnbshinef: float | None = None
    fncycle: float | None = None
    fnesep: float | None = None
    foh_stress: float | None = None
    fpeakb: float | None = None
    fpinj: float | None = None
    fpnetel: float | None = None
    fportsz: float | None = None
    fpsepbqar: float | None = None
    fpsepr: float | None = None
    fptemp: float | None = None
    fq: float | None = None
    fqval: float | None = None
    fradpwr: float | None = None
    fradwall: float | None = None
    freinke: float | None = None
    fstrcase: float | None = None
    fstrcond: float | None = None
    fstr_wp: float | None = None
    fmaxvvstress: float | None = None
    ftbr: float | None = None
    ft_burn: float | None = None
    ftcycl: float | None = None
    ftmargoh: float | None = None
    ftmargtf: float | None = None
    ft_current_ramp_up: float | None = None
    ftpeak: float | None = None
    fvdump: float | None = None
    fvs: float | None = None
    fvvhe: float | None = None
    fwalld: float | None = None
    fzeffmax: float | None = None
    gammax: float | None = None
    maxradwallload: float | None = None
    mvalim: float | None = None
    nbshinefmax: float | None = None
    nflutfmax: float | None = None
    pdivtlim: float | None = None
    peakfactrad: float | None = None
    pnetelin: float | None = None
    powfmax: float | None = None
    psepbqarmax: float | None = None
    pseprmax: float | None = None
    ptfnucmax: float | None = None
    tbrmin: float | None = None
    t_burn_min: float | None = None
    vvhealw: float | None = None
    walalw: float | None = None
    taulimit: float | None = None
    ftaulimit: float | None = None
    fniterpump: float | None = None
    zeffmax: float | None = None
    fpoloidalpower: float | None = None
    fpsep: float | None = None
    fcqt: float | None = None

    # Build variables
    aplasmin: float | None = None
    blbmith: float | None = None
    blbmoth: float | None = None
    blbpith: float | None = None
    blbpoth: float | None = None
    blbuith: float | None = None
    blbuoth: float | None = None
    blnkith: float | None = None
    blnkoth: float | None = None
    bore: float | None = None
    clhsf: float | None = None
    ddwex: float | None = None
    d_vv_in: float | None = None
    d_vv_out: float | None = None
    d_vv_top: float | None = None
    d_vv_bot: float | None = None
    f_avspace: float | None = None
    fcspc: float | None = None
    fhole: float | None = None
    fseppc: float | None = None
    gapds: float | None = None
    gapoh: float | None = None
    gapomin: float | None = None
    iohcl: int | None = None
    iprecomp: int | None = None
    ohcth: float | None = None
    rinboard: float | None = None
    f_r_cp: float | None = None
    scrapli: float | None = None
    scraplo: float | None = None
    shldith: float | None = None
    shldlth: float | None = None
    shldoth: float | None = None
    shldtth: float | None = None
    sigallpc: float | None = None
    tfoofti: float | None = None
    thshield_ib: float | None = None
    thshield_ob: float | None = None
    thshield_vb: float | None = None
    vgap_xpoint_divertor: float | None = None
    vgap_vv_thermalshield: float | None = None
    vgaptop: float | None = None
    vvblgap: float | None = None
    plleni: float | None = None
    plsepi: float | None = None
    plsepo: float | None = None
    tfootfi: float | None = None

    # Buildings

    # Current drive
    beamwd: float | None = None
    bootstrap_current_fraction_max: float | None = None
    cboot: float | None = None
    harnum: float | None = None
    beam_energy: float | None = None
    etaech: float | None = None
    etanbi: float | None = None
    feffcd: float | None = None
    frbeam: float | None = None
    f_tritium_beam: float | None = None
    gamma_ecrh: float | None = None
    xi_ebw: float | None = None
    iefrf: int | None = None
    wave_mode: int | None = None
    irfcf: int | None = None
    nbshield: float | None = None
    pheat: float | None = None  # Listed as an output
    pinjalw: float | None = None
    tbeamin: float | None = None

    # Impurity radiation
    coreradius: float | None = None
    coreradiationfraction: float | None = None
    fimp: list[float] | None = None

    # Reinke
    impvardiv: int | None = None
    lhat: float | None = None
    fzactual: float | None = None

    # Divertor
    divdum: int | None = None
    anginc: float | None = None
    beta_div: float | None = None
    betai: float | None = None
    betao: float | None = None
    bpsout: float | None = None
    c1div: float | None = None
    c2div: float | None = None
    c3div: float | None = None
    c4div: float | None = None
    c5div: float | None = None
    delld: float | None = None
    divclfr: float | None = None
    divdens: float | None = None
    divfix: float | None = None
    divplt: float | None = None
    fdfs: float | None = None
    fdiva: float | None = None
    fififi: float | None = None
    flux_exp: float | None = None
    frrp: float | None = None
    hldivlim: float | None = None
    ksic: float | None = None
    omegan: float | None = None
    prn1: float | None = None
    rlenmax: float | None = None
    tdiv: float | None = None
    xparain: float | None = None
    xpertin: float | None = None
    zeffdiv: float | None = None
    i_hldiv: int | None = None

    # Pulse
    bctmp: float | None = None
    dtstor: float | None = None
    istore: int | None = None
    itcycl: int | None = None
    lpulse: int | None = None  # Listed as an output

    # IFE

    # Heat transport
    baseel: float | None = None
    crypw_max: float | None = None
    f_crypmw: float | None = None
    etatf: float | None = None
    etath: float | None = None
    fpumpblkt: float | None = None
    fpumpdiv: float | None = None
    fpumpfw: float | None = None
    fpumpshld: float | None = None
    ipowerflow: int | None = None
    iprimshld: int | None = None
    pinjmax: float | None = None
    pwpm2: float | None = None
    trithtmw: float | None = None
    vachtmw: float | None = None
    irfcd: int | None = None
    # Water usage

    # Vacuum
    ntype: int | None = None
    pbase: float | None = None
    prdiv: float | None = None
    pumptp: float | None = None
    rat: float | None = None
    tn: float | None = None
    pumpareafraction: float | None = None
    pumpspeedmax: float | None = None
    pumpspeedfactor: float | None = None
    initialpressure: float | None = None
    outgasindex: float | None = None
    outgasfactor: float | None = None

    # PF coil
    alfapf: float | None = None
    alstroh: float | None = None
    coheof: float | None = None
    cptdin: list[float] | None = None
    etapsu: float | None = None
    fcohbop: float | None = None
    fcuohsu: float | None = None
    fcupfsu: float | None = None
    fvssu: float | None = None
    ipfloc: list[int] | None = None
    ipfres: int | None = None  # Listed as an output
    isumatoh: int | None = None
    isumatpf: int | None = None
    i_pf_current: int | None = None
    i_sup_pf_shape: int | None = None
    ncls: list[int] | None = None
    nfxfh: int | None = None
    ngrp: int | None = None
    ohhghf: float | None = None
    oh_steel_frac: float | None = None
    pfclres: float | None = None
    rjconpf: list[float] | None = None
    routr: float | None = None
    rpf2: float | None = None
    rref: list[float] | None = None
    sigpfcalw: float | None = None
    sigpfcf: float | None = None
    vf: list[float] | None = None
    vhohc: float | None = None
    zref: list[float] | None = None
    bmaxcs_lim: float | None = None
    fbmaxcs: float | None = None
    ld_ratio_cst: float | None = None

    # Physics
    alphaj: float | None = None
    alphan: float | None = None
    alphat: float | None = None
    aspect: float | None = None
    beamfus0: float | None = None
    beta: float | None = None
    betbm0: float | None = None
    bt: float | None = None
    csawth: float | None = None
    cvol: float | None = None
    cwrmax: float | None = None
    dene: float | None = None
    beta_norm_max: float | None = None
    epbetmax: float | None = None
    f_alpha_plasma: float | None = None
    f_deuterium: float | None = None
    ftar: float | None = None
    ffwal: float | None = None
    fgwped: float | None = None
    fgwsep: float | None = None
    fkzohm: float | None = None
    fpdivlim: float | None = None
    fne0: float | None = None
    f_tritium: float | None = None
    f_helium3: float | None = None
    fvsbrnni: float | None = None
    gamma: float | None = None
    hfact: float | None = None
    taumax: float | None = None
    i_bootstrap_current: int | None = None
    i_beta_component: int | None = None  # listed as an output...
    i_plasma_current: int | None = None
    i_density_limit: int | None = None
    i_diamagnetic_current: int | None = None
    i_beta_fast_alpha: int | None = None
    iinvqd: int | None = None
    ipedestal: int | None = None
    i_pfirsch_schluter_current: int | None = None
    eped_sf: float | None = None
    neped: float | None = None
    nesep: float | None = None
    plasma_res_factor: float | None = None
    rhopedn: float | None = None
    rhopedt: float | None = None
    tbeta: float | None = None
    teped: float | None = None
    tesep: float | None = None
    iprofile: int | None = None
    iradloss: int | None = None
    isc: int | None = None
    iscrp: int | None = None
    ishape: int | None = None  # listed as an output...
    itart: int | None = None  # listed as an output...
    itartpf: int | None = None  # listed as an output...
    iwalld: int | None = None
    kappa: float | None = None
    kappa95: float | None = None
    m_s_limit: float | None = None
    ilhthresh: int | None = None
    q: float | None = None
    q0: float | None = None
    tauratio: float | None = None
    rad_fraction_sol: float | None = None
    ralpne: float | None = None
    rli: float | None = None
    rmajor: float | None = None
    rnbeam: float | None = None
    i_single_null: int | None = None
    ssync: float | None = None
    te: float | None = None
    ti: float | None = None
    tratio: float | None = None
    triang: float | None = None
    triang95: float | None = None
    # Stellarator
    fblvd: float | None = None

    # first wall, blanket and
    # shield components variables
    fwcoolant: str | None = None
    icooldual: int | None = None
    ipump: int | None = None
    i_bb_liq: int | None = None
    ims: int | None = None
    ifci: int | None = None

    def __iter__(self) -> Iterator[tuple[str, float | list | dict]]:
        """
        Iterate over this dataclass

        The order is based on the order in which the values were
        declared.

        Yields
        ------
        :
            the field name and its value
        """
        for _field in fields(self):
            yield _field.name, getattr(self, _field.name)

    def to_invariable(self) -> dict[str, _INVariable]:
        """
        Wrap each value in an INVariable object

        Needed for compatibility with PROCESS InDat writer

        Returns
        -------
        :
            Converted input dictionary
        """
        out_dict = {}
        for name, value in self:
            if name not in {"icc", "ixc", "bounds"} and value is not None:
                new_val = _INVariable(name, value, "Parameter", "", "")
                out_dict[name] = new_val
        out_dict["icc"] = _INVariable(
            "icc",
            [] if self.icc is None else self.icc,
            "Constraint Equation",
            "Constraint Equation",
            "Constraint Equations",
        )
        # PROCESS iteration variables need to be sorted to converge well(!)
        out_dict["ixc"] = _INVariable(
            "ixc",
            [] if self.ixc is None else sorted(self.ixc),
            "Iteration Variable",
            "Iteration Variable",
            "Iteration Variables",
        )
        out_dict["bounds"] = _INVariable(
            "bounds",
            {} if self.bounds is None else self.bounds,
            "Bound",
            "Bound",
            "Bounds",
        )
        return out_dict

    def to_dict(self) -> dict[str, float | list | dict]:
        """
        Returns
        -------
        :
            A dictionary representation of the dataclass
        """
        return dict(self)
