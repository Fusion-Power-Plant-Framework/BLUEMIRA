# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Parameter classes/structures for Process
"""

from dataclasses import dataclass, fields
from typing import Dict, Generator, List, Optional, Tuple, Union

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

    runtitle: Optional[str] = None

    # Optimisation problem setup
    bounds: Optional[Dict[str, Dict[str, str]]] = None
    icc: Optional[List[int]] = None
    ixc: Optional[List[int]] = None

    # Settings
    maxcal: Optional[int] = None
    minmax: Optional[int] = None
    epsvmc: Optional[float] = None
    ioptimz: Optional[int] = None
    output_costs: Optional[int] = None
    isweep: Optional[int] = None
    nsweep: Optional[int] = None
    sweep: Optional[List[float]] = None
    pulsetimings: Optional[int] = None
    scan_dim: Optional[int] = None
    # Top down of PROCESS variables list

    # Times
    tburn: Optional[float] = None
    tdwell: Optional[float] = None
    t_fusion_ramp: Optional[float] = None
    tohs: Optional[float] = None
    tqnch: Optional[float] = None
    tramp: Optional[float] = None

    # FWBS
    ibkt_life: Optional[int] = None
    denstl: Optional[float] = None
    denw: Optional[float] = None
    emult: Optional[float] = None
    fblss: Optional[float] = None
    fdiv: Optional[float] = None
    fwbsshape: Optional[int] = None
    fw_armour_thickness: Optional[float] = None
    iblanket: Optional[int] = None
    iblnkith: Optional[int] = None
    li6enrich: Optional[float] = None
    breeder_f: Optional[float] = None
    breeder_multiplier: Optional[float] = None
    vfcblkt: Optional[float] = None
    vfpblkt: Optional[float] = None
    blktmodel: Optional[int] = None  # Listed as an output...
    # f_neut_shield: float = # -1.0 the documentation defaults cannot be right...
    breedmat: Optional[int] = None
    fblbe: Optional[float] = None
    fblbreed: Optional[float] = None
    fblhebmi: Optional[float] = None
    fblhebmo: Optional[float] = None
    fblhebpi: Optional[float] = None
    fblhebpo: Optional[float] = None
    hcdportsize: Optional[int] = None
    npdiv: Optional[int] = None
    nphcdin: Optional[int] = None
    nphcdout: Optional[int] = None
    wallpf: Optional[float] = None
    iblanket_thickness: Optional[int] = None
    secondary_cycle: Optional[int] = None  # Listed as an output...
    secondary_cycle_liq: Optional[int] = None
    afwi: Optional[float] = None
    afwo: Optional[float] = None
    fw_wall: Optional[float] = None
    afw: Optional[float] = None
    pitch: Optional[float] = None
    fwinlet: Optional[float] = None
    fwoutlet: Optional[float] = None
    fwpressure: Optional[float] = None
    roughness: Optional[float] = None
    fw_channel_length: Optional[float] = None
    peaking_factor: Optional[float] = None
    blpressure: Optional[float] = None
    inlet_temp: Optional[float] = None
    outlet_temp: Optional[float] = None
    coolp: Optional[float] = None
    coolwh: Optional[int] = None
    nblktmodpo: Optional[int] = None
    nblktmodpi: Optional[int] = None
    nblktmodto: Optional[int] = None
    nblktmodti: Optional[int] = None
    tfwmatmax: Optional[float] = None
    fw_th_conductivity: Optional[float] = None
    fvoldw: Optional[float] = None
    fvolsi: Optional[float] = None
    fvolso: Optional[float] = None
    fwclfr: Optional[float] = None
    rpf2dewar: Optional[float] = None
    vfshld: Optional[float] = None
    irefprop: Optional[int] = None
    fblli2o: Optional[float] = None
    fbllipb: Optional[float] = None
    vfblkt: Optional[float] = None
    declblkt: Optional[float] = None
    declfw: Optional[float] = None
    declshld: Optional[float] = None
    blkttype: Optional[int] = None
    etaiso: Optional[float] = None
    etahtp: Optional[float] = None
    n_liq_recirc: Optional[int] = None
    bz_channel_conduct_liq: Optional[float] = None
    blpressure_liq: Optional[float] = None
    inlet_temp_liq: Optional[float] = None
    outlet_temp_liq: Optional[float] = None
    f_nuc_pow_bz_struct: Optional[float] = None
    pnuc_fw_ratio_dcll: Optional[float] = None

    # TF coil
    sig_tf_case_max: Optional[float] = None
    sig_tf_wp_max: Optional[float] = None
    bcritsc: Optional[float] = None
    casthi_fraction: Optional[float] = None
    casths_fraction: Optional[float] = None
    f_t_turn_tf: Optional[float] = None
    t_turn_tf_max: Optional[float] = None
    cpttf: Optional[float] = None
    cpttf_max: Optional[float] = None
    dcase: Optional[float] = None
    dcond: Optional[List[float]] = None
    dcondins: Optional[float] = None
    dhecoil: Optional[float] = None
    b_crit_upper_nbti: Optional[float] = None
    t_crit_nbti: Optional[float] = None
    fcutfsu: Optional[float] = None
    fhts: Optional[float] = None
    i_tf_stress_model: Optional[int] = None
    i_tf_wp_geom: Optional[int] = None
    i_tf_case_geom: Optional[int] = None  # Listed as an output
    i_tf_turns_integer: Optional[int] = None  # Listed as an output
    i_tf_sc_mat: Optional[int] = None
    i_tf_sup: Optional[int] = None
    i_tf_shape: Optional[int] = None  # Listed as an output
    i_tf_cond_eyoung_trans: Optional[int] = None
    i_r_cp_top: Optional[int] = None
    i_tf_tresca: Optional[int] = None
    n_pancake: Optional[int] = None
    n_layer: Optional[int] = None
    n_rad_per_layer: Optional[int] = None
    i_tf_bucking: Optional[int] = None
    n_tf_graded_layers: Optional[int] = None
    jbus: Optional[float] = None
    eyoung_ins: Optional[float] = None
    eyoung_steel: Optional[float] = None
    eyong_cond_axial: Optional[float] = None
    eyoung_res_tf_buck: Optional[float] = None
    # eyoung_al: Optional[float] = 69000000000.0 # defaults  cannot be right
    poisson_steel: Optional[float] = None
    poisson_copper: Optional[float] = None
    poisson_al: Optional[float] = None
    str_cs_con_res: Optional[float] = None
    str_pf_con_res: Optional[float] = None
    str_tf_con_res: Optional[float] = None
    str_wp_max: Optional[float] = None
    i_str_wp: Optional[int] = None
    quench_model: str = None
    tcritsc: Optional[float] = None
    tdmptf: Optional[float] = None
    tfinsgap: Optional[float] = None
    # rhotfbus: Optional[float] = -1.0 # defaults cannot be right
    frhocp: Optional[float] = None
    frholeg: Optional[float] = None
    i_cp_joints: Optional[int] = None
    rho_tf_joints: Optional[float] = None
    n_tf_joints_contact: Optional[int] = None
    n_tf_joints: Optional[int] = None
    th_joint_contact: Optional[float] = None
    # eff_tf_cryo: Optional[float] = -1.0 # defaults cannot be right
    n_tf: Optional[int] = None
    tftmp: Optional[float] = None
    thicndut: Optional[float] = None
    thkcas: Optional[float] = None
    thwcndut: Optional[float] = None
    tinstf: Optional[float] = None
    tmaxpro: Optional[float] = None
    tmax_croco: Optional[float] = None
    tmpcry: Optional[float] = None
    vdalw: Optional[float] = None
    f_vforce_inboard: Optional[float] = None
    vftf: Optional[float] = None
    etapump: Optional[float] = None
    fcoolcp: Optional[float] = None
    fcoolleg: Optional[float] = None
    ptempalw: Optional[float] = None
    rcool: Optional[float] = None
    tcoolin: Optional[float] = None
    tcpav: Optional[float] = None
    vcool: Optional[float] = None
    theta1_coil: Optional[float] = None
    theta1_vv: Optional[float] = None
    max_vv_stress: Optional[float] = None
    inuclear: Optional[int] = None
    qnuc: Optional[float] = None
    ripmax: Optional[float] = None
    tf_in_cs: Optional[int] = None
    tfcth: Optional[float] = None
    tftsgap: Optional[float] = None
    casthi: Optional[float] = None
    casths: Optional[float] = None
    tmargmin: Optional[float] = None
    oacdcp: Optional[float] = None
    t_turn_tf: Optional[int] = None

    # PF Power
    iscenr: Optional[int] = None
    maxpoloidalpower: Optional[float] = None

    # Cost variables
    abktflnc: Optional[float] = None
    adivflnc: Optional[float] = None
    cconfix: Optional[float] = None
    cconshpf: Optional[float] = None
    cconshtf: Optional[float] = None
    cfactr: Optional[float] = None
    cfind: Optional[List[float]] = None
    cland: Optional[float] = None
    costexp: Optional[float] = None
    costexp_pebbles: Optional[float] = None
    cost_factor_buildings: Optional[float] = None
    cost_factor_land: Optional[float] = None
    cost_factor_tf_coils: Optional[float] = None
    cost_factor_fwbs: Optional[float] = None
    cost_factor_tf_rh: Optional[float] = None
    cost_factor_tf_vv: Optional[float] = None
    cost_factor_tf_bop: Optional[float] = None
    cost_factor_tf_misc: Optional[float] = None
    maintenance_fwbs: Optional[float] = None
    maintenance_gen: Optional[float] = None
    amortization: Optional[float] = None
    cost_model: Optional[int] = None
    cowner: Optional[float] = None
    cplife_input: Optional[float] = None
    cpstflnc: Optional[float] = None
    csi: Optional[float] = None
    # cturbb: Optional[float] = 38.0 # defaults cannot be right
    decomf: Optional[float] = None
    dintrt: Optional[float] = None
    fcap0: Optional[float] = None
    fcap0cp: Optional[float] = None
    fcdfuel: Optional[float] = None
    fcontng: Optional[float] = None
    fcr0: Optional[float] = None
    fkind: Optional[float] = None
    iavail: Optional[int] = None
    life_dpa: Optional[float] = None
    avail_min: Optional[float] = None
    favail: Optional[float] = None
    num_rh_systems: Optional[int] = None
    conf_mag: Optional[float] = None
    div_prob_fail: Optional[float] = None
    div_umain_time: Optional[float] = None
    div_nref: Optional[float] = None
    div_nu: Optional[float] = None
    fwbs_nref: Optional[float] = None
    fwbs_nu: Optional[float] = None
    fwbs_prob_fail: Optional[float] = None
    fwbs_umain_time: Optional[float] = None
    redun_vacp: Optional[float] = None
    tbktrepl: Optional[float] = None
    tcomrepl: Optional[float] = None
    tdivrepl: Optional[float] = None
    uubop: Optional[float] = None
    uucd: Optional[float] = None
    uudiv: Optional[float] = None
    uufuel: Optional[float] = None
    uufw: Optional[float] = None
    uumag: Optional[float] = None
    uuves: Optional[float] = None
    ifueltyp: Optional[int] = None
    ucblvd: Optional[float] = None
    ucdiv: Optional[float] = None
    ucme: Optional[float] = None
    ireactor: Optional[int] = None
    lsa: Optional[int] = None
    discount_rate: Optional[float] = None
    startupratio: Optional[float] = None
    tlife: Optional[float] = None
    bkt_life_csf: Optional[int] = None
    i_bldgs_size: Optional[int] = None
    # ...

    # CS fatigue
    residual_sig_hoop: Optional[float] = None
    n_cycle_min: Optional[int] = None
    t_crack_vertical: Optional[float] = None
    t_crack_radial: Optional[float] = None
    t_structural_radial: Optional[float] = None
    t_structural_vertical: Optional[float] = None
    sf_vertical_crack: Optional[float] = None
    sf_radial_crack: Optional[float] = None
    sf_fast_fracture: Optional[float] = None
    paris_coefficient: Optional[float] = None
    paris_power_law: Optional[float] = None
    walker_coefficient: Optional[float] = None
    fracture_toughness: Optional[float] = None

    # REBCO
    rebco_thickness: Optional[float] = None
    copper_thick: Optional[float] = None
    hastelloy_thickness: Optional[float] = None
    tape_width: Optional[float] = None
    tape_thickness: Optional[float] = None
    croco_thick: Optional[float] = None
    copper_rrr: Optional[float] = None
    copper_m2_max: Optional[float] = None
    f_coppera_m2: Optional[float] = None
    copperaoh_m2_max: Optional[float] = None
    f_copperaoh_m2: Optional[float] = None

    # Primary pumping
    primary_pumping: Optional[int] = None
    gamma_he: Optional[float] = None
    t_in_bb: Optional[float] = None
    t_out_bb: Optional[float] = None
    p_he: Optional[float] = None
    dp_he: Optional[float] = None

    # Constraint variables
    auxmin: Optional[float] = None
    betpmx: Optional[float] = None
    bigqmin: Optional[float] = None
    bmxlim: Optional[float] = None
    dr_tf_wp: Optional[float] = None
    fauxmn: Optional[float] = None
    fbeta: Optional[float] = None
    fbetap: Optional[float] = None
    fbetatry: Optional[float] = None
    fbetatry_lower: Optional[float] = None
    fcwr: Optional[float] = None
    fdene: Optional[float] = None
    fdivcol: Optional[float] = None
    fdtmp: Optional[float] = None
    fecrh_ignition: Optional[float] = None
    fflutf: Optional[float] = None
    ffuspow: Optional[float] = None
    fgamcd: Optional[float] = None
    fhldiv: Optional[float] = None
    fiooic: Optional[float] = None
    fipir: Optional[float] = None
    fjohc: Optional[float] = None
    fjohc0: Optional[float] = None
    fjprot: Optional[float] = None
    flhthresh: Optional[float] = None
    fmva: Optional[float] = None
    fnbshinef: Optional[float] = None
    fncycle: Optional[float] = None
    fnesep: Optional[float] = None
    foh_stress: Optional[float] = None
    fpeakb: Optional[float] = None
    fpinj: Optional[float] = None
    fpnetel: Optional[float] = None
    fportsz: Optional[float] = None
    fpsepbqar: Optional[float] = None
    fpsepr: Optional[float] = None
    fptemp: Optional[float] = None
    fq: Optional[float] = None
    fqval: Optional[float] = None
    fradpwr: Optional[float] = None
    fradwall: Optional[float] = None
    freinke: Optional[float] = None
    fstrcase: Optional[float] = None
    fstrcond: Optional[float] = None
    fstr_wp: Optional[float] = None
    fmaxvvstress: Optional[float] = None
    ftbr: Optional[float] = None
    ftburn: Optional[float] = None
    ftcycl: Optional[float] = None
    ftmargoh: Optional[float] = None
    ftmargtf: Optional[float] = None
    ftohs: Optional[float] = None
    ftpeak: Optional[float] = None
    fvdump: Optional[float] = None
    fvs: Optional[float] = None
    fvvhe: Optional[float] = None
    fwalld: Optional[float] = None
    fzeffmax: Optional[float] = None
    gammax: Optional[float] = None
    maxradwallload: Optional[float] = None
    mvalim: Optional[float] = None
    nbshinefmax: Optional[float] = None
    nflutfmax: Optional[float] = None
    pdivtlim: Optional[float] = None
    peakfactrad: Optional[float] = None
    pnetelin: Optional[float] = None
    powfmax: Optional[float] = None
    psepbqarmax: Optional[float] = None
    pseprmax: Optional[float] = None
    ptfnucmax: Optional[float] = None
    tbrmin: Optional[float] = None
    tbrnmn: Optional[float] = None
    vvhealw: Optional[float] = None
    walalw: Optional[float] = None
    taulimit: Optional[float] = None
    ftaulimit: Optional[float] = None
    fniterpump: Optional[float] = None
    zeffmax: Optional[float] = None
    fpoloidalpower: Optional[float] = None
    fpsep: Optional[float] = None
    fcqt: Optional[float] = None

    # Build variables
    aplasmin: Optional[float] = None
    blbmith: Optional[float] = None
    blbmoth: Optional[float] = None
    blbpith: Optional[float] = None
    blbpoth: Optional[float] = None
    blbuith: Optional[float] = None
    blbuoth: Optional[float] = None
    blnkith: Optional[float] = None
    blnkoth: Optional[float] = None
    bore: Optional[float] = None
    clhsf: Optional[float] = None
    ddwex: Optional[float] = None
    d_vv_in: Optional[float] = None
    d_vv_out: Optional[float] = None
    d_vv_top: Optional[float] = None
    d_vv_bot: Optional[float] = None
    f_avspace: Optional[float] = None
    fcspc: Optional[float] = None
    fhole: Optional[float] = None
    fseppc: Optional[float] = None
    gapds: Optional[float] = None
    gapoh: Optional[float] = None
    gapomin: Optional[float] = None
    iohcl: Optional[int] = None
    iprecomp: Optional[int] = None
    ohcth: Optional[float] = None
    rinboard: Optional[float] = None
    f_r_cp: Optional[float] = None
    scrapli: Optional[float] = None
    scraplo: Optional[float] = None
    shldith: Optional[float] = None
    shldlth: Optional[float] = None
    shldoth: Optional[float] = None
    shldtth: Optional[float] = None
    sigallpc: Optional[float] = None
    tfoofti: Optional[float] = None
    thshield_ib: Optional[float] = None
    thshield_ob: Optional[float] = None
    thshield_vb: Optional[float] = None
    vgap: Optional[float] = None
    vgap2: Optional[float] = None
    vgaptop: Optional[float] = None
    vvblgap: Optional[float] = None
    plleni: Optional[float] = None
    plsepi: Optional[float] = None
    plsepo: Optional[float] = None
    tfootfi: Optional[float] = None

    # Buildings

    # Current drive
    beamwd: Optional[float] = None
    bscfmax: Optional[float] = None
    cboot: Optional[float] = None
    harnum: Optional[float] = None
    enbeam: Optional[float] = None
    etaech: Optional[float] = None
    etanbi: Optional[float] = None
    feffcd: Optional[float] = None
    frbeam: Optional[float] = None
    ftritbm: Optional[float] = None
    gamma_ecrh: Optional[float] = None
    xi_ebw: Optional[float] = None
    iefrf: Optional[int] = None
    irfcf: Optional[int] = None
    nbshield: Optional[float] = None
    pheat: Optional[float] = None  # Listed as an output
    pinjalw: Optional[float] = None
    tbeamin: Optional[float] = None

    # Impurity radiation
    coreradius: Optional[float] = None
    coreradiationfraction: Optional[float] = None
    fimp: Optional[List[float]] = None

    # Reinke
    impvardiv: Optional[int] = None
    lhat: Optional[float] = None
    fzactual: Optional[float] = None

    # Divertor
    divdum: Optional[int] = None
    anginc: Optional[float] = None
    beta_div: Optional[float] = None
    betai: Optional[float] = None
    betao: Optional[float] = None
    bpsout: Optional[float] = None
    c1div: Optional[float] = None
    c2div: Optional[float] = None
    c3div: Optional[float] = None
    c4div: Optional[float] = None
    c5div: Optional[float] = None
    delld: Optional[float] = None
    divclfr: Optional[float] = None
    divdens: Optional[float] = None
    divfix: Optional[float] = None
    divplt: Optional[float] = None
    fdfs: Optional[float] = None
    fdiva: Optional[float] = None
    fififi: Optional[float] = None
    flux_exp: Optional[float] = None
    frrp: Optional[float] = None
    hldivlim: Optional[float] = None
    ksic: Optional[float] = None
    omegan: Optional[float] = None
    prn1: Optional[float] = None
    rlenmax: Optional[float] = None
    tdiv: Optional[float] = None
    xparain: Optional[float] = None
    xpertin: Optional[float] = None
    zeffdiv: Optional[float] = None
    i_hldiv: Optional[int] = None

    # Pulse
    bctmp: Optional[float] = None
    dtstor: Optional[float] = None
    istore: Optional[int] = None
    itcycl: Optional[int] = None
    lpulse: Optional[int] = None  # Listed as an output

    # IFE

    # Heat transport
    baseel: Optional[float] = None
    crypw_max: Optional[float] = None
    f_crypmw: Optional[float] = None
    etatf: Optional[float] = None
    etath: Optional[float] = None
    fpumpblkt: Optional[float] = None
    fpumpdiv: Optional[float] = None
    fpumpfw: Optional[float] = None
    fpumpshld: Optional[float] = None
    ipowerflow: Optional[int] = None
    iprimshld: Optional[int] = None
    pinjmax: Optional[float] = None
    pwpm2: Optional[float] = None
    trithtmw: Optional[float] = None
    vachtmw: Optional[float] = None
    irfcd: Optional[int] = None
    # Water usage

    # Vacuum
    ntype: Optional[int] = None
    pbase: Optional[float] = None
    prdiv: Optional[float] = None
    pumptp: Optional[float] = None
    rat: Optional[float] = None
    tn: Optional[float] = None
    pumpareafraction: Optional[float] = None
    pumpspeedmax: Optional[float] = None
    pumpspeedfactor: Optional[float] = None
    initialpressure: Optional[float] = None
    outgasindex: Optional[float] = None
    outgasfactor: Optional[float] = None

    # PF coil
    alfapf: Optional[float] = None
    alstroh: Optional[float] = None
    coheof: Optional[float] = None
    cptdin: Optional[List[float]] = None
    etapsu: Optional[float] = None
    fcohbop: Optional[float] = None
    fcuohsu: Optional[float] = None
    fcupfsu: Optional[float] = None
    fvssu: Optional[float] = None
    ipfloc: Optional[List[int]] = None
    ipfres: Optional[int] = None  # Listed as an output
    isumatoh: Optional[int] = None
    isumatpf: Optional[int] = None
    i_pf_current: Optional[int] = None
    ncls: Optional[List[int]] = None
    nfxfh: Optional[int] = None
    ngrp: Optional[int] = None
    ohhghf: Optional[float] = None
    oh_steel_frac: Optional[float] = None
    pfclres: Optional[float] = None
    rjconpf: Optional[List[float]] = None
    routr: Optional[float] = None
    rpf2: Optional[float] = None
    rref: Optional[List[float]] = None
    sigpfcalw: Optional[float] = None
    sigpfcf: Optional[float] = None
    vf: Optional[List[float]] = None
    vhohc: Optional[float] = None
    zref: Optional[List[float]] = None
    bmaxcs_lim: Optional[float] = None
    fbmaxcs: Optional[float] = None
    ld_ratio_cst: Optional[float] = None

    # Physics
    alphaj: Optional[float] = None
    alphan: Optional[float] = None
    alphat: Optional[float] = None
    aspect: Optional[float] = None
    beamfus0: Optional[float] = None
    beta: Optional[float] = None
    betbm0: Optional[float] = None
    bt: Optional[float] = None
    csawth: Optional[float] = None
    cvol: Optional[float] = None
    cwrmax: Optional[float] = None
    dene: Optional[float] = None
    dnbeta: Optional[float] = None
    epbetmax: Optional[float] = None
    falpha: Optional[float] = None
    fdeut: Optional[float] = None
    ftar: Optional[float] = None
    ffwal: Optional[float] = None
    fgwped: Optional[float] = None
    fgwsep: Optional[float] = None
    fkzohm: Optional[float] = None
    fpdivlim: Optional[float] = None
    fne0: Optional[float] = None
    ftrit: Optional[float] = None
    fvsbrnni: Optional[float] = None
    gamma: Optional[float] = None
    hfact: Optional[float] = None
    taumax: Optional[float] = None
    ibss: Optional[int] = None
    iculbl: Optional[int] = None  # listed as an output...
    icurr: Optional[int] = None
    idensl: Optional[int] = None
    idia: Optional[int] = None
    ifalphap: Optional[int] = None
    iinvqd: Optional[int] = None
    ipedestal: Optional[int] = None
    ieped: Optional[int] = None  # listed as an output...
    ips: Optional[int] = None
    eped_sf: Optional[float] = None
    neped: Optional[float] = None
    nesep: Optional[float] = None
    plasma_res_factor: Optional[float] = None
    rhopedn: Optional[float] = None
    rhopedt: Optional[float] = None
    tbeta: Optional[float] = None
    teped: Optional[float] = None
    tesep: Optional[float] = None
    iprofile: Optional[int] = None
    iradloss: Optional[int] = None
    isc: Optional[int] = None
    iscrp: Optional[int] = None
    ishape: Optional[int] = None  # listed as an output...
    itart: Optional[int] = None  # listed as an output...
    itartpf: Optional[int] = None  # listed as an output...
    iwalld: Optional[int] = None
    kappa: Optional[float] = None
    kappa95: Optional[float] = None
    m_s_limit: Optional[float] = None
    ilhthresh: Optional[int] = None
    q95: Optional[float] = None
    q0: Optional[float] = None
    tauratio: Optional[float] = None
    rad_fraction_sol: Optional[float] = None
    ralpne: Optional[float] = None
    rli: Optional[float] = None
    rmajor: Optional[float] = None
    rnbeam: Optional[float] = None
    i_single_null: Optional[int] = None
    ssync: Optional[float] = None
    te: Optional[float] = None
    ti: Optional[float] = None
    tratio: Optional[float] = None
    triang: Optional[float] = None
    triang95: Optional[float] = None
    # Stellarator
    fblvd: Optional[float] = None

    # first wall, blanket and
    # shield components variables
    fwcoolant: Optional[str] = None
    icooldual: Optional[int] = None
    ipump: Optional[int] = None
    i_bb_liq: Optional[int] = None
    ims: Optional[int] = None
    ifci: Optional[int] = None

    def __iter__(self) -> Generator[Tuple[str, Union[float, List, Dict]], None, None]:
        """
        Iterate over this dataclass

        The order is based on the order in which the values were
        declared.
        """
        for _field in fields(self):
            yield _field.name, getattr(self, _field.name)

    def to_invariable(self) -> Dict[str, _INVariable]:
        """
        Wrap each value in an INVariable object

        Needed for compatibility with PROCESS InDat writer
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

    def to_dict(self) -> Dict[str, Union[float, List, Dict]]:
        """
        A dictionary representation of the dataclass

        """
        return dict(self)
