# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Death to PROCESS integers
"""

from dataclasses import dataclass, field

from bluemira.codes.utilities import Model


class Objective(Model):
    """
    Enum for PROCESS optimisation objective
    """

    MAJOR_RADIUS = 1
    # 2 NOT USED
    NEUTRON_WALL_LOAD = 3
    MAX_COIL_POWER = 4
    FUSION_GAIN = 5
    ELECTRICITY_COST = 6
    CAPITAL_COST = 7
    ASPECT_RATIO = 8
    DIVERTOR_HEAT_LOAD = 9
    TOROIDAL_FIELD = 10
    INJECTED_POWER = 11
    # 12, 13 NOT USED
    PULSE_LENGTH = 14
    AVAILABILITY = 15
    MAJOR_RADIUS_PULSE_LENGTH = 16
    NET_ELECTRICITY = 17
    NULL = 18
    FUSION_GAIN_PULSE_LENGTH = 19


OBJECTIVE_MIN_ONLY = (16, 19)


@dataclass
class ConstraintSelection:
    """
    Mixin dataclass for a Constraint selection in PROCESSModel

    Parameters
    ----------
    _value_:
        Integer value of the constraint
    requires_variables:
        List of required iteration variables for the constraint
    requires_values:
        List of required inputs for the constraint
    description:
        Short description of the model constraint
    """

    _value_: int
    requires_variables: tuple[int] = field(default_factory=tuple)
    requires_values: tuple[str] = field(default_factory=tuple)
    description: str = ""


class Constraint(ConstraintSelection, Model):
    """
    Enum for PROCESS constraints
    """

    BETA_CONSISTENCY = 1, (5,), (), "Beta consistency"
    GLOBAL_POWER_CONSISTENCY = (
        2,
        (1, 2, 3, 4, 6, 10, 11),
        (),
        "Global Power Balance Consistency",
    )
    ION_POWER_CONSISTENCY = (
        3,
        (1, 2, 3, 4, 6, 10, 11),
        (),
        "DEPRECATED - Ion Power Balance Consistency",
    )
    ELECTRON_POWER_CONSISTENCY = (
        4,
        (1, 2, 3, 4, 6, 10, 11),
        (),
        "DEPRECATED - Electron Power Balance Consistency",
    )
    DENSITY_UPPER_LIMIT = (
        5,
        (1, 2, 3, 4, 6, 9),
        (),
        "Density Upper Limit (Greenwald)",
    )
    EPS_BETA_POL_UPPER_LIMIT = (
        6,
        (1, 2, 3, 4, 6, 8),
        ("epbetmax",),
        "Equation for epsilon beta-poloidal upper limit",
    )
    HOT_BEAM_ION_DENSITY = 7, (7,), (), "Equation for hot beam ion density"
    NWL_UPPER_LIMIT = (
        8,
        (1, 2, 3, 4, 6, 14),
        ("walalw",),
        "Neutron wall load upper limit",
    )
    FUSION_POWER_UPPER_LIMIT = (
        9,
        (1, 2, 3, 4, 6, 26),
        ("powfmax",),
        "Equation for fusion power upper limit",
    )
    # 10 NOT USED
    RADIAL_BUILD_CONSISTENCY = (
        11,
        (1, 3, 13, 16, 29, 42, 61),
        (),
        "Radial Build Consistency",
    )
    VS_LOWER_LIMIT = (
        12,
        (1, 2, 3, 15),
        (),
        "Equation for volt-second capability lower limit",
    )
    BURN_TIME_LOWER_LIMIT = (
        13,
        (1, 2, 3, 6, 16, 17, 19, 29, 42, 44, 61),
        (),
        "Burn time lower limit",
    )
    NBI_LAMBDA_CENTRE = (
        14,
        (),
        (),
        "Equation to fix number of NBI decay lengths to plasma centre",
    )
    LH_THRESHHOLD_LIMIT = 15, (103,), (), "L-H Power ThresHhold Limit"
    NET_ELEC_LOWER_LIMIT = (
        16,
        (1, 2, 3, 25),
        ("pnetelin",),
        "Net electric power lower limit",
    )
    RAD_POWER_UPPER_LIMIT = 17, (28,), (), "Equation for radiation power upper limit"
    DIVERTOR_HEAT_UPPER_LIMIT = (
        18,
        (27),
        (),
        "Equation for divertor heat load upper limit",
    )
    MVA_UPPER_LIMIT = 19, (30,), ("mvalim",), "Equation for MVA upper limit"
    NBI_TANGENCY_UPPER_LIMIT = (
        20,
        (3, 13, 31, 33),
        (),
        "Equation for neutral beam tangency radius upper limit",
    )
    AMINOR_LOWER_LIMIT = 21, (32,), (), "Equation for minor radius lower limit"
    DIV_COLL_CONN_UPPER_LIMIT = (
        22,
        (34,),
        (),
        "Equation for divertor collision/connection length ratio upper limit",
    )
    COND_SHELL_R_RATIO_UPPER_LIMIT = (
        23,
        (1, 74, 104),
        ("cwrmax",),
        "Equation for conducting shell radius / rminor upper limit",
    )
    BETA_UPPER_LIMIT = 24, (1, 2, 3, 4, 6, 18, 36), (), "Beta Upper Limit"
    PEAK_TF_UPPER_LIMIT = (
        25,
        (3, 13, 29, 35),
        ("bmxlim",),
        "Peak toroidal field upper limit",
    )
    CS_EOF_DENSITY_LIMIT = (
        26,
        (37, 38, 41),
        (),
        "Central solenoid EOF current density upper limit",
    )
    CS_BOP_DENSITY_LIMIT = (
        27,
        (37, 38, 41),
        (),
        "Central solenoid bop current density upper limit",
    )
    Q_LOWER_LIMIT = (
        28,
        (40, 45, 47),
        ("bigqmin",),
        "Equation for fusion gain (big Q) lower limit",
    )
    IB_RADIAL_BUILD_CONSISTENCY = (
        29,
        (1, 3, 13, 16, 29, 42, 61),
        (),
        "Equation for minor radius lower limit OR Inboard radial build consistency",
    )
    PINJ_UPPER_LIMIT = 30, (11, 46, 47), ("pinjalw",), "Injection Power Upper Limit"
    TF_CASE_STRESS_UPPER_LIMIT = (
        31,
        (48, 56, 57, 58, 59, 60),
        ("sig_tf_case_max",),
        "TF coil case stress upper limit",
    )
    TF_JACKET_STRESS_UPPER_LIMIT = (
        32,
        (49, 56, 57, 58, 59, 60),
        ("sig_tf_wp_max",),
        "TF WP steel jacket/conduit stress upper limit",
    )
    TF_JCRIT_RATIO_UPPER_LIMIT = (
        33,
        (50, 56, 57, 58, 59, 60),
        (),
        "TF superconductor operating current / critical current density",
    )
    TF_DUMP_VOLTAGE_UPPER_LIMIT = (
        34,
        (51, 56, 57, 58, 59, 60),
        ("vdalw",),
        "TF dump voltage upper limit",
    )
    TF_CURRENT_DENSITY_UPPER_LIMIT = (
        35,
        (53, 56, 57, 58, 59, 60),
        (),
        "TF winding pack current density upper limit",
    )
    TF_T_MARGIN_LOWER_LIMIT = (
        36,
        (54, 56, 57, 58, 59, 60),
        ("tftmp",),
        "TF temperature margin upper limit",
    )
    CD_GAMMA_UPPER_LIMIT = (
        37,
        (40, 47),
        ("gammax",),
        "Equation for current drive gamma upper limit",
    )
    # 38 NOT USED
    FW_TEMP_UPPER_LIMIT = 39, (63,), (), "First wall peak temperature upper limit"
    PAUX_LOWER_LIMIT = (
        40,
        (64,),
        ("auxmin",),
        "Start-up injection power upper limit (PULSE)",
    )
    IP_RAMP_LOWER_LIMIT = (
        41,
        (65, 66),
        ("tohsmn",),
        "Plasma ramp-up time lower limit (PULSE)",
    )
    CYCLE_TIME_LOWER_LIMIT = (
        42,
        (17, 65, 67),
        ("tcycmn",),
        "Cycle time lower limit (PULSE)",
    )
    CENTREPOST_TEMP_AVERAGE = (
        43,
        (13, 20, 69, 70),
        (),
        "Average centrepost temperature (TART) consistency equation",
    )
    CENTREPOST_TEMP_UPPER_LIMIT = (
        44,
        (68, 69, 70),
        ("ptempalw",),
        "Peak centrepost temperature upper limit (TART)",
    )
    QEDGE_LOWER_LIMIT = 45, (1, 2, 3, 70), (), "Edge safety factor lower limit (TART)"
    IP_IROD_UPPER_LIMIT = 46, (2, 60, 72), (), "Equation for Ip/Irod upper limit (TART)"
    # 47 NOT USED (or maybe it is, WTF?!)
    BETAPOL_UPPER_LIMIT = 48, (2, 3, 18, 79), ("betpmax",), "Poloidal beta upper limit"
    # 49 NOT USED
    REP_RATE_UPPER_LIMIT = 50, (86,), (), "IFE repetition rate upper limit (IFE)"
    CS_FLUX_CONSISTENCY = (
        51,
        (1, 3, 16, 29),
        (),
        "Startup volt-seconds consistency (PULSE)",
    )
    TBR_LOWER_LIMIT = 52, (89, 90, 91), ("tbrmin",), "Tritium breeding ratio lower limit"
    NFLUENCE_TF_UPPER_LIMIT = (
        53,
        (92, 93, 94),
        ("nflutfmax",),
        "Neutron fluence on TF coil upper limit",
    )
    PNUCL_TF_UPPER_LIMIT = (
        54,
        (93, 94, 95),
        ("ptfnucmax",),
        "Peak TF coil nuclear heating upper limit",
    )
    HE_VV_UPPER_LIMIT = (
        55,
        (93, 94, 96),
        ("vvhealw",),
        "Vacuum vessel helium concentration upper limit iblanket=2",
    )
    PSEPR_UPPER_LIMIT = (
        56,
        (1, 3, 97),
        ("pseprmax",),
        "Pseparatrix/Rmajor upper limit",
    )
    # 57, 58 NOT USED
    NBI_SHINETHROUGH_UPPER_LIMIT = (
        59,
        (4, 6, 19, 105),
        ("nbshinefmax",),
        "Neutral beam shinethrough fraction upper limit (NBI)",
    )
    CS_T_MARGIN_LOWER_LIMIT = (
        60,
        (106,),
        (),
        "Central solenoid temperature margin lower limit (SCTF)[sic.."
        " I guess they mean SCCS]",
    )
    AVAIL_LOWER_LIMIT = 61, (107,), ("avail_min",), "Minimum availability value"
    CONFINEMENT_RATIO_LOWER_LIMIT = (
        62,
        (110,),
        ("taulimit",),
        "taup/taueff the ratio of particle to energy confinement times",
    )
    NITERPUMP_UPPER_LIMIT = (
        63,
        (111,),
        (),
        "The number of ITER-like vacuum pumps niterpump < tfno",
    )
    ZEFF_UPPER_LIMIT = 64, (112,), ("zeffmax",), "Zeff less than or equal to zeffmax"
    DUMP_TIME_LOWER_LIMIT = (
        65,
        (56, 113),
        ("max_vv_stress",),
        "Dump time set by VV loads",
    )
    PF_ENERGY_RATE_UPPER_LIMIT = (
        66,
        (65, 113),
        ("t_current_ramp_up",),
        "Limit on rate of change of energy in poloidal field",
    )
    WALL_RADIATION_UPPER_LIMIT = (
        67,
        (4, 6, 116),
        ("peakfactrad", "peakradwallload"),
        "Simple radiation wall load limit",
    )
    PSEPB_QAR_UPPER_LIMIT = (
        68,
        (117,),
        ("psepbqarmax",),
        "P_separatrix Bt / q A R upper limit",
    )
    PSEP_KALLENBACH_UPPER_LIMIT = (
        69,
        (118,),
        (),
        "ensure the separatrix power = the value from Kallenbach divertor",
    )
    TSEP_CONSISTENCY = (
        70,
        (119,),
        (),
        "ensure that temp = separatrix in the pedestal profile",
    )
    NSEP_CONSISTENCY = (
        71,
        (),
        (),
        "ensure that neomp = separatrix density (nesep) x neratio",
    )
    CS_STRESS_UPPER_LIMIT = (
        72,
        (123,),
        (),
        "Central solenoid shear stress limit (Tresca yield criterion)",
    )
    PSEP_LH_AUX_CONSISTENCY = 73, (137,), (), "Psep >= Plh + Paux"
    TF_CROCO_T_UPPER_LIMIT = 74, (141,), ("tmax_croco",), "TFC quench"
    TF_CROCO_CU_AREA_CONSTRAINT = (
        75,
        (143,),
        ("coppera_m2_max",),
        "TFC current / copper area < maximum",
    )
    EICH_SEP_DENSITY_CONSTRAINT = 76, (144,), (), "Eich critical separatrix density"
    TF_TURN_CURRENT_UPPER_LIMIT = (
        77,
        (146,),
        ("cpttf_max",),
        "TF coil current per turn upper limit",
    )
    REINKE_IMP_FRAC_LOWER_LIMIT = (
        78,
        (147,),
        (),
        "Reinke criterion impurity fraction lower limit",
    )
    BMAX_CS_UPPER_LIMIT = 79, (149,), ("bmaxcs_lim",), "Peak CS field upper limit"
    PDIVT_LOWER_LIMIT = 80, (153,), ("pdivtlim",), "Divertor power lower limit"
    DENSITY_PROFILE_CONSISTENCY = 81, (154,), (), "Ne(0) > ne(ped) constraint"
    STELLARATOR_COIL_CONSISTENCY = (
        82,
        (171,),
        ("toroidalgap",),
    )
    STELLARATOR_RADIAL_BUILD_CONSISTENCY = (
        83,
        (172,),
        (),
        "Radial build consistency for stellarators",
    )
    BETA_LOWER_LIMIT = 84, (173,), (), "Lower limit for beta"
    CP_LIFETIME_LOWER_LIMIT = (
        85,
        (),
        ("nflutfmax",),
        "Constraint for centrepost lifetime",
    )
    TURN_SIZE_UPPER_LIMIT = (
        86,
        (),
        ("t_turn_tf_max",),
        "Constraint for TF coil turn dimension",
    )
    CRYOPOWER_UPPER_LIMIT = 87, (), (), "Constraint for cryogenic power"
    TF_STRAIN_UPPER_LIMIT = (
        88,
        (),
        ("str_wp_max",),
        "Constraint for TF coil strain absolute value",
    )
    OH_CROCO_CU_AREA_CONSTRAINT = (
        89,
        (166,),
        ("copperaoh_m2_max",),
        "Constraint for CS coil quench protection",
    )
    CS_FATIGUE = (
        90,
        (167,),
        (
            "residual_sig_hoop",
            "n_cycle_min",
            "t_crack_radial",
            "t_crack_vertical",
            "t_structural_radial",
            "t_structural_vertical",
            "sf_vertical_crack",
            "sf_radial_crack",
            "sf_fast_fracture",
            "paris_coefficient",
            "paris_power_law",
            "walker_coefficient",
            "fracture_toughness",
        ),
        "CS fatigue constraints",
    )
    ECRH_IGNITABILITY = 91, (168,), (), "Checking if the design point is ECRH ignitable"


# The dreaded f-values
FV_CONSTRAINT_ITVAR_MAPPING = {
    5: 9,
    6: 8,
    8: 14,
    9: 26,
    12: 15,
    13: 21,
    15: 103,
    16: 25,
    17: 28,
    18: 27,
    19: 30,
    20: 33,
    21: 32,
    22: 34,
    23: 104,
    24: 36,
    25: 35,
    26: 38,
    27: 39,
    28: 45,
    # 30: 46,  # Keep this as an equality constraint by default
    31: 48,
    32: 49,
    33: 50,
    34: 51,
    35: 53,
    36: 54,
    37: 40,
    38: 62,
    39: 63,
    40: 64,
    41: 66,
    42: 67,
    44: 68,
    45: 71,
    46: 72,
    48: 79,
    50: 86,
    52: 89,
    53: 92,
    54: 95,
    55: 96,
    56: 97,
    59: 105,
    60: 106,
    61: 107,
    62: 110,
    63: 111,
    64: 112,
    65: 113,
    66: 115,
    67: 116,
    68: 117,
    69: 118,
    72: 123,
    73: 137,
    74: 141,
    75: 143,
    76: 144,
    77: 146,
    78: 146,
    81: 154,
    83: 160,  # OR 172?!
    84: 161,  # OR 173?!
    89: 166,
    90: 167,
    91: 168,
}

ITERATION_VAR_MAPPING = {
    "aspect": 1,
    "bt": 2,
    "rmajor": 3,
    "te": 4,
    "beta": 5,
    "dene": 6,
    "rnbeam": 7,
    "fbeta": 8,
    "fdene": 9,
    "hfact": 10,
    "pheat": 11,
    # NO LONGER USED "oacdp": 12,
    "tfcth": 13,
    "fwalld": 14,
    "fvs": 15,
    "ohcth": 16,
    "t_between_pulse": 17,
    "q": 18,
    "beam_energy": 19,
    "tcpav": 20,
    "ft_burn": 21,
    # 22 NOT USED
    "fcoolcp": 23,
    # 24 NOT USED
    "fpnetel": 25,
    "ffuspow": 26,
    "fhldiv": 27,
    "fradpwr": 28,
    "bore": 29,
    "fmva": 30,
    "gapomin": 31,
    "frminor": 32,
    "fportsz": 33,
    "fdivcol": 34,
    "fpeakb": 35,
    "fbeta_max": 36,
    "coheof": 37,
    "fjohc": 38,
    "fjohc0": 39,
    "fgamcd": 40,
    "fcohbop": 41,
    "gapoh": 42,
    # 43 NOT USED
    "fvsbrnni": 44,
    "fqval": 45,
    "fpinj": 46,
    "feffcd": 47,
    "fstrcase": 48,
    "fstrcond": 49,
    "fiooic": 50,
    "fvdump": 51,
    # 52 NOT USED
    "fjprot": 53,
    "ftmargtf": 54,
    # 55 NOT USED
    "tdmptf": 56,
    "thkcas": 57,
    "thwcndut": 58,
    "fcutfsu": 59,
    "cpttf": 60,
    "gapds": 61,
    "fdtmp": 62,
    "ftpeak": 63,
    "fauxmn": 64,
    "t_current_ramp_up": 65,
    "ft_current_ramp_up": 66,
    "ftcycl": 67,
    "fptemp": 68,
    "rcool": 69,
    "vcool": 70,
    "fq": 71,
    "fipir": 72,
    "scrapli": 73,
    "scraplo": 74,
    "tfootfi": 75,
    # 76, 77, 78 NOT USED
    "fbeta_poloidal": 79,
    # 80 NOT USED
    "edrive": 81,
    "drveff": 82,
    "tgain": 83,
    "chrad": 84,
    "pdrive": 85,
    "frrmax": 86,
    # 87, 88 NOT USED
    "ftbr": 89,
    "blbuith": 90,
    "blbuoth": 91,
    "fflutf": 92,
    "shldith": 93,
    "shldoth": 94,
    "fptfnuc": 95,
    "fvvhe": 96,
    "fpsepr": 97,
    "li6enrich": 98,
    # 99, 100, 101, 102 NOT USED
    "flhthresh": 103,
    "fcwr": 104,
    "fnbshinef": 105,
    "ftmargoh": 106,
    "favail": 107,
    "breeder_f": 108,
    "ralpne": 109,
    "ftaulimit": 110,
    "fniterpump": 111,
    "fzeffmax": 112,
    "fmaxvvstress": 113,  # OR IS IT fmaxvvstress ?! ftaucq
    "fw_channel_length": 114,
    "fpoloidalpower": 115,
    "fradwall": 116,
    "fpsepbqar": 117,
    "fpsep": 118,
    "tesep": 119,
    "ttarget": 120,
    "neratio": 121,
    "oh_steel_frac": 122,
    "foh_stress": 123,
    "qtargettotal": 124,
    "fimp(3)": 125,  # Beryllium
    "fimp(4)": 126,  # Carbon
    "fimp(5)": 127,  # Nitrogen
    "fimp(6)": 128,  # Oxygen
    "fimp(7)": 129,  # Neon
    "fimp(8)": 130,  # Silicon
    "fimp(9)": 131,  # Argon
    "fimp(10)": 132,  # Iron
    "fimp(11)": 133,  # Nickel
    "fimp(12)": 134,  # Krypton
    "fimp(13)": 135,  # Xenon
    "fimp(14)": 136,  # Tungsten
    "fplhsep": 137,
    "rebco_thickness": 138,
    "copper_thick": 139,
    "dr_tf_wp": 140,
    "fcqt": 141,
    "nesep": 142,
    "f_coppera_m2": 143,
    "fnesep": 144,
    "fgwped": 145,
    "fcpttf": 146,
    "freinke": 147,
    "fzactual": 148,
    "fbmaxcs": 149,
    # 150, 151 NOT USED
    "fgwsep": 152,
    "fpdivlim": 153,
    "fne0": 154,
    "pfusife": 155,
    "rrin": 156,
    "fvssu": 157,
    "croco_thick": 158,
    "ftoroidalgap": 159,
    "f_avspace": 160,
    "fbetatry_lower": 161,
    "r_cp_top": 162,
    "f_t_turn_tf": 163,
    "f_crypmw": 164,
    "fstr_wp": 165,
    "f_copperaoh_m2": 166,
    "fncycle": 167,
    "fecrh_ignition": 168,
    "te0_ecrh_achievable": 169,
    "beta_div": 170,
}


VAR_ITERATION_MAPPING = {v: k for k, v in ITERATION_VAR_MAPPING.items()}
