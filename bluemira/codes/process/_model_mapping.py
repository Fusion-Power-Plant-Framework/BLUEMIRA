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

"""
PROCESS model mappings
"""
from bluemira.codes.utilities import Model


class PROCESSModel(Model):
    """
    Baseclass for PROCESS models
    """

    @property
    def switch_name(self) -> str:
        return ""


class PlasmaGeometryModel(Model):
    """
    Switch for plasma geometry

    PROCESS variable name: "ishape"
    """

    HENDER_K_D_100 = 0
    GALAMBOS_K_D_95 = 1
    ZOHM_ITER = 2
    ZOHM_ITER_D_95 = 3
    HENDER_K_D_95 = 4
    MAST_95 = 5
    MAST_100 = 6
    FIESTA_95 = 7
    FIESTA_100 = 8


class PlasmaNullConfigurationModel(Model):
    """
    Switch for single-null / double-null

    PROCESS variable name: "snull"
    """

    DOUBLE_NULL = 0
    SINGLE_NULL = 1


class PlasmaProfileModel(Model):
    """
    Switch for plasma profile model

    PROCESS variable name: "ipedestal"
    """

    NO_PEDESTAL = 0
    PEDESTAL_GW = 1
    PLASMOD_GW = 2
    PLASMOD = 3


class BetaLimitModel(Model):
    """
    Switch for the plasma beta limit model

    PROCESS variable name: "iculbl"
    """

    TOTAL = 0  # Including fast ion contribution
    THERMAL = 1
    THERMAL_NBI = 2
    TOTAL_TF = 3  # Calculated using only the toroidal field


class BetaGScalingModel(Model):
    """
    Switch for the beta g coefficient dnbeta model

    PROCESS variable name: "gtscale"

    NOTE: Over-ridden if iprofile = 1
    """

    INPUT = 0  # dnbeta is an input
    CONVENTIONAL = 1
    MENARD_ST = 2


class AlphaPressureModel(Model):
    """
    Switch for the pressure contribution from fast alphas

    PROCESS variable name: "ifalphap"
    """

    HENDER = 0
    LUX = 1


class DensityLimitModel(Model):
    """
    Switch for the density limit model

    PROCESS variable name: "idensl"
    """

    ASDEX = 1
    BORRASS_ITER_I = 2
    BORRASS_ITER_II = 3
    JET_RADIATION = 4
    JET_SIMPLE = 5
    HUGILL_MURAKAMI = 6
    GREENWALD = 7


class PlasmaCurrentScalingLaw(Model):
    """
    Switch for plasma current scaling law

    PROCESS variable name: "icurr"
    """

    PENG = 1
    PENG_DN = 2
    ITER_SIMPLE = 3
    ITER_REVISED = 4  # Recommended for iprofile = 1
    TODD_I = 5
    TODD_II = 6
    CONNOR_HASTIE = 7
    SAUTER = 8
    FIESTA = 9


class ConfinementTimeScalingLaw(Model):
    """
    Switch for the energy confinement time scaling law

    PROCESS variable name: "isc"
    """

    NEO_ALCATOR_OHMIC = 1
    MIRNOV_H_MODE = 2
    MEREZHKIN_MUHKOVATOV_L_MODE = 3
    SHIMOMURA_H_MODE = 4
    KAYE_GOLDSTON_L_MODE = 5
    ITER_89_P_L_MODE = 6
    ITER_89_O_L_MODE = 7
    REBUT_LALLIA_L_MODE = 8
    GOLDSTON_L_MODE = 9
    T10_L_MODE = 10
    JAERI_88_L_MODE = 11
    KAYE_BIG_COMPLEX_L_MODE = 12
    ITER_H90_P_H_MODE = 13
    ITER_MIX = 14  # Minimum of 6 and 7
    RIEDEL_L_MODE = 15
    CHRISTIANSEN_L_MODE = 16
    LACKNER_GOTTARDI_L_MODE = 17
    NEO_KAYE_L_MODE = 18
    RIEDEL_H_MODE = 19
    ITER_H90_P_H_MODE_AMENDED = 20
    LHD_STELLARATOR = 21
    GRYO_RED_BOHM_STELLARATOR = 22
    LACKNER_GOTTARDI_STELLARATOR = 23
    ITER_93H_H_MODE = 24
    TITAN_RFP = 25
    ITER_H97_P_NO_ELM_H_MODE = 26
    ITER_H97_P_ELMY_H_MODE = 27
    ITER_96P_L_MODE = 28
    VALOVIC_ELMY_H_MODE = 29
    KAYE_PPPL98_L_MODE = 30
    ITERH_PB98P_H_MODE = 31
    IPB98_Y_H_MODE = 32
    IPB98_Y1_H_MODE = 33
    IPB98_Y2_H_MODE = 34
    IPB98_Y3_H_MODE = 35
    IPB98_Y4_H_MODE = 36
    ISS95_STELLARATOR = 37
    ISS04_STELLARATOR = 38
    DS03_H_MODE = 39
    MURARI_H_MODE = 40
    PETTY_H_MODE = 41
    LANG_H_MODE = 42
    HUBBARD_NOM_I_MODE = 43
    HUBBARD_LOW_I_MODE = 44
    HUBBARD_HI_I_MODE = 45
    NSTX_H_MODE = 46
    NSTX_PETTY_H_MODE = 47
    NSTX_GB_H_MODE = 48
    INPUT = 49  # tauee_in


class BootstrapCurrentScalingLaw(Model):
    """
    Switch for the model to calculate bootstrap fraction

    PROCESS variable name: "ibss"
    """

    ITER = 1
    GENERAL = 2
    NUMERICAL = 3
    SAUTER = 4


class LHThreshholdScalingLaw(Model):
    """
    Switch for the model to calculate the L-H power threshhold

    PROCESS variable name: "ilhthresh"
    """

    ITER_1996_NOM = 1
    ITER_1996_LOW = 2
    ITER_1996_HI = 3
    ITER_1997 = 4
    ITER_1997_K = 5
    MARTIN_NOM = 6
    MARTIN_HI = 7
    MARTIN_LOW = 8
    SNIPES_NOM = 9
    SNIPES_HI = 10
    SNIPES_LOW = 11
    SNIPES_CLOSED_DIVERTOR_NOM = 12
    SNIPES_CLOSED_DIVERTOR_HI = 13
    SNIPES_CLOSED_DIVERTOR_LOW = 14
    HUBBARD_LI_NOM = 15
    HUBBARD_LI_HI = 16
    HUBBARD_LI_LOW = 17
    HUBBARD_2017_LI = 18
    MARTIN_ACORRECT_NOM = 19
    MARTIN_ACORRECT_HI = 20
    MARTIN_ACORRECT_LOW = 21


class PlasmaWallGapModel(Model):
    """
    Switch to select plasma-wall gap model

    PROCESS variable name: "iscrp"
    """

    TEN_PERCENT = 0
    INPUT = 1  # scrapli and scraplo are inputs


class OperationModel(Model):
    """
    Switch to set the operation mode

    PROCESS variable name: "lpulse"
    """

    STEADY_STATE = 0
    PULSED = 1


class ThermalStorageModel(Model):
    """
    Switch to et the power cycle thermal storage model

    PROCESS variable name: "istore"
    """

    INHERENT_STEAM = 1
    BOILER = 2
    STEEL = 3  # Obsolete


class BlanketModel(Model):
    """
    Switch to select the blanket model

    PROCESS variable name: "blktmodel"
    """

    CCFE_HCPB = 1
    KIT_HCPB = 2
    CCFE_HCPB_TBR = 3


class TFCSTopologyModel(Model):
    """
    Switch to select the TF-CS topology
    """

    ITER = 0
    INSANITY = 1


class TFCoilConductorTechnology(Model):
    """
    Switch for TF coil conductor model:

    0 - copper
    1 - superconductor
    2 - Cryogenic aluminium

    PROCESS variable name: "i_tf_sup"
    """

    COPPER = 0
    SC = 1
    CRYO_AL = 2


class TFSuperconductorModel(Model):
    """
    Switch for the TF superconductor model

    PROCESS variable name: "i_tf_sc_mat"
    """

    NB3SN_ITER_STD = 1
    BI_2212 = 2
    NBTI = 3
    NB3SN_ITER_INPUT = 4  # User-defined critical parameters
    NB3SN_WST = 5
    REBCO_CROCO = 6
    NBTI_DGL = 7
    REBCO_DGL = 8
    REBCO_ZHAI = 9


class TFCasingGeometryModel(Model):
    """
    Switch for the TF casing geometry model

    PROCESS variable name: "i_tf_case_geom"
    """

    CURVED = 0
    FLAT = 1


class TFWindingPackGeometryModel(Model):
    """
    Switch for the TF winding pack geometry model

    PROCESS variable name: "i_tf_wp_geom"
    """

    RECTANGULAR = 0
    DOUBLE_RECTANGULAR = 1
    TRAPEZOIDAL = 2


class TFWindingPackTurnModel(Model):
    """
    Switch for the TF winding pack turn model

    PROCESS variable name: "i_tf_turns_integer"
    """

    CURRENT_PER_TURN = 0  # set cpttf or t_cable_tf or t_turn_tf
    INTEGER_TURN = 1  # set n_layer and n_pancake


class TFCoilShapeModel(Model):
    """
    Switch for the TF coil shape model

    PROCESS variable name: "i_tf_shape"
    """

    PRINCETON = 1
    PICTURE_FRAME = 2


class ResistiveCentrepostModel(Model):
    """
    Swtich for the resistive centrepost model

    PROCESS variable name: "i_r_cp_top"
    """

    CALCULATED = 0
    INPUT = 1
    MID_TOP_RATIO = 2


class TFCoilJointsModel(Model):
    """
    Switch for the TF coil joints

    PROCESS variable name: "i_cp_joints"
    """

    NO_JOINTS = 0
    SLIDING_JOINTS = 1


class TFStressModel(Model):
    """
    Switch for the TF inboard midplane stress model

    PROCESS variable name: "i_tf_stress_model"
    """

    GEN_PLANE_STRAIN = 0
    PLANE_STRESS = 1
    GEN_PLANE_STRAIN_NEW = 2


class TFCoilSupportModel(Model):
    """
    Switch for the TF inboard coil support model

    PROCESS variable name: "i_tf_bucking"
    """

    NO_SUPPORT = 0
    BUCKED = 1
    BUCKED_WEDGED = 2


class PFConductorModel(Model):
    """
    Switch for the PF conductor technology model

    PROCESS variable name: "ipfres"
    """

    SUPERCONDUCTING = 0
    RESISTIVE = 1


class PFSuperconductorModel(Model):
    """
    Switch for the PF superconductor model

    PROCESS variable name: "isumatpf"
    """

    NB3SN_ITER_STD = 1
    BI_2212 = 2
    NBTI = 3
    NB3SN_ITER_INPUT = 4  # User-defined critical parameters
    NB3SN_WST = 5
    REBCO_CROCO = 6
    NBTI_DGL = 7
    REBCO_DGL = 8
    REBCO_ZHAI = 9


class CSPrecompressionModel(Model):
    """
    Switch to control the existence of pre-compression tie plates in the CS

    PROCESS variable name: "iprecomp"
    """

    ABSENT = 0
    PRESENT = 1


class DivertorHeatFluxModel(Model):
    """
    Switch for the divertor heat flux model

    PROCESS variable name: "i_hldiv"
    """

    # TODO: What about Kallenbach?
    INPUT = 0
    CHAMBER = 1
    WADE = 2


class DivertorThermalHeatUse(Model):
    """
    Switch to control if the divertor thermal power is used in the
    power cycle

    PROCESS variable name: "iprimdiv"
    """

    LOW_GRADE_HEAT = 0
    HIGH_GRADE_HEAT = 1


class PrimaryPumpingModel(Model):
    """
    Switch for the calculation method of the pumping power
    required for the primary coolant

    PROCESS variable name: "primary_pumping"
    """

    INPUT = 0
    FRACTION = 1
    PRESSURE_DROP = 3


class SecondaryCycleModel(Model):
    """
    Switch for the calculation of thermal to electric conversion efficiency

    PROCESS variable name: "secondary_cycle"
    """

    FIXED = 0
    FIXED_W_DIVERTOR = 1
    INPUT = 2
    RANKINE = 3
    BRAYTON = 4


class CurrentDriveEfficiencyModel(Model):
    """
    Switch for current drive efficiency model:

    1 - Fenstermacher Lower Hybrid
    2 - Ion Cyclotron current drive
    3 - Fenstermacher ECH
    4 - Ehst Lower Hybrid
    5 - ITER Neutral Beam
    6 - new Culham Lower Hybrid model
    7 - new Culham ECCD model
    8 - new Culham Neutral Beam model
    10 - ECRH user input gamma
    11 - ECRH "HARE" model (E. Poli, Physics of Plasmas 2019)
    12 - EBW user scaling input. Scaling (S. Freethy)

    PROCESS variable name: "iefrf"
    """

    FENSTER_LH = 1
    ICYCCD = 2
    FENSTER_ECH = 3
    EHST_LH = 4
    ITER_NB = 5
    CUL_LH = 6
    CUL_ECCD = 7
    CUL_NB = 8
    ECRH_UI_GAM = 10
    ECRH_HARE = 11
    EBW_UI = 12


class PlasmaIgnitionModel(Model):
    """
    Switch to control whether or not the plasma is ignited

    PROCESS variable name: "ignite"
    """

    NOT_IGNITED = 0
    IGNITED = 1


class VacuumPumpingModel(Model):
    """
    Switch to control the vacuum pumping technology model

    PROCESS variable name: ntype
    """

    TURBO_PUMP = 0
    CRYO_PUMP = 1


class AvailabilityModel(Model):
    """
    Switch to control the availability model

    PROCESS variable name: "iavail"
    """

    INPUT = 0
    TAYLOR_WARD = 1
    MORRIS = 2


class CostModel(Model):
    """
    Switch to control the cost model used

    PROCESS variable name: "cost_model"
    """

    TETRA_1990 = 0
    KOVARI_2015 = 1
