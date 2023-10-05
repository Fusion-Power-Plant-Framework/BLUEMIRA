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
PLASMOD mappings
"""

from bluemira.codes.utilities import Model, create_mapping


class ImpurityModel(Model):
    """
    Impurity Model selector

    0 - fixed concentration,
    1 - concentration fixed at pedestal top, then fixed density.

    Plasmod variable name: "i_impmodel"
    """

    FIXED = 0
    PED_FIXED = 1


class TransportModel(Model):
    """
    Transport Model Selector

    1 - simple gyrobohm scaling with imposed H factor,
    555 - H factor scaling from F. Palermo
    111 - another model based on gyro-Bohm transport
    2 - no reference in the source code

    Plasmod variable name: "i_modeltype"
    """

    GYROBOHM_1 = 1
    GYROBOHM_2 = 111
    UNKNOWN = 2
    H_FACTOR = 555


class EquilibriumModel(Model):
    """
    Equilibrium Model Selector

    1 - EMEQ solves equilibrium with given q95, with sawteeth.
    2 - EMEQ solves with given Ip, with sawteeth

    Plasmod variable name: "i_equiltype"
    """

    q95_sawtooth = 1
    Ip_sawtooth = 2


class SafetyProfileModel(Model):
    """
    Safety Factor Profile Model Selector

    0 - PLASMOD allows q < 1 in the core (fully relaxed q profile)
    1 - PLASMOD clamps q >= 1 in the core (sawteeth forced)

    Plasmod variable name: isawt

    NOTE: Running with 1 means that p' and FF' will not correspond well
    with jpar.
    """

    FULLY_RELAXED = 0
    SAWTEETH = 1


class PedestalModel(Model):
    """
    Pedestal Model Selector

    1 - fixed pedestal temperature (Teped_in),
    2 - Saarelma scaling

    Plasmod variable name: "i_pedestal"
    """

    FIX_TEMP = 1
    SAARELMA = 2


class SOLModel(Model):
    """
    SOL Model Selector:

    0 - fit based on Eich scaling
    1 - Mattia Siccinio's model

    Plasmod variable name: "isiccir"
    """

    EICH_FIT = 0
    SICCINIO = 1


class PLHModel(Model):
    """
    L-H transition power scaling model

    6 - Martin

    Plasmod variable name: "plh"
    """

    MARTIN = 6


class Profiles(Model):
    """
    Profile Selector:

    x       [-] normalized toroidal flux coordinate (Phi/Phi_b)
    ne      [10¹⁹/m3] electron density profile
    Te      [keV] Electron temperature profile
    Ti      [keV] Ion temperature profile
    psi     [Wb] Poloidal flux profile
    phi     [Wb] Toroidal flux profile
    press   [Pa] Plasma pressure profile
    pprime  [Pa/Wb] p' profile
    ffprime [(m*T) * (m*T) / Wb == T] FF' profile
    kprof   [-] Elongation profile
    dprof   [-] Triangularity profile
    shif    [m] Grad-Shafranov shift profile
    g2      [m²] < mod(grad V)²/r²> g2 metric coefficient's profile
    g3      [m⁻²] < 1/r²> g3 metric coefficient's profile
    volprof [m³] Volume profile
    vprime  [m³] Volume profile
    ipol    [m*T] Poloidal current profile
    qprof   [-] Safety factor profile
    jpar    [A/m²] Parallel current density profile
    jbs     [A/m²] Bootstrap parallel current density profile
    jcd     [A/m²] CD parallel current density profile
    nions   [10¹⁹/m³] ion density profile
    nfuel   [10¹⁹/m³] fuel density profile
    ndeut   [10¹⁹/m³] deuterium density profile
    ntrit   [10¹⁹/m³] tritium density profile
    nalf    [10¹⁹/m³] helium density profile

    Not yet enabled in plasmod:
     * qrad   radiation density profile
     * qneut  neutron fusion power density profile

    """

    x = "x"
    ne = "n_e"
    Te = "Te"
    Ti = "Ti"
    psi = "psi"
    phi = "phi"
    press = "pressure"
    pprime = "pprime"
    ffprime = "ffprime"
    kprof = "kappa"
    dprof = "delta"
    shif = "GS"
    g2 = "g2"
    g3 = "g3"
    volprof = "V"
    vprime = "Vprime"
    ipol = "i_pol"
    qprof = "q"
    jpar = "jpar"
    jbs = "jbs"
    jcd = "jcd"
    nions = "n_ion"
    nfuel = "n_fuel"
    ndeut = "n_D"
    ntrit = "n_T"
    nalf = "n_He"
    # qrad = "q_rad"
    # qneut = "q_neut"


PLASMOD_INPUTS = {
    ############################
    # Numeric properties
    #############################
    # [-] max iteration error between transport/equilibrium iterations
    # ###### "BM_INP": ("tol", "dimensionless"),
    # [-] min time step between iterations
    # ###### "BM_INP": ("dtmin", "dimensionless"),
    # [-] max time step between iterations
    # ###### "BM_INP": ("dtmax", "dimensionless"),
    # [-] exponent of jipperdo2
    # ###### "BM_INP": ("dtmaxmin", "dimensionless"),
    # [m²/s] stabilizing diff for TGLF in
    # ###### "BM_INP": ("dtmaxmax", "m^2/s"),
    # [-] tolerance above which TGLF should be always called
    # ###### "BM_INP": ("dtminmax", "dimensionless"),
    # [-] !time step
    # ###### "BM_INP": ("dt", "s"),
    # [-] !decrease of dt
    # ###### "BM_INP": ("dtinc", "s"),
    # [-] !increase of dt
    # ###### "BM_INP": ("Ainc", "s"),
    # [-] max number of iteration
    # ###### "BM_INP": ("test", "dimensionless"),
    # [-] ! multiplier of etolm that should not be overcome
    # ###### "BM_INP": ("tolmin", "dimensionless"),
    # [-] Newton differential
    # ###### "BM_INP": ("dgy", "dimensionless"),
    # [-] !exponent of jipperdo
    # ###### "BM_INP": ("eopt", "dimensionless"),
    # [-] first radial grid point
    # ###### "BM_INP": ("capA", "dimensionless"),
    # [-] diagnostics for ASTRA (0 or 1)
    # ###### "BM_INP": ("i_diagz", "dimensionless"),
    # [-] number of interpolated grid points
    # ###### "BM_INP": ("nx", "dimensionless"),
    # [-] number of reduced grid points
    # ###### "BM_INP": ("nxt", "dimensionless"),
    # [-] number of unknowns in the transport solver
    # (ne, Te, Ti) leave this equal to 3!
    # ###### "BM_INP": ("nchannels", "dimensionless"),
    # [-] number of tglf points, below positions
    # ###### "BM_INP": ("ntglf", "dimensionless"),
    # [-] tglf points, position 1
    # ###### "BM_INP": ("xtglf_1", "dimensionless") ,
    # [-] tglf points, position 2
    # ###### "BM_INP": ("xtglf_2", "dimensionless"),
    # [-] tglf points, position 3
    # ###### "BM_INP": ("xtglf_3", "dimensionless"),
    # [-] tglf points, position 4,
    # ###### "BM_INP": ("xtglf_4", "dimensionless"),
    # [-] tglf points, position 5
    # ###### "BM_INP": ("xtglf_5", "dimensionless"),
    # [-] tglf points, position 6
    # ###### "BM_INP": ("xtglf_6", "dimensionless"),
    # [-] tglf points, position 7
    # ###### "BM_INP": ("xtglf_7", "dimensionless"),
    # [-] tglf points, position 8
    # ###### "BM_INP": ("xtglf_8", "dimensionless"),
    # [-] tglf points, position 9
    # ###### "BM_INP": ("xtglf_9", "dimensionless"),
    # [-] tglf points, position 10
    # ###### "BM_INP": ("xtglf_10", "dimensionless"),
    # [-] tglf points, position 11
    # ###### "BM_INP": ("xtglf_11", "dimensionless"),
    # ###########################
    # Geometry properties
    # ###########################
    # [-] plasma aspect ratio
    "A": ("A", "dimensionless"),
    # [T] Toroidal field at plasma center
    "B_0": ("Bt", "T"),
    # [-] plasma triangularity at 95 % flux
    "delta_95": ("d95", "dimensionless"),
    # [-] plasma elongation at 95 % flux
    "kappa_95": ("k95", "dimensionless"),
    # [m] plasma major radius
    "R_0": ("R", "m"),
    # [m3] constrained plasma volume (set negative value to disable volume constraining)
    "V_p": ("volume_in", "m^3"),
    # ###########################
    # Composition properties
    # ############################
    # [-] fuel mix D/T
    # ###### "BM_INP": ("fuelmix", "dimensionless"),
    # [-] He3 as fuel concentration
    # ###### "BM_INP": ("fuelhe3", # TODO
    # [-] tauparticle / tauE for D
    # ###### "BM_INP": ("globtau_d", "dimensionless"),
    # [-] tauparticle / tauE for T
    # ###### "BM_INP": ("globtau_t", "dimensionless"),
    # [-] tauparticle / tauE for He
    # ###### "BM_INP": ("globtau_he", "dimensionless"),
    # [-] tauparticle / tauE for Xe
    # ###### "BM_INP": ("globtau_xe", "dimensionless"),
    # [-] tauparticle / tauE for Ar
    # ###### "BM_INP": ("globtau_ar", "dimensionless"),
    # [-] min P_sep/P_LH. if Psep/PLH < Psep/PLH_max -> use heating
    # ###### "BM_INP": ("psepplh_inf", "dimensionless"),
    # [-] max P_sep/P_LH. if Psep/PLH > Psep/PLH_max -> use Xe
    # ###### "BM_INP": ("psepplh_sup", "dimensionless"),
    # [-] position after which radiation is "edge"
    # ###### "BM_INP": ("pradpos", # TODO
    # [-] radiation fraction used for core transport
    # ###### "BM_INP": ("pradfrac", "dimensionless"),
    # [MW*T/m] Divertor challenge criterion Psep * Bt / (q95 * A * R_0)
    # if PsepBt_qAR > PsepBt_qAR_max seed Xenon
    "PsepB_qAR_max": ("psepb_q95AR_sup", "MW.T/m"),
    # [MW/m] Divertor challenging criterion Psep / R0
    # if Psep/R0 > Psep_R0_max seed Xenon
    # ###### "BM_INP": ("psep_r_sup" "MW/m"),
    # [-] ratio of Pline(Xe)/(Psep0 - Psepcrit), or -1 to ignore
    # Psep0 = Palpha + Paux - Pline(Ar) - Pbrehm - Psync
    # ###### "BM_INP": ("fcoreraditv", "dimensionless"),
    # [MW/m2] max divertor heat flux -->
    # if qdivt > qdivt_sup -> seed argon
    # ###### "BM_INP": ("qdivt_sup" "MW/m^2"),
    # [-] compression factor between sol and div
    # e.g. 10 means there is
    # 10 more Argon concentration in the divertor than in the core
    # ###### "BM_INP": ("c_car", "dimensionless"),
    # ###########################
    # Pedestal properties
    # ############################
    # [-] normalized coordinate of pedestal density
    # ###### "BM_INP": ("rho_n", "dimensionless"),
    # [-] normalized coordinate of pedestal temperature
    # ###### "BM_INP": ("rho_T", "dimensionless"),
    # [keV] electrons/ions temperature at separatrix
    # ###### "BM_INP": ("tesep", "keV"),
    # [-] scaling factor for p_ped scaling formula
    # ###### "BM_INP": ("pedscal", "dimensionless"),
    # ###########################
    # General inputs: control, confinement, B.C., etc
    # ############################
    # [-] Greenwald density fraction at pedestal
    # ###### "BM_INP": ("f_gw", "dimensionless"),
    # [-] Greenwald density fraction at separatrix
    # ###### "BM_INP": ("f_gws", "dimensionless"),
    # [-] fraction of NBI power to ions
    # ###### "BM_INP": ("fpion", "dimensionless"),
    # [m*MA/MW] Normalized CD efficiency   # tentative g_cd_nb but normalise wrt to what?
    # ###### "BM_INP": ("nbcdeff", "m.MA/MW"),
    # [m*MA/MW] Normalized EC efficiency   # tentative g_cd_nb but normalise wrt to what?
    # ###### "BM_INP": ("eccdeff", "m.MA/MW")
    # [-]  normalized mean location of fixed NBI heating
    # ###### "BM_INP": ("x_control_nbi", "dimensionless"),
    # [-]  normalized mean location of fixed EC heating
    # ###### "BM_INP": ("x_control_ech", "dimensionless"),
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x for fixed NBI heating
    # ###### "BM_INP": ("dx_control_nbi", "dimensionless"),
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x for fixed EC heating
    # ###### "BM_INP": ("dx_control_ech", "dimensionless"),
    # [-]  normalized mean location of NBI power for
    # controlling loop voltage or f_ni
    # ###### "BM_INP": ("x_cd_nbi", "dimensionless"),
    # [-]  normalized mean location of NBI power for
    # controlling loop voltage or f_ni
    # ###### "BM_INP": ("x_cd_ech", "dimensionless"),
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x for NBI heating (CD) to control Vloop or f_ni
    # ###### "BM_INP": ("dx_cd_nbi", "dimensionless"),
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x for EC heating to control Vloop or f_ni
    # ###### "BM_INP": ("dx_cd_ech", "dimensionless"),
    # [-]  normalized mean location of NBI heating for
    # controlling fusion power (Pfus = Pfus_req)
    # ###### "BM_INP": ("x_fus_nbi", "dimensionless"),
    # [-]  normalized mean location of EC heating for
    # controlling fusion power (Pfus = Pfus_req)
    # ###### "BM_INP": ("x_fus_ech", "dimensionless"),
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x, for NBI heating to control fusion power
    # ###### "BM_INP": ("dx_fus_nbi", "dimensionless"),
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x, for EC heating to control fusion power
    # ###### "BM_INP": ("dx_fus_ech", "dimensionless"),
    # [-]  normalized mean location of aux. NBI heating for
    # controlling H-mode operation (P_sep/P_LH > P_sep_P_LH_min)
    # ###### "BM_INP": ("x_heat_nbi", "dimensionless"),
    # [-]  normalized mean location of aux. EC heating for
    # controlling H-mode operation (P_sep/P_LH > P_sep_P_LH_min)
    # ###### "BM_INP": ("x_heat_ech", "dimensionless"),
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x, for NBI heating to control H-mode
    # ###### "BM_INP": ("dx_heat_nbi", "dimensionless"),
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x, for EC heating to control H-mode
    # ###### "BM_INP": ("dx_heat_ech", "dimensionless"),
    # [keV] NBI energy
    "e_nbi": ("nbi_energy", "keV"),
    # [MW] required fusion power.
    # 0. - ignored
    # > 0 - Auxiliary heating is calculated to match Pfus_req
    # ###### "BM_INP": ("pfus_req", "MW")
    # [-] required fraction of non inductive current, if 0, dont use CD
    "f_ni": ("f_ni", "dimensionless"),
    # [MW] max allowed power for control (fusion power, H-mode)
    # ###### "BM_INP": ("pheat_max", "MW")
    # [MW] fixed auxiliary heating power required for control
    "q_control": ("q_control", "MW"),
    # [MW] total auxiliary power  (0.) DO NOT CHANGE
    # ###### "BM_INP": ("q_heat", "MW"),
    # [MW] total auxiliary current drive power (0.) DO NOT CHANGE
    # ###### "BM_INP": ("q_cd", "MW"),
    # [MW] total fusion power (0.) DO NOT CHANGE
    # ###### "BM_INP": ("q_fus", "MW"),
    # [MW] ECH power (not used)
    # ###### "BM_INP": ("pech", "MW"),
    # [MW] NBI power (not used)
    # ###### "BM_INP": ("pnbi", "MW"),
    # [-] ratio of PCD-Pothers over Pmax - Pothers
    # ###### "BM_INP": ("fcdp": -1.0, "dimensionless"),
    # [-] maximum Paux/R allowed
    # ###### "BM_INP": ("maxpauxor", "dimensionless"),
    # [-] scaling factor for newton scheme on NBI (100.)
    # ###### "BM_INP": ("qnbi_psepfac", "dimensionless"),
    # [-] scale factor for newton scheme on Xe (1.e-3)
    # ###### "BM_INP": ("cxe_psepfac", "dimensionless"),
    # [-] scale factor for newton scheme on Ar (1.e-4)
    # ###### "BM_INP": ("car_qdivt", "dimensionless"),
    # [MW / m²] Pcontrol / S_lateral(0.)
    # ###### "BM_INP": ("contrpovs", "MW/m^2"),
    # [MW / m²] Pcontrol / R(0.)
    # ###### "BM_INP": ("contrpovr", "MW/m^2"),
}


PLASMOD_OUTPUTS = {
    # ###########################################
    # Geometry properties (geom type)
    # ###########################################
    # [m] plasma perimeter
    # ##### "BM_OUT": ("perim", "m"),
    # ###########################################
    # MHD equilibrium properties (MHD type)
    # ###########################################
    # [T] average poloidal field
    # ##### "BM_OUT": ("bpolavg", "T"),
    # [-] toroidal beta
    # ##### "BM_OUT": ("betator", "dimensionless"),
    # [-] poloidal beta
    "beta_p": ("betapol", "dimensionless"),
    # [-] normalized beta
    "beta_N": ("betan", "dimensionless"),
    # [-] Greenwald density at pedestal top
    # ##### "BM_OUT": (f_gwpedtop", "dimensionless),
    # [-] plasma bootstrap current fraction
    "f_bs": ("fbs", "dimensionless"),
    # [-] plasma current drive fraction
    # ##### "BM_OUT": ("fcd", "dimensionless"),
    # [-] Edge safety factor
    # ##### "BM_OUT": ("q_sep", "dimensionless"),
    # [-] cylindrical safety factor
    # ##### "BM_OUT": ("qstar", "dimensionless"),
    # [-] normalised plasma internal inductance
    "l_i": ("rli", "dimensionless"),
    # [m²] plasma poloidal cross section area
    # ##### "BM_OUT": ("Sp", "m^2"),
    # [m²] plasma toroidal surface
    # ##### "BM_OUT": ("torsurf", "m^2"),
    # [m³] plasma volume
    # ##### "BM_OUT": ("Vp", "m^3"),
    # ###########################################
    # Confinement properties (loss type)
    # ###########################################
    # [-] radiation-corrected H-factor
    "H_star": ("Hcorr", "dimensionless"),
    # [s] global energy confinement time
    "tau_e": ("taueff", "s"),
    # [s] electrons energy confinement time
    # ##### "BM_OUT": ("tauee", "s"),
    # [s] ions energy confinement time
    # ##### "BM_OUT": ("tauei", "s"),
    # [J] plasma thermal energy
    # ##### "BM_OUT": ("Wth", "J"),
    # [Ohm] plasma resistance
    "res_plasma": ("rplas", "ohm"),
    # ###########################################
    # Power properties (loss type)
    # ###########################################
    # [W] DD fusion power
    "P_fus_DD": ("Pfusdd", "W"),
    # [W] DT fusion power
    "P_fus_DT": ("Pfusdt", "W"),
    # [W] Fusion power
    "P_fus": ("Pfus", "W"),
    # [W] neutron fusion power
    # ##### "BM_OUT": ("Pneut", "W"),
    # [W] total auxiliary heating power
    # ##### "BM_OUT": ("Paux", "W"),
    # [W] auxiliary heating power to electrons
    # ##### "BM_OUT": ("Peaux", "W"),
    # [W] auxiliary heating power to ions
    # ##### "BM_OUT": ("Piaux", "W"),
    # [W] alpha power
    # ##### "BM_OUT": ("Palpha", "W"),
    # [W] total radiation power
    "P_rad": ("Prad", "W"),
    # [W] core radiation power
    # ##### "BM_OUT": ("Pradcore", "W"),
    # [W] core radiation power
    # ##### "BM_OUT": ("Pradedge", "W"),
    # [W] total power across plasma separatrix
    "P_sep": ("Psep", "W"),
    # [W] Synchrotron radiation power
    "P_sync": ("Psync", "W"),
    # [W] Bremsstrahlung radiation power
    "P_brehms": ("Pbrehms", "W"),
    # [W] Line radiation power
    "P_line": ("Pline", "W"),
    # [W] LH transition power
    "P_LH": ("PLH", "W"),
    # [W] Ohimic heating power
    "P_ohm": ("Pohm", "W"),
    # [W] Auxiliary heating power added to control f_ni or v_loop
    # ##### "BM_OUT": ("qcd", "W"),
    # [W] Auxiliary heating power added to operate in H-mode
    # ##### "BM_OUT": ("qheat", "W"),
    # [W] Auxiliary heating power added to control Pfus
    # ##### "BM_OUT": ("qfus", "W"),
    # [W/m2] divertor heat flux
    # ##### "BM_OUT": ("qdivt", "W/m^2"),
    # [MW/m] Divertor challenging criterion Psep/R0
    # ##### "BM_OUT": ("psep_r", "MW/m"),
    # [MW * T/ m] Divertor challenging criterion Psep * Bt /(q95 * a)
    # ##### "BM_OUT": ("psepb_q95AR", "MW.T/m"),
    # ###########################
    # Composition properties (type comp)
    # ############################
    # [-] plasma effective charge
    "Z_eff": ("Zeff", "amu"),  # TODO check dimensionless?
    # [V] target loop voltage (if lower than -1e-3, ignored)-> plasma loop voltage
    "v_burn": ("v_loop", "V"),
    # ###########################
    # Pedestal properties (type ped)
    # ############################
    # [1E19/m3] electron/ion density at pedestal height
    # ##### "BM_OUT": ("nped", # TODO
    # [1E19/m3] electron/ion density at separatrix
    # ##### "BM_OUT": ("nsep", # TODO
    # ###########################
    # Average properties for profiles (type radp)
    # ############################
    # [1E19/m3] volume-averaged ion density
    # ##### "BM_OUT": ("av_ni", # TODO
    # [1E19/m3] volume-averaged fuel density
    # ##### "BM_OUT": ("av_nd", # TODO
    # [1E19/m3] volume-averaged plasma impurities density
    # ##### "BM_OUT": ("av_nz", # TODO
    # [1E19/m3] volume-averaged helium density
    # ##### "BM_OUT": ("av_nhe", # TODO
    # [keV] volume-averaged ions temperature
    # ##### "BM_OUT": ("av_Ti", "keV"),
    # [keV] volume-averaged electrons temperature
    # ##### "BM_OUT": ("av_Te", "keV"),
    # [keV] density-averaged electrons temperature
    # ##### "BM_OUT": ("av_Ten", "keV"),
}

PLASMOD_INOUTS = {
    # ###########################################
    # Geometry properties (geome type)
    # ###########################################
    # [-] plasma edge triangularity (used only for first iteration,
    # then iterated to constrain delta95)
    "delta": ("d", "dimensionless"),
    # [-] plasma edge elongation (used only for first iteration,
    # then iterated to constrain kappa95)
    "kappa": ("k", "dimensionless"),
    # [-] plasma minor radius (just initial guess)
    # ##### "BM_IO": ("amin", "m"),
    # ###########################################
    # MHD equilibrium properties (mhd type)
    # ###########################################
    # [MA] plasma current
    # (used if i_equiltype == 2. Otherwise Ip is calculated
    # and q95 is used as input)
    "I_p": ("Ip", "MA"),
    # [-] safety factor at 95% flux surface
    # (used if i_equiltype == 1. Otherwise q95 is calculated
    # and Ip is used as input)
    "q_95": ("q95", "dimensionless"),
    # [-] plasma current inductive fraction
    # ##### "BM_IO": ("f_ni", "dimensionless"),
    # ###########################
    # Composition properties
    # ############################
    # [-] Hydrogen concentration
    # ##### "BM_IO": ("cprotium", # TODO
    # [-] helium concentration
    # (used if  globtau_he = 0)
    # total Helium concentration (he4 + He3)
    # ##### "BM_IO": ("che", # TODO
    # [-] He3 concentration
    # ##### "BM_IO": ("che3", # TODO
    # [-] Argon concentration
    # ###### "BM_IO": ("car", # TODO
    # [-] Xenon concentration
    # #### "BM_IO": ("cxe", # TODO
    # [-] Tungsten concentration
    # ##### "BM_IO": ("cwol": 0.0,
    # ###########################
    # Pedestal properties
    # ############################
    # [keV] electrons/ions temperature at pedestal (ignored if i_pedestal = 2)
    "T_e_ped": ("teped", "keV"),
    # ###########################
    # Confinement properties (type loss)
    # ############################
    # [-] H-factor:if i_modeltype > 1 H factor calculated
    # ##### "BM_IO": ("Hfact", "dimensionless"),
}

mappings = create_mapping(PLASMOD_INPUTS, PLASMOD_OUTPUTS, PLASMOD_INOUTS)
