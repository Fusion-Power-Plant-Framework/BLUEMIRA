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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import LinearTriInterpolator, Triangulation
from scipy.interpolate import interp1d

from bluemira.codes.plasmod.equilibrium_2d_coupling import (
    calc_curr_dens_profiles,
    calc_metric_coefficients,
)
from bluemira.equilibria.flux_surfaces import ClosedFluxSurface
from bluemira.equilibria.shapes import flux_surface_zakharov
from bluemira.geometry._private_tools import offset
from bluemira.optimisation._tools import approx_derivative


class PLASMODVerificationRawData:
    # fmt:off
    x = np.array([0.00000000000E+0000, 0.24915370422E-0001, 0.49830740845E-0001, 0.74746111267E-0001,    0.99661481689E-0001,      0.12457685211E+0000,      0.14949222253E+0000,      0.17440759296E+0000,      0.19932296338E+0000,      0.22423833380E+0000,      0.24915370422E+0000,      0.27406907465E+0000,      0.29898444507E+0000,      0.32389981549E+0000,      0.34881518591E+0000,      0.37373055634E+0000,      0.39864592676E+0000,      0.42356129718E+0000,      0.44847666760E+0000,      0.47339203803E+0000,      0.49830740845E+0000,      0.52322277887E+0000,      0.54813814929E+0000,      0.57305351971E+0000,      0.59796889014E+0000,      0.62288426056E+0000,      0.64779963098E+0000,      0.67271500140E+0000,      0.69763037183E+0000,      0.72254574225E+0000,     0.74746111267E+0000,      0.77237648309E+0000,      0.79729185352E+0000,     0.82220722394E+0000,     0.84712259436E+0000,      0.87203796478E+0000,     0.89695333521E+0000,     0.92186870563E+0000,      0.94678407605E+0000,      0.97169944647E+0000,      0.99661481689E+0000])  # noqa: E241
    pprime = np.array([-0.47221539896E+0005, -0.22431292511E+0005, -0.86875727899E+0004, -0.29871168410E+0004,     0.79115841589E+0003,    0.35705781947E+0004,   0.58006610869E+0004,     0.77051386777E+0004,    0.93985683515E+0004,   0.10936692503E+0005,     0.12340718829E+0005,    0.13610612778E+0005,   0.14733525527E+0005,     0.15689282338E+0005,    0.16460875548E+0005,   0.13781070397E+0005,     0.10353390446E+0005,    0.97015104443E+0004,   0.91970256955E+0004,     0.87091289709E+0004,    0.82380673106E+0004,   0.77832740909E+0004,     0.73578597173E+0004,    0.69600041379E+0004,   0.65672068765E+0004,     0.61826632544E+0004,    0.58164255669E+0004,   0.54681899424E+0004,     0.51375378900E+0004,    0.48240223653E+0004,   0.45293076395E+0004,     0.42524727350E+0004,   0.39900250682E+0004,   0.37408518611E+0004,     0.35024972785E+0004,   0.32723927636E+0004,   0.30533145611E+0004,     0.43975983863E+0004,   0.78406750693E+0004,   0.87186341472E+0004,     0.65536842039E+0004])  # noqa: E241
    ffprime = np.array([0.23322541115E+0002, 0.16954883014E+0002, 0.10891765459E+0002, 0.91092170305E+0001,  0.76935402448E+0001,    0.64621386431E+0001,    0.53455530560E+0001,  0.43313127345E+0001,    0.34216027576E+0001,    0.26198021012E+0001,  0.19257254741E+0001,    0.13348163773E+0001,    0.83896778862E+0000,  0.42786843211E+0000,    0.59386013251E-0001,    0.68368419851E-0001,  0.22945730638E+0000,    0.16694678396E+0000,    0.11280032899E+0000,  0.72841195527E-0001,    0.42955039549E-0001,    0.20754904215E-0001,  0.22833541706E-0002,   -0.13435625219E-0001,  -0.24231944405E-0001, -0.31351429021E-0001,   -0.36420352247E-0001,  -0.40084832835E-0001, -0.42795604604E-0001,   -0.45862442932E-0001,  -0.49152684336E-0001, -0.50916462484E-0001,   -0.51217447680E-0001,  -0.51347363097E-0001, -0.49299796315E-0001,   -0.45575773539E-0001,   0.66145734728E-0002,  0.80199882369E-0002,   -0.49896555629E-0001,  -0.31062128787E+0000, -0.68244434712E+0000])  # noqa: E241
    g2 = np.array([0.86472092673E-0002, 0.89334922228E+0001, 0.35953584660E+0002, 0.81614822011E+0002,    0.14627099362E+0003,     0.23016724551E+0003,  0.33360569863E+0003,    0.45698191650E+0003,     0.60077737676E+0003,  0.76555619797E+0003,    0.95196232244E+0003,     0.11607163719E+0004,  0.13926130437E+0004,    0.16485082232E+0004,     0.19293418957E+0004,  0.22368427133E+0004,    0.25725602699E+0004,     0.29377474891E+0004,  0.33343085482E+0004,    0.37640091797E+0004,     0.42289016947E+0004,  0.47313091183E+0004,    0.52739483131E+0004,     0.58599938300E+0004,  0.64931260094E+0004,    0.71777802776E+0004,     0.79192518801E+0004,  0.87240008494E+0004,    0.95998979793E+0004,     0.10557315889E+0005,  0.11609509254E+0005,    0.12772741029E+0005,     0.14068921018E+0005,  0.15529721176E+0005,    0.17195487173E+0005,     0.19173772923E+0005,  0.21472431112E+0005,    0.24045819919E+0005,     0.25699388332E+0005,  0.28376510254E+0005,    0.33925388768E+0005])  # noqa: E241
    g3 = np.array([0.11281276917E-0001, 0.11281705001E-0001, 0.11283395154E-0001, 0.11286674854E-0001,      0.11291921452E-0001,     0.11299283363E-0001,    0.11309067437E-0001,      0.11321656335E-0001,     0.11337514251E-0001,    0.11357185554E-0001,      0.11381288025E-0001,     0.11410499809E-0001,    0.11445540984E-0001,      0.11487066049E-0001,     0.11538449076E-0001,    0.11592681663E-0001,      0.11649239453E-0001,     0.11717037727E-0001,   0.11788923516E-0001,      0.11867420289E-0001,     0.11953028426E-0001,   0.12046018688E-0001,      0.12146939176E-0001,     0.12256363328E-0001,    0.12374630073E-0001,      0.12502471340E-0001,     0.12640733929E-0001,    0.12790364018E-0001,      0.12952468618E-0001,     0.13128841007E-0001,    0.13321589181E-0001,      0.13532799575E-0001,     0.13765732844E-0001,    0.14025721476E-0001,      0.14318020459E-0001,     0.14666643173E-0001,    0.15037183199E-0001,      0.15473563615E-0001,     0.15799707145E-0001,    0.16359179513E-0001,     0.17373280237E-0001])  # noqa: E241
    volprof = np.array([0.33944524629E-0001, 0.10947109461E+0001, 0.43039755104E+0001, 0.97177650322E+0001,      0.17372992555E+0002,      0.27292162087E+0002,      0.39502349712E+0002,      0.54039004608E+0002,      0.70944568492E+0002,      0.90267582900E+0002,      0.11206163623E+0003,      0.13638403503E+0003,      0.16329426633E+0003,      0.19285176039E+0003,      0.22511467787E+0003,      0.26016397744E+0003,      0.29811118532E+0003,      0.33904751686E+0003,      0.38305675651E+0003,      0.43024052590E+0003,      0.48069561449E+0003,      0.53452532513E+0003,      0.59184261506E+0003,      0.65277291519E+0003,      0.71745713107E+0003,      0.78605540839E+0003,      0.85875095039E+0003,      0.93575562588E+0003,      0.10173172498E+0004,      0.11037324515E+0004,      0.11953642691E+0004,      0.12926538652E+0004,      0.13961415747E+0004,      0.15065171330E+0004,      0.16246667377E+0004,      0.17519800647E+0004,      0.18902715619E+0004,      0.20402440300E+0004,      0.21950898614E+0004,      0.23521952591E+0004,      0.25278472700E+0004])  # noqa: E241
    vprime = np.array([0.90752201874E+0000, 0.29267585821E+0002, 0.59193963572E+0002, 0.89807889514E+0002,      0.12078619481E+0003,      0.15210075244E+0003,      0.18385379492E+0003,      0.21614362040E+0003,      0.24905995522E+0003,      0.28268036228E+0003,      0.31706601725E+0003,      0.35225789480E+0003,      0.38827497641E+0003,      0.42509475330E+0003,      0.46270954236E+0003,      0.50182299714E+0003,      0.54248819858E+0003,      0.58403772986E+0003,      0.62706209245E+0003,      0.67138469439E+0003,      0.71707442850E+0003,      0.76424412244E+0003,      0.81304805039E+0003,      0.86367571200E+0003,      0.91636456435E+0003,      0.97140781508E+0003,      0.10291471627E+0004,      0.10900334834E+0004,      0.11546067161E+0004,      0.12236815227E+0004,      0.12982756292E+0004,      0.13795204732E+0004,      0.14690321146E+0004,      0.15693424986E+0004,      0.16833287660E+0004,      0.18223719919E+0004,      0.19861551666E+0004,      0.21428638126E+0004,      0.21108972019E+0004,      0.22120696055E+0004,      0.26499510862E+0004])  # noqa: E241
    qprof = np.array([0.99996678674E-0001, 0.10056622457E+0000, 0.13787148137E+0000, 0.16019194892E+0000,      0.17729061481E+0000,     0.19284326842E+0000,      0.20865405381E+0000,      0.22569762899E+0000,     0.24455147130E+0000,      0.26557321963E+0000,      0.28898361654E+0000,     0.31491510330E+0000,      0.34344936282E+0000,      0.37465006374E+0000,     0.40870636562E+0000,      0.44670089689E+0000,      0.48827557374E+0000,     0.53295400972E+0000,      0.58052289697E+0000,      0.63122349581E+0000,     0.68530493063E+0000,      0.74307488124E+0000,      0.80491354311E+0000,     0.87137201615E+0000,      0.94298001031E+0000,      0.10204243240E+0001,     0.11045244994E+0001,      0.11962928780E+0001,     0.12969800765E+0001,     0.14082775474E+0001,      0.15325456243E+0001,     0.16723782066E+0001,     0.18312301333E+0001,     0.20143726927E+0001,     0.22286984904E+0001,     0.24924555066E+0001,     0.28025499489E+0001,     0.31329326474E+0001,     0.32659052628E+0001,     0.34787761319E+0001,     0.43991541073E+0001])  # noqa: E241
    psi = np.array([0.00000000000E+0000, 0.10199211327E+0001, 0.36549284819E+0001, 0.71014968481E+0001,      0.11337406079E+0002,      0.16272359734E+0002,      0.21800434134E+0002,      0.27813053338E+0002,      0.34207082057E+0002,      0.40890066407E+0002,      0.47783034520E+0002,      0.54821227986E+0002,      0.61953256026E+0002,      0.69139069940E+0002,      0.76347792350E+0002,      0.83549429308E+0002,      0.90714716674E+0002,      0.97824847564E+0002,      0.10487295892E+0003,      0.11185790158E+0003,      0.11877823026E+0003,      0.12563298513E+0003,      0.13242163182E+0003,      0.13914366764E+0003,      0.14579857883E+0003,      0.15238614449E+0003,      0.15890627523E+0003,      0.16535903526E+0003,      0.17174462027E+0003,      0.17806348774E+0003,      0.18431606651E+0003,      0.19050266151E+0003,      0.19662424633E+0003,      0.20268285956E+0003,      0.20868110166E+0003,      0.21462717902E+0003,      0.22053123730E+0003,      0.22639018657E+0003,      0.23211794208E+0003,      0.23781331337E+0003,     0.24361350586E+0003])  # noqa: E241
    phi = np.array([0.38233205842E-0003, 0.10266067900E+0000, 0.40911339867E+0000, 0.91988098052E+0000,      0.16328189297E+0001,      0.25445031965E+0001,      0.36525357814E+0001,      0.49563208719E+0001,      0.64572973336E+0001,     0.81589803088E+0001,      0.10066845579E+0002,     0.12188105072E+0002,      0.14531431874E+0002,      0.17106625614E+0002,      0.19924788547E+0002,      0.22998878087E+0002,      0.26341942543E+0002,      0.29965531581E+0002,      0.33882324638E+0002,      0.38106905386E+0002,      0.42654622989E+0002,      0.47542211760E+0002,      0.52788199960E+0002,      0.58413370029E+0002,      0.64441141813E+0002,      0.70898107756E+0002,      0.77814728789E+0002,      0.85226230813E+0002,      0.93173751781E+0002,     0.10170636722E+0003,      0.11088381503E+0003,      0.12077872579E+0003,      0.13148049894E+0003,      0.14310358753E+0003,      0.15579660310E+0003,      0.16978896785E+0003,      0.18536636869E+0003,      0.20270034176E+0003,     0.22101791766E+0003,      0.24020551793E+0003,      0.26274043316E+0003])  # noqa: E241
    shif = np.array([0.43201729748E+0000, 0.43200947321E+0000, 0.43190860545E+0000, 0.43165009197E+0000,      0.43118646435E+0000,      0.43047122924E+0000,      0.42947167136E+0000,      0.42814235103E+0000,      0.42642800462E+0000,      0.42426438373E+0000,      0.42157925040E+0000,      0.41829388822E+0000,      0.41432519278E+0000,      0.40959073645E+0000,      0.40394257836E+0000,      0.39736780707E+0000,      0.39005856249E+0000,      0.38197768149E+0000,      0.37305367404E+0000,      0.36335708200E+0000,      0.35289426465E+0000,      0.34167049236E+0000,      0.32969197134E+0000,      0.31696296447E+0000,      0.30349187125E+0000,      0.28929092364E+0000,      0.27437140704E+0000,      0.25874536370E+0000,      0.24242741260E+0000,      0.22543132987E+0000,      0.20776822225E+0000,      0.18945495741E+0000,      0.17051518915E+0000,      0.15096767624E+0000,      0.13083693035E+0000,      0.11006259996E+0000,      0.88792815919E-0001,      0.67090861912E-0001,      0.45385556496E-0001,     0.23359828914E-0001,   0.00000000000E+0000])  # noqa: E241
    kprof = np.array([0.10773002385E+0001, 0.10798810926E+0001, 0.10868197183E+0001, 0.10955610447E+0001,      0.11035402555E+0001,      0.11104357234E+0001,      0.11167177873E+0001,      0.11228144464E+0001,      0.11289946429E+0001,      0.11354213358E+0001,      0.11421909386E+0001,      0.11493560426E+0001,      0.11569389634E+0001,      0.11649365167E+0001,      0.11733397400E+0001,      0.11822288353E+0001,      0.11917327891E+0001,      0.12018583676E+0001,      0.12125795311E+0001,      0.12239055297E+0001,      0.12358266544E+0001,      0.12483450664E+0001,      0.12614776501E+0001,      0.12752569840E+0001,      0.12897305021E+0001,      0.13049614439E+0001,      0.13210299984E+0001,      0.13380357799E+0001,      0.13561014603E+0001,      0.13753832996E+0001,      0.13960840602E+0001,      0.14184532207E+0001,      0.14427986942E+0001,      0.14695296254E+0001,      0.14991818477E+0001,      0.15327151781E+0001,      0.15712732998E+0001,      0.16148631476E+0001,      0.16570455689E+0001,      0.16960789687E+0001,      0.17461968078E+0001])  # noqa: E241
    press = np.array([0.17513987699E+0007, 0.17995610164E+0007, 0.18333835399E+0007, 0.18523951483E+0007,      0.18563319982E+0007,      0.18451395429E+0007,      0.18189729383E+0007,      0.17781958904E+0007,      0.17233779953E+0007,      0.16552906889E+0007,      0.15749019774E+0007,      0.14833701405E+0007,      0.13820365790E+0007,      0.12724178501E+0007,      0.11561966339E+0007,      0.10352107173E+0007,      0.95820503792E+0006,      0.88741173793E+0006,      0.82084870256E+0006,      0.75834937921E+0006,      0.69974590081E+0006,      0.64486931346E+0006,      0.59355556661E+0006,      0.54545960663E+0006,      0.50045195982E+0006,      0.45849352137E+0006,      0.41941148673E+0006,      0.38303766763E+0006,      0.34920897737E+0006,      0.31776806148E+0006,      0.28856403939E+0006,      0.26142721652E+0006,      0.23622384016E+0006,      0.21282792087E+0006,      0.19112093035E+0006,      0.17099297435E+0006,      0.15234263213E+0006,      0.13507681203E+0006,      0.10138895773E+0006,      0.45511786254E+0005,      0.11633426195E+0004])  # noqa: E241
    dprof = np.array([0.00000000000E+0000, 0.19744630848E-0003, 0.46561124132E-0003, 0.83912959078E-0003,      0.12911890855E-0002,      0.18068677519E-0002,      0.23973387433E-0002,      0.30847369392E-0002,      0.38960785315E-0002,      0.48632409560E-0002,      0.60230344120E-0002,      0.74168604800E-0002,      0.90899272690E-0002,      0.11089021539E-0001,      0.13479579887E-0001,      0.16302180510E-0001,      0.19533498171E-0001,      0.23178990193E-0001,      0.27270525106E-0001,      0.31828696328E-0001,      0.36887508193E-0001,      0.42484424661E-0001,      0.48662765250E-0001,      0.55475228563E-0001,     0.62983006655E-0001,     0.71257299865E-0001,     0.80383394063E-0001,      0.90464033775E-0001,      0.10162364024E+0000,      0.11401990265E+0000,      0.12785804610E+0000,      0.14339429174E+0000,     0.16094984235E+0000,      0.18095739450E+0000,      0.20399317942E+0000,     0.23109789045E+0000,      0.26327628148E+0000,      0.30056597073E+0000,     0.33740662361E+0000,      0.37321552658E+0000,      0.42389365851E+0000])  # noqa: E241
    # fmt:on

    # Convert to x = sqrt()
    rho = np.sqrt(psi / psi[-1])
    # Flip sign of psi so that peak is at centre
    psi = -(psi - np.max(psi))  # noqa: PIE794
    psi_ax = psi[0]
    psi_b = psi[-1]

    R_0 = 8.98300000  # [m]
    B_0 = 5.31000000  # [T]
    I_p = 20.501396465e6  # [A]
    amin = 2.9075846464  # [m]
    n = len(rho)
    a = np.linspace(0, amin, n)


class TestPLASMODVerificationMetricCoefficients(PLASMODVerificationRawData):
    """
    A verification test for which a series of flux surfaces are calculated
    using the parameterisation used in EMEQ, in PLASMOD.

    Flux surface metric coefficients are then calculated for these analytical
    flux surfaces for a given PLASMOD run, and are compared for equality.

    Note that PLASMOD uses a number of smoothing and extrapolation tricks,
    which we have not mimicked here. As such, it is to be expected that
    the comparison is not perfect.
    """

    @classmethod
    def setup_class(cls):
        # Build flux surfaces
        flux_surfaces = []
        f, ax = plt.subplots()
        for i in range(cls.n):
            fs = flux_surface_zakharov(
                cls.R_0 + cls.shif[i], 0, cls.a[i], cls.kprof[i], cls.dprof[i], n=1000
            )
            fs.close()
            flux_surfaces.append(ClosedFluxSurface(fs))
            ax.plot(*fs.xz)

        ax.set_aspect("equal")
        plt.show()
        cls.flux_surfaces = flux_surfaces
        cls._calculate_metrics(cls)

    def _calculate_metrics(self):
        """
        This effectively mocks a FEM fixed boundary equilibrium by fully analytical
        specification of the flux surfaces, with interpolation on psi and psi_norm
        between them.
        """
        # Extract coordinates
        x = []
        z = []
        psi = []
        for i, fs in enumerate(self.flux_surfaces):
            x.append(fs.coords.x[:-1])
            z.append(fs.coords.z[:-1])
            psi.append(self.psi[i] * np.ones(len(fs.coords.x) - 1))
        # Now, add a fictitious flux surface to enable gradient calculation at
        # boundary
        delta = 1e-8
        delta_psi = (self.psi[-1] - self.psi[-2]) / (self.rho[-1] - self.rho[-2])
        x_off, z_off = offset(x[-1], z[-1], delta)

        x.append(x_off)
        z.append(z_off)
        psi.append((self.psi[-1] - abs(delta_psi) * delta) * np.ones(len(x_off)))
        x2d = np.concatenate(x)
        z2d = np.concatenate(z)
        psi2d = np.concatenate(psi)

        # Make a callable for grad_psi, mimicking a G-S solver.
        # Thank you matplotlib, again..
        tri = Triangulation(x2d, z2d)
        lti = LinearTriInterpolator(tri, psi2d)

        def f_grad_psi(x):
            return lti.gradient(x[0], x[1])

        x1D, volume, g1, g2, g3 = calc_metric_coefficients(
            self.flux_surfaces[1:], f_grad_psi, self.rho, self.psi_ax
        )
        self.results = {
            "x_1d": x1D,
            "V": volume,
            "g1": g1,
            "g2": g2,
            "g3": g3,
        }

    def test_volume(self):
        f, ax = plt.subplots()
        ax.plot(
            self.rho,
            self.volprof,
            label="$V_{PLASMOD}$",
        )
        ax.plot(
            self.rho,
            self.results["V"],
            label="$V_{Zakharov}$",
        )
        ax.set_ylabel("[$m^3$]")
        ax.set_xlabel("$\\rho$")
        ax.legend()
        plt.show()
        np.testing.assert_allclose(self.results["V"][1:], self.volprof[1:], rtol=5e-2)

    def test_gradV(self):
        """
        For reasons not yet understood, the vprime in PLASMOD (or at least in
        this run) is by radius, with a constant minor radius division.
        """
        volume = self.results["V"]
        f_volume = interp1d(self.a, volume, fill_value="extrapolate")
        grad_vol = approx_derivative(f_volume, self.a).diagonal()

        f, ax = plt.subplots()
        ax.plot(self.rho, grad_vol, label="$V^'$ calculated")
        ax.plot(self.rho, self.vprime, label="$V^'$ PLASMOD")
        ax.set_xlabel("$\\rho$")
        ax.legend()
        plt.show()
        np.testing.assert_allclose(grad_vol[1:], self.vprime[1:], rtol=9e-2)

    def test_g2(self):
        f, ax = plt.subplots()
        ax.plot(self.results["x_1d"], self.results["g2"], label="$g_2$ calculated")
        ax.plot(self.rho, self.g2, label="$g_2$ PLASMOD")
        ax.set_xlabel("$g_{2}$")
        ax.set_xlabel("$\\rho$")
        ax.legend()
        plt.show()
        np.testing.assert_allclose(self.results["g2"][1:], self.g2[1:], rtol=0.26)

    def test_g3(self):
        f, ax = plt.subplots()
        ax.plot(self.results["x_1d"], self.results["g3"], label="$g_3$ calculated")
        ax.plot(self.rho, self.g3, label="$g_3$ PLASMOD")
        ax.set_xlabel("$g_{3}$")
        ax.set_xlabel("$\\rho$")
        ax.legend()
        plt.show()
        np.testing.assert_allclose(self.results["g3"], self.g3, rtol=0.02)


class TestPLASMODVerificationCurrentProfiles(PLASMODVerificationRawData):
    """
    A verification test for which we take the raw PLASMOD profiles
    and recalculate p' and FF', among other things.

    Note that PLASMOD uses a number of smoothing and extrapolation tricks,
    which we have not mimicked here. As such, it is to be expected that
    the comparison is not perfect.
    """

    @classmethod
    def setup_class(cls):
        I_p, phi_1D, psi_1D, pprime, F, ff_prime = calc_curr_dens_profiles(
            cls.rho,
            cls.press,
            cls.qprof,
            cls.g2,
            cls.g3,
            cls.volprof,
            0,
            cls.B_0,
            cls.R_0,
            cls.psi_ax,
            cls.psi_b,
        )
        cls.results = {
            "I_p": I_p,
            "phi_1D": phi_1D,
            "psi_1D": psi_1D,
            "pprime": pprime,
            "F": F,
            "FFprime": ff_prime,
        }
        cls.f, cls.ax = plt.subplots(2, 4)

    @classmethod
    def teardown_class(cls):
        for i, a in enumerate(cls.ax.flat):
            if i < 4:
                a.legend()
            else:
                a.set_xlabel("x")
        cls.ax[1, 0].set_ylabel("PLASMOD-bluemira")
        plt.show()

    def test_plasma_current(self):
        # 15/03/23: Max relative difference: 4.71793662e-05
        np.testing.assert_allclose(self.results["I_p"], self.I_p, rtol=5e-5)

    def test_psi(self):
        self.ax[0, 0].plot(self.rho, self.psi, label="PLASMOD")
        self.ax[0, 0].plot(self.rho, self.results["psi_1D"], ls="--", label="bluemira")
        self.ax[0, 0].set_title("$\\psi$")
        self.ax[1, 0].plot(self.rho, self.psi - self.results["psi_1D"])
        np.testing.assert_allclose(self.results["psi_1D"], self.psi, rtol=0.01)

    def test_phi(self):
        self.ax[0, 1].plot(self.rho, self.phi, label="PLASMOD")
        self.ax[0, 1].plot(self.rho, self.results["phi_1D"], ls="--", label="bluemira")
        self.ax[0, 1].set_title("$\\phi$")
        self.ax[1, 1].plot(self.rho, self.phi - self.results["phi_1D"])
        np.testing.assert_allclose(self.results["phi_1D"][1:], self.phi[1:], rtol=0.0175)

    def test_pprime(self):
        self.ax[0, 2].plot(self.rho, self.pprime, label="PLASMOD")
        self.ax[0, 2].plot(self.rho, self.results["pprime"], ls="--", label="bluemira")
        self.ax[0, 2].set_title("$p^{'}$")
        self.ax[1, 2].plot(self.rho, self.pprime - self.results["pprime"])
        np.testing.assert_allclose(self.results["pprime"], self.pprime, rtol=0.35)

    def test_ffprime(self):
        self.ax[0, 3].plot(self.rho, self.ffprime, label="PLASMOD")
        self.ax[0, 3].plot(self.rho, self.results["FFprime"], ls="--", label="bluemira")
        self.ax[0, 3].set_title("$FF^{'}$")
        self.ax[1, 3].plot(self.rho, self.ffprime - self.results["FFprime"])
        np.testing.assert_allclose(self.results["FFprime"], self.ffprime, rtol=0.25)
