import matplotlib.pyplot as plt
import numpy as np

from bluemira.magnets.base import parall_k
from bluemira.magnets.cable import DummySquareCable
from bluemira.magnets.case import CaseTF
from bluemira.magnets.conductor import SquareConductor
from bluemira.magnets.materials import Copper300, Material, OperationalPoint
from bluemira.magnets.strand import Strand, Wire_NbTi
from bluemira.magnets.utils import (
    delayed_exp_func,
    optimize_jacket_conductor,
    optimize_n_stab_cable,
)
from bluemira.magnets.winding_pack import WindingPack


class DummyInsulator(Material):
    def res(self, **kwargs):
        return 1e6

    def ym(self, **kwargs):
        return 1e7


# plot options
show = True

# input values
I0 = 2.0e4
B0 = 6
T0 = 6.8
t_delay = 3
tau = 20
t0 = 0
tf = tau
hotspot_target_temperature = 250.0

# Current and magnetic field behaviour
I = delayed_exp_func(I0, tau, t_delay)
B = delayed_exp_func(B0, tau, t_delay)

# define the conductor (with a dummy number of stabilizer strands )
sc_strand = Wire_NbTi()
copper300 = Copper300()
insulator = DummyInsulator()
stab_strand = Strand([copper300], [1])
cable = DummySquareCable(sc_strand, stab_strand, 100, 100, 0.005)
# c.plot(0,0, show=True)

conductor = SquareConductor(
    cable=cable, mat_jacket=copper300, mat_ins=insulator, dx_jacket=0.01, dx_ins=5e-3
)

# optimize the number of stabilizer strands using the hot spot criteria
result = optimize_n_stab_cable(
    cable, t0, tf, T0, hotspot_target_temperature, B, I, bounds=[1, 10000], show=show
)

# optimize the cable jacket thickness considering 0D stress model for the single cable
MU_0 = 4 * np.pi * 1e-7
pm = B0 ** 2 / (2 * MU_0)
allowable_sigma = 667e6

result_opt_jacket = optimize_jacket_conductor(
    conductor, pm, T0, B0, allowable_sigma, bounds=[1e-7, 1]
)

# creation of WPs
wp1 = WindingPack(conductor, 50, 30)
n_conductors = wp1.nl * wp1.nt
case = CaseTF(
    Ri=5, dy_ps=0.1, dy_vault=0.6, theta_TF=360 / 16, mat_case=copper300, WPs=[wp1]
)

if show:
    ax = case.plot(homogenized=False)
    ax.set_aspect("equal")
    plt.show()

case.rearrange_conductors_in_wp_type1(
    n_conductors, conductor, case.R_wp_i[0], 0.2, 0.075, 4
)
if show:
    ax = case.plot(homogenized=False)
    ax.set_aspect("equal")
    plt.show()

op = OperationalPoint(T=T0, B=B0)

temp = [
    parall_k([case.Kx_lat(**op)[i], w.Kx(**op), case.Kx_lat(**op)[i]])
    for i, w in enumerate(case.WPs)
]

sigma_r_jacket_corrected = pm * case.Xx(T=T0, B=B0)
print(f"corrected sigma_r jacket: {sigma_r_jacket_corrected / 1e6} MPa")
