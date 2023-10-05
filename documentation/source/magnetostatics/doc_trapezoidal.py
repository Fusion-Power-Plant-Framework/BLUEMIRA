import matplotlib.pyplot as plt
import numpy as np

from bluemira.magnetostatics.trapezoidal_prism import TrapezoidalPrismCurrentSource

source = TrapezoidalPrismCurrentSource(
    origin=[1, 1, 1],  # the centroid of the current source
    ds=[0, 0, 4],  # length of the source is determined by the norm of ds
    normal=[0, 1, 0],
    t_vec=[1, 0, 0],
    breadth=0.5,  # in t_vec direction
    depth=0.25,  # in normal direction
    alpha=45.0,  # angle at the tip of the current source
    beta=22.5,  # angle at the tail of the current source
    current=1e6,
)

x = np.linspace(0, 2, 100)
y = np.linspace(0, 2, 100)
xx, yy = np.meshgrid(x, y)

# Calculate field values in global x, y, z Cartesian coordinates.
Bx, By, Bz = source.field(xx, yy, np.ones_like(xx))
B = np.sqrt(Bx**2 + By**2 + Bz**2)

source.plot()
ax = plt.gca()
ax.contourf(xx, yy, B, zdir="z", offset=1)
