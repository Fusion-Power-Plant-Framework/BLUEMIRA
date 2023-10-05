import matplotlib.pyplot as plt
import numpy as np

from bluemira.magnetostatics.circular_arc import CircularArcCurrentSource

source = CircularArcCurrentSource(
    origin=[1, 1, 1],
    ds=[1, 0, 0],
    normal=[0, 1, 0],
    t_vec=[0, 0, 1],
    breadth=0.25,
    depth=1,
    radius=2,
    dtheta=270,
    current=1e6,
)

x = np.linspace(-2, 4, 100)
y = np.linspace(-2, 4, 100)
xx, yy = np.meshgrid(x, y)


# Calculate field values in global x, y, z Cartesian coordinates.
Bx, By, Bz = source.field(xx, yy, 0.25 * np.ones_like(xx))
B = np.sqrt(Bx**2 + By**2 + Bz**2)

source.plot()
ax = plt.gca()
ax.contourf(xx, yy, B, zdir="z", offset=0.25)
