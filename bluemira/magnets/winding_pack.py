import matplotlib.pyplot as plt
import numpy as np

from bluemira.magnets.base import StructuralComponent, parall_k, serie_k
from bluemira.magnets.conductor import Conductor


class WindingPack(StructuralComponent):
    def __init__(self, conductor: Conductor, nl: np.int32, nt: np.int32, name: str = ""):
        self.name = name
        self.nl = nl
        self.nt = nt
        self.conductor = conductor

    @property
    def dx(self):
        return self.conductor.dx * self.nl

    @property
    def dy(self):
        return self.conductor.dy * self.nt

    def Kx(self, **kwargs):
        """Total equivalent stiffness along x-axis"""
        return serie_k([parall_k([self.conductor.Ky(**kwargs)] * self.nl)] * self.nt)

    def Ky(self, **kwargs):
        """Total equivalent stiffness along x-axis"""
        return serie_k([parall_k([self.conductor.Kx(**kwargs)] * self.nt)] * self.nl)

    def Xx(self, **kwargs):
        return 0

    def Yy(self, **kwargs):
        return 0

    def plot(
            self,
            xc: float = 0,
            yc: float = 0,
            show: bool = False,
            ax=None,
            homogenized: bool = True,
    ):
        if ax is None:
            _, ax = plt.subplots()

        ax.plot([0], [0])

        pc = np.array([xc, yc])
        a = self.dx / 2
        b = self.dy / 2

        p0 = np.array([-a, -b])
        p1 = np.array([a, -b])
        p2 = np.array([a, b])
        p3 = np.array([-a, b])

        points_ext = np.vstack((p0, p1, p2, p3, p0)) + pc

        ax.fill(points_ext[:, 0], points_ext[:, 1], "gold")
        ax.plot(points_ext[:, 0], points_ext[:, 1], "k")

        if not homogenized:
            for i in range(self.nl):
                for j in range(self.nt):
                    xc_c = xc - self.dx / 2 + (i + 0.5) * self.conductor.dx
                    yc_c = yc - self.dy / 2 + (j + 0.5) * self.conductor.dy
                    self.conductor.plot(xc=xc_c, yc=yc_c, ax=ax)

        if show:
            plt.show()
        return ax
