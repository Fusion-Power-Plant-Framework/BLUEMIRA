import matplotlib.pyplot as plt
import numpy as np

from bluemira.magnets.base import (
    StructuralComponent,
    parall_k,
    parall_r,
    serie_k,
    serie_r,
)
from bluemira.magnets.cable import Cable
from bluemira.magnets.materials import Material


class Conductor(StructuralComponent):
    def __init__(
            self,
            cable: Cable,
            mat_jacket: Material,
            mat_ins: Material,
            dx_jacket: float,
            dy_jacket: float,
            dx_ins: float,
            dy_ins: float,
            name: str = "",
    ):
        """

        Parameters
        ----------
        cable:
            the conductor cable
        mat_jacket:
            jacket's material
        mat_ins:
            insulator's material
        dx_jacket:
            x-thickness of the jacket
        dy_jacket:
            y-tickness of the jacket
        dx_ins:
            x-thickness of the insulator
        dy_ins:
            y-tickness of the insulator
        name:
            string identifier
        """
        self.name = name
        self._dx_jacket = dx_jacket
        self._dy_jacket = dy_jacket
        self._dy_ins = dy_ins
        self._dx_ins = dx_ins
        self.mat_ins = mat_ins
        self.mat_jacket = mat_jacket
        self.cable = cable

    @property
    def dx(self):
        """x-dimension of the conductor [m]"""
        return self.dx_ins * 2 + self.dx_jacket * 2 + self.cable.dx

    @property
    def dy(self):
        """y-dimension of the conductor [m]"""
        return self.dy_ins * 2 + self.dy_jacket * 2 + self.cable.dy

    @property
    def dx_jacket(self):
        return self._dx_jacket

    @dx_jacket.setter
    def dx_jacket(self, value):
        self._dx_jacket = value

    @property
    def dy_jacket(self):
        return self._dy_jacket

    @dy_jacket.setter
    def dy_jacket(self, value):
        self._dy_jacket = value

    @property
    def dx_ins(self):
        return self._dx_ins

    @dx_ins.setter
    def dx_ins(self, value):
        self._dx_ins = value

    @property
    def dy_ins(self):
        return self._dy_ins

    @dy_ins.setter
    def dy_ins(self, value):
        self._dy_ins = value

    @property
    def area(self):
        """Area of the conductor [m^2]"""
        return self.dx * self.dy

    @property
    def area_jacket(self):
        """Area of the jacket"""
        return (self.cable.dx + self.dx_jacket) * (
                self.cable.dy + self.dy_jacket
        ) - self.cable.area

    @property
    def area_ins(self):
        """Area of the insulator"""
        return self.area - self.area_jacket - self.cable.area

    def res(self, **kwargs):
        """
        Cable's equivalent resistivity, computed as the parallel between strands' resistivity

        Parameters
        ----------
        Return
        ------
            float [Ohm m]
        """
        resistances = np.array([
            self.cable.res(**kwargs) / self.cable.area,
            self.mat_jacket.res(**kwargs) / self.area_jacket,
        ])
        res_tot = parall_r(resistances)
        return res_tot * self.area

    def cp_v(self, **kwargs):
        """
        Strand's equivalent Specific Heat, compute the series between strand's components

        Parameters
        ----------
        Return
        ------
            float [J/K/m]

        TODO: decide if also the insulator should be considered
        """
        weighted_specific_heat = np.array([
            self.cable.cp_v(**kwargs) * self.cable.area,
            self.mat_jacket.cp_v(**kwargs) * self.area_jacket,
        ])
        return serie_r(weighted_specific_heat) / self.area

    def saf_x(self, **kwargs):
        return self.dx / (
                self.cable.dx * self.cable.ym(**kwargs) / self.cable.ym
                + self.dx_ins * 2 * self.mat_ins.ym(**kwargs)
                + self.dx_jacket * 2 * self.mat_jacket.ym(**kwargs)
        )

    def Kx_topbot_ins(self, **kwargs):
        return self.mat_ins.ym(**kwargs) * self.dy_ins / self.dx

    def Kx_lat_ins(self, **kwargs):
        return self.mat_ins.ym(**kwargs) * (self.dy - 2 * self.dy_ins) / self.dx_ins

    def Kx_topbot_jacket(self, **kwargs):
        return (
                self.mat_jacket.ym(**kwargs) * self.dy_jacket / (
                    self.dx - 2 * self.dx_ins)
        )

    def Kx_lat_jacket(self, **kwargs):
        return (
                self.mat_jacket.ym(**kwargs)
                * (self.dy - 2 * self.dy_ins - 2 * self.dy_jacket)
                / self.dx_jacket
        )

    def Kx_cable(self, **kwargs):
        return self.cable.ym(**kwargs) * self.cable.dy / self.cable.dx

    def Kx(self, **kwargs):
        return serie_k([
            2 * self.Kx_topbot_ins(**kwargs),
            serie_k([
                self.Kx_lat_ins(**kwargs) / 2,
                parall_k([
                    2 * self.Kx_topbot_jacket(**kwargs),
                    serie_k([self.Kx_cable(**kwargs), self.Kx_lat_jacket(**kwargs) / 2]),
                ]),
            ]),
        ])

    def Xx(self, **kwargs):
        return 2 * self.Kx_lat_jacket(**kwargs) / self.Kx(**kwargs)

    def saf_y(self, **kwargs):
        return self.dy / (
                self.cable.dy * self.cable.ym(**kwargs) / self.cable.ym
                + self.dy_ins * 2 * self.mat_ins.ym(**kwargs)
                + self.dy_jacket * 2 * self.mat_jacket.ym(**kwargs)
        )

    def Ky_topbot_ins(self, **kwargs):
        return self.mat_ins.ym(**kwargs) * (self.dx - 2 * self.dx_ins) / self.dy_ins

    def Ky_lat_ins(self, **kwargs):
        return self.mat_ins.ym(**kwargs) * self.dx_ins / self.dy

    def Ky_topbot_jacket(self, **kwargs):
        return (
                self.mat_jacket.ym(**kwargs)
                * (self.dx - 2 * self.dx_ins - 2 * self.dx_jacket)
                / self.dy_jacket
        )

    def Ky_lat_jacket(self, **kwargs):
        return (
                self.mat_jacket.ym(**kwargs) * self.dx_jacket / (
                    self.dy - 2 * self.dy_ins)
        )

    def Ky(self, **kwargs):
        return serie_k([
            2 * self.Ky_topbot_ins(**kwargs),
            serie_k([
                self.Ky_lat_ins(**kwargs) / 2,
                parall_k([
                    2 * self.Ky_topbot_jacket(**kwargs),
                    serie_k([self.cable.Ky(**kwargs), self.Ky_lat_jacket(**kwargs) / 2]),
                ]),
            ]),
        ])

    def Yy(self, **kwargs):
        return 2 * self.Ky_lat_jacket(**kwargs) / self.Ky(**kwargs)

    def plot(self, xc: float = 0, yc: float = 0, show: bool = False, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        pc = np.array([xc, yc])
        a = self.cable.dx / 2 + self.dx_jacket
        b = self.cable.dy / 2 + self.dy_jacket

        p0 = np.array([-a, -b])
        p1 = np.array([a, -b])
        p2 = np.array([[a, b]])
        p3 = np.array([-a, b])
        points_ext_jacket = np.vstack((p0, p1, p2, p3, p0)) + pc

        c = a + self.dx_ins
        d = b + self.dy_ins

        p0 = np.array([-c, -d])
        p1 = np.array([c, -d])
        p2 = np.array([[c, d]])
        p3 = np.array([-c, d])
        points_ext_ins = np.vstack((p0, p1, p2, p3, p0)) + pc

        ax.fill(points_ext_ins[:, 0], points_ext_ins[:, 1], "red")
        ax.fill(points_ext_jacket[:, 0], points_ext_jacket[:, 1], "blue")

        ax = self.cable.plot(xc=xc, yc=yc, show=False, ax=ax)

        if show:
            ax.set_aspect("equal")
            plt.show()

        return ax


class SquareConductor(Conductor):
    def __init__(
            self,
            cable: Cable,
            mat_jacket: Material,
            mat_ins: Material,
            dx_jacket: float,
            dx_ins: float,
            name: str = "",
    ):
        dy_jacket = dx_jacket
        dy_ins = dx_ins
        super().__init__(
            cable=cable,
            mat_jacket=mat_jacket,
            mat_ins=mat_ins,
            dx_jacket=dx_jacket,
            dy_jacket=dy_jacket,
            dx_ins=dx_ins,
            dy_ins=dy_ins,
            name=name,
        )

    @property
    def dy_jacket(self):
        return self.dx_jacket

    @property
    def dy_ins(self):
        return self.dx_ins
