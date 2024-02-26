import matplotlib.pyplot as plt
import numpy as np

from bluemira.magnets.base import StructuralComponent, parall_r, serie_r
from bluemira.magnets.strand import Strand


class Cable(StructuralComponent):
    def __init__(
            self,
            dx: float,
            sc_strand: Strand,
            stab_strand: Strand,
            n_sc_strand: int,
            n_stab_strand: int,
            d_cooling_channel: float,
            void_fraction: float = 0.725,
            cos_theta: float = 0.97,
            name: str = "",
    ):
        """
        Representation of a cable

        Parameters
        ----------
        dx:
            x-dimension of the cable [m]
        sc_strand:
            strand of the superconductor
        stab_strand:
            strand of the stabilizer
        d_cooling_channel:
            diameter of the cooling channel
        n_sc_strand:
            number of superconducting strands
        n_stab_strand:
            number of stabilizer strands
        void_fraction:
            void fraction defined as material_volume/total_volume
        cos_theta:
            corrective factor that consider the twist of the cable
        name:
            cable string identifier

        #todo decide if it is the case to add also the cooling material
        """
        self.name = name
        self._dx = dx
        self.sc_strand = sc_strand
        self.stab_strand = stab_strand
        self.void_fraction = void_fraction
        self.d_cooling_channel = d_cooling_channel
        self.n_sc_strand = int(n_sc_strand)
        self._n_stab_strand = int(n_stab_strand)
        self.cos_theta = cos_theta
        self._check_consistency()

    @property
    def n_stab_strand(self):
        return self._n_stab_strand

    @n_stab_strand.setter
    def n_stab_strand(self, value):
        self._n_stab_strand = int(np.ceil(value))

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
            self.sc_strand.res(**kwargs) / self.area_sc,
            self.stab_strand.res(**kwargs) / self.area_stab,
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
        """
        weighted_specific_heat = np.array([
            self.sc_strand.cp_v(**kwargs) * self.area_sc,
            self.stab_strand.cp_v(**kwargs) * self.area_stab,
        ])
        return serie_r(weighted_specific_heat) / (self.area_sc + self.area_stab)

    def _check_consistency(self):
        """Check consistency and return True if all checks are passed."""
        if self.dx <= self.d_cooling_channel or self.dy <= self.d_cooling_channel:
            print("WARNING: inconsistency between dx, dy and d_cooling_channel")
            return False
        return True

    @property
    def area_stab(self):
        """Area of the stabilizer"""
        return self.stab_strand.area * self.n_stab_strand

    @property
    def area_sc(self):
        """Area of the superconductor"""
        return self.sc_strand.area * self.n_sc_strand

    @property
    def area_cc(self):
        """Area of the cooling"""
        return self.d_cooling_channel ** 2 / 4 * np.pi

    @property
    def area(self):
        """Area of the cable considering the void fraction"""
        return (
                self.area_sc + self.area_stab
        ) / self.void_fraction / self.cos_theta + self.area_cc

    @property
    def dx(self):
        return self._dx

    @dx.setter
    def dx(self, value: float):
        self._dx = value

    @property
    def dy(self):
        """y-dimension of the cable [m]"""
        return self.area / self.dx

    def ym(self, **kwargs):
        return 0

    def Kx(self, **kwargs):
        """Total equivalent stiffness along x-axis"""
        return self.ym(**kwargs) * self.dy / self.dx

    def Ky(self, **kwargs):
        """Total equivalent stiffness along y-axis"""
        return self.ym(**kwargs) * self.dx / self.dy

    def Xx(self, **kwargs):
        return 0

    def Yy(self, **kwargs):
        return 0

    def plot(self, xc: float = 0, yc: float = 0, show: bool = False, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()

        pc = np.array([xc, yc])
        a = self.dx / 2
        b = self.dy / 2

        p0 = np.array([-a, -b])
        p1 = np.array([a, -b])
        p2 = np.array([[a, b]])
        p3 = np.array([-a, b])

        points_ext = np.vstack((p0, p1, p2, p3, p0)) + pc
        points_cc = (
                np.array([
                    np.array([np.cos(theta), np.sin(theta)]) * self.d_cooling_channel / 2
                    for theta in np.linspace(0, np.radians(360), 19)
                ])
                + pc
        )

        ax.fill(points_ext[:, 0], points_ext[:, 1], "gold")
        ax.fill(points_cc[:, 0], points_cc[:, 1], "r")

        if show:
            plt.show()
        return ax


class SquareCable(Cable):
    def __init__(
            self,
            sc_strand: Strand,
            stab_strand: Strand,
            n_sc_strand: int,
            n_stab_strand: int,
            d_cooling_channel: float,
            void_fraction: float = 0.725,
            cos_theta: float = 0.97,
            name: str = "",
    ):
        """
        Representation of a square cable

        Parameters
        ----------
        sc_strand:
            strand of the superconductor
        stab_strand:
            strand of the stabilizer
        d_cooling_channel:
            diameter of the cooling channel
        n_sc_strand:
            number of superconducting strands
        n_stab_strand:
            number of stabilizer strands
        void_fraction:
            void fraction defined as material_volume/total_volume
        cos_theta:
            corrective factor that consider the twist of the cable
        name:
            cable string identifier

        #todo decide if it is the case to add also the cooling material
        """
        dx = 0.1
        super().__init__(
            dx=dx,
            sc_strand=sc_strand,
            stab_strand=stab_strand,
            n_sc_strand=n_sc_strand,
            n_stab_strand=n_stab_strand,
            d_cooling_channel=d_cooling_channel,
            void_fraction=void_fraction,
            cos_theta=cos_theta,
            name=name,
        )

    @property
    def dx(self):
        return np.sqrt(self.area)


class DummySquareCable(SquareCable):
    def ym(self, **kwargs):
        return 120
