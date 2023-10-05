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
import pytest

from bluemira.base.constants import T_LAMBDA
from bluemira.fuel_cycle.timeline_tools import (
    generate_exponential_distribution,
    generate_lognorm_distribution,
    generate_truncnorm_distribution,
)
from bluemira.fuel_cycle.tools import _dec_I_mdot, _find_t15, _fountain_linear_sink


@pytest.mark.parametrize(
    ("n", "integral", "parameter"),
    [(100, 600, 1.0), (6000, 1.5e7, 1.5), (6000, 1.5e7, 0.5)],
)
def test_distributions(n, integral, parameter):
    for func in [
        generate_lognorm_distribution,
        generate_truncnorm_distribution,
        generate_exponential_distribution,
    ]:
        d = func(n, integral, parameter)
        assert len(d) == n
        assert np.isclose(np.sum(d), integral)


class TestSinkTools:
    def setup_method(self):
        self.I_min, self.I_max = 3.0, 5.0
        self.t_in, self.t_out = 5.0, 6.0

    @classmethod
    def teardown_class(cls):
        plt.close("all")

    def test_timestep(self):
        """
        th \\n
        :math:`I_{end} = Ie^{-{\\lambda}{\\Delta}t}+\\dot{m}\\sum_{t=0}^{T}e^{-\\lambda(T-t)}`\\n
        :math:`I_{end} = Ie^{-{\\lambda}{\\Delta}t}+\\dot{m}\\dfrac{e^{-{\\lambda}T}\\big(e^{{\\lambda}(T+1)}-1\\big)}{e^{\\lambda}-1}`
        """  # noqa: W505, E501
        time_y = 0.50  # years
        n = 100000  # intersteps

        t = np.linspace(0, time_y, n)

        inventory = 150  # KG
        m_in = 3.0  # kg/year
        m_int = m_in * time_y / len(t)
        i, ii = np.zeros(n), np.zeros(n)
        eta = 1
        i[0], ii[0] = inventory, inventory
        for j in range(1, n):
            mint = m_int
            # if j == n:
            #    mint=0
            i[j] = i[j - 1] * np.exp(-T_LAMBDA * (t[j] - t[j - 1])) + mint
            ii[j] = ii[j - 1] * np.exp(-T_LAMBDA * (t[j] - t[j - 1]))

        iend = inventory * np.exp(-T_LAMBDA * time_y)
        iend2 = _dec_I_mdot(inventory, eta, m_in, 0, time_y)
        cross = 80
        dt15 = _find_t15(inventory, eta, m_in, 0, time_y, cross)

        _, ax = plt.subplots(figsize=[10, 8])
        ax.plot(t, i, label="Idecay+m_in")
        ax.plot(t, ii, label="Idecay")
        ax.plot(time_y, iend, marker="o", ms=15)
        ax.plot(time_y, iend2, marker="s", ms=15)

        ax.plot([dt15, dt15], [0, inventory])

        ax.plot([0, time_y], [cross, cross])

        ax.plot(dt15, cross, color="r", marker="*", ms=15)
        plt.show()

        # high n required to converge..
        assert np.isclose(iend2, i[-1], rtol=0.01), f"{iend2} != {i[-1]}"
        assert np.isclose(iend, ii[-1], rtol=0.0001), f"{iend} != {ii[-1]}"
        idx = np.argmin(abs(i - cross))
        assert np.isclose(dt15, t[idx], rtol=0.03), f"{dt15} != {t[idx]}"

    def f(self, m_flow, inventory, fs):
        return _fountain_linear_sink(
            m_flow,
            self.t_in,
            self.t_out,
            inventory,
            fs,
            self.I_max,
            self.I_min,
            0.0,
            0.0,
        )[:2][::-1]

    def test_tlossft(self):
        def _build():
            _, ax = plt.subplots(figsize=[10, 8])
            ax.plot(
                [self.t_in - 0.5, self.t_out + 0.5],
                [self.I_min, self.I_min],
                ls="--",
                color="r",
                lw=2,
            )
            ax.plot(
                [self.t_in - 0.5, self.t_out + 0.5],
                [self.I_max, self.I_max],
                ls="--",
                color="r",
                lw=2,
            )
            ax.plot([self.t_in, self.t_in], [0, 6], ls="--", color="k", lw=1)
            ax.plot([self.t_out, self.t_out], [0, 6], ls="--", color="k", lw=1)
            # ax.set_ylim([1, 6])
            return ax

        def plot(ax, i_new, label=None):
            ax.plot([self.t_in, self.t_out], [inventory, i_new], marker="o", label=label)

        def plotter():
            ax = _build()
            plot(ax, i2, label="little")
            plot(ax, i22, label="decay only")
            plot(ax, i23, label="into")
            plot(ax, i24, label="over")
            plot(ax, i25, label="kill")
            if "Id" in locals():
                plot(ax, id, label="pure decay")
            ax.legend()

        def checker():
            m_in2 = m_flow * (self.t_out - self.t_in) * 365 * 24 * 3600  # kg
            m_in22 = 0
            m_in23 = m_flow_into * (self.t_out - self.t_in) * 365 * 24 * 3600  # kg
            m_in24 = m_flow_over * (self.t_out - self.t_in) * 365 * 24 * 3600  # kg
            m_in25 = m_flow_kill * (self.t_out - self.t_in) * 365 * 24 * 3600  # kg
            assert m2 < m_in2
            assert m22 == m_in22
            assert m23 < m_in23
            assert m24 < m_in24
            assert m25 < m_in25

        def setter():
            i2, m2 = self.f(m_flow, inventory, eta)
            i22, m22 = self.f(0, inventory, eta)
            i23, m23 = self.f(m_flow_into, inventory, eta)
            i24, m24 = self.f(m_flow_over, inventory, eta)
            i25, m25 = self.f(m_flow_kill, inventory, eta)
            return (i2, m2), (i22, m22), (i23, m23), (i24, m24), (i25, m25)

        # For when you're in the tub
        m_flow = 1e-8  # kg/s --> +0.62 kg
        m_flow_into = 4e-8  # kg/s --> +2.83 kg
        m_flow_over = 5e-5  # kg/s --> + loads but into uncanny with eta = 0.9995
        m_flow_kill = 1e3  # kg/s --> + loads kg (used to push up to Imax)
        eta = 0.9995  # --> typical eta value crossing into uncanny valley
        # results in I_min
        inventory = 2.0

        (i2, m2), (i22, m22), (i23, m23), (i24, m24), (i25, m25) = setter()
        checker()
        plotter()

        # for when you're in the valley
        m_flow = 5e-6  # kg/s --> +0.62 kg
        m_flow_into = 1e-5  # kg/s --> +2.83 kg
        # results in I_min
        inventory = 3.5

        (i2, m2), (i22, m22), (i23, m23), (i24, m24), (i25, m25) = setter()
        checker()
        plotter()

        # for when you want to cross down into the shadow
        # results in I_min
        inventory = 3.05

        (i2, m2), (i22, m22), (i23, m23), (i24, m24), (i25, m25) = setter()
        checker()
        plotter()

        # for when you want to cross down into the shadow
        # results in I_min
        inventory = 4.7

        (i2, m2), (i22, m22), (i23, m23), (i24, m24), (i25, m25) = setter()
        checker()
        plotter()

        # for when you want to test absurdity
        m_flow = 1e-9  # kg/s --> push down into I_min
        # results in I_min

        inventory = 5.0
        self.t_out = 30

        (i2, m2), (i22, m22), (i23, m23), (i24, m24), (i25, m25) = setter()
        checker()
        plotter()
