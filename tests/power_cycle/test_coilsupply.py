# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later


from bluemira.power_cycle.coilsupply import (
    CoilSupplyCorrector,
    CoilSupplyCorrectorConfig,
    CoilVariable,
    ThyristorBridges,
    ThyristorBridgesConfig,
)


class TestCoilVariable:
    def test_from_str():
        assert CoilVariable.from_str("voltage") == CoilVariable.ACTIVE
        assert CoilVariable.from_str("current") == CoilVariable.REACTIVE


class TestCoilSupplyCorrector:
    def test_init():
        name = "name of corrector"
        description = "description of corrector"
        variable = "voltage"
        factor = -0.5
        converter_config = CoilSupplyCorrectorConfig(
            {
                "name": name,
                "description": description,
                "correction_variable": variable,
                "correction_factor": factor,
            }
        )
        converter = CoilSupplyCorrector(converter_config)
        assert converter.name == name
        assert converter.description == description
        assert converter.correction_variable == CoilVariable.from_str(variable)
        assert converter.correction_factor == factor

    def test_correct():
        pass

    def test_compute_correction():
        pass


class TestThyristorBridges:
    def test_init():
        name = "name of thyristor bridge"
        description = "description of thyristor bridge"
        max_bridge_voltage = 10
        power_loss_percentages = {
            "losses 1": 1.5,
            "losses 2": 2,
            "losses 3": 0.5,
        }
        bridge_config = ThyristorBridgesConfig(
            {
                "name": name,
                "description": description,
                "max_bridge_voltagee": max_bridge_voltage,
                "power_loss_percentages": power_loss_percentages,
            }
        )
        bridges = ThyristorBridges(bridge_config)
        assert bridges.name == name
        assert bridges.description == description
        assert bridges.max_bridge_voltage == max_bridge_voltage
        assert bridges.power_loss_percentages == power_loss_percentages

    def test_compute_conversion():
        pass


class TestCoilSupplySystem:
    pass
