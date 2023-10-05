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
Parameter classes/structures for Process
"""

from dataclasses import dataclass, field, fields
from typing import Dict, Generator, List, Tuple, Union

from bluemira.codes.process.api import _INVariable


@dataclass
class ProcessInputs:
    """
    Process Inputs dataclass

    Notes
    -----
    All entries get wrapped in an INVariable class to enable easy InDat writing.

    Units for these are available in bluemira.codes.process.mapping for mapped
    variables otherwise
    `process.io.python_fortran_dicts.get_dicts()["DICT_DESCRIPTIONS"]`
    """

    bounds: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: {
            "2": {"u": "20.0"},
            "3": {"u": "13"},
            "4": {"u": "150.0"},
            "9": {"u": "1.2"},
            "18": {"l": "3.5"},
            "29": {"l": "0.1"},
            "38": {"u": "1.0"},
            "39": {"u": "1.0"},
            "42": {"l": "0.05", "u": "0.1"},
            "50": {"u": "1.0"},
            "52": {"u": "10.0"},
            "61": {"l": "0.02"},
            "103": {"u": "10.0"},
            "60": {"l": "6.0e4", "u": "9.0e4"},
            "59": {"l": "0.50", "u": "0.94"},
        }
    )
    # fmt: off
    icc: List[int] = field(default_factory=lambda: [1, 2, 5, 8, 11, 13, 15, 16, 24, 25,
                                                    26, 27, 30, 31, 32, 33, 34, 35, 36,
                                                    60, 62, 65, 68, 72])
    ixc: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 9, 13, 14, 16, 18,
                                                    29, 36, 37, 38, 39, 41, 42, 44, 48,
                                                    49, 50, 51, 52, 53, 54, 56, 57, 58,
                                                    59, 60, 61, 102, 103, 106, 109, 110,
                                                    113, 117, 122, 123])
    # fmt: on
    abktflnc: float = 15.0
    adivflnc: float = 20.0
    alphan: float = 1.0
    alphat: float = 1.45
    alstroh: float = 660000000.0
    aspect: float = 3.1
    beta: float = 0.031421
    blnkith: float = 0.755
    blnkoth: float = 0.982
    bmxlim: float = 11.2
    bore: float = 2.3322
    bscfmax: float = 0.99
    bt: float = 5.3292
    casths: float = 0.05
    cfactr: float = 0.75
    coheof: float = 20726000.0
    coreradiationfraction: float = 0.6
    coreradius: float = 0.75
    cost_model: int = 0
    cptdin: List[float] = field(
        default_factory=lambda: [*([42200.0] * 4), *([43000.0] * 4)]
    )
    cpttf: float = 65000.0
    d_vv_bot: float = 0.6
    d_vv_in: float = 0.6
    d_vv_out: float = 1.1
    d_vv_top: float = 0.6
    ddwex: float = 0.15
    dene: float = 7.4321e19
    dhecoil: float = 0.01
    dintrt: float = 0.0
    discount_rate: float = 0.06
    divdum: int = 1
    divfix: float = 0.621
    dnbeta: float = 3.0
    dr_tf_case_in: float = 0.52465
    dr_tf_case_out: float = 0.06
    emult: float = 1.35
    enbeam: float = 1e3
    epsvmc: float = 1e-08
    etaech: float = 0.4
    etahtp: float = 0.87
    etaiso: float = 0.9
    etanbi: float = 0.3
    etath: float = 0.375
    fbetatry: float = 0.48251
    fcap0: float = 1.15
    fcap0cp: float = 1.06
    fcohbop: float = 0.93176
    fcontng: float = 0.15
    fcr0: float = 0.065
    fcuohsu: float = 0.7
    fcutfsu: float = 0.80884
    fdene: float = 1.2
    ffuspow: float = 1.0
    fgwped: float = 0.85
    fimp: List[float] = field(
        default_factory=lambda: [1.0, 0.1, *([0.0] * 10), 0.00044, 5e-05]
    )
    fimpvar: float = 0.00037786
    fiooic: float = 0.63437
    fjohc0: float = 0.53923
    fjohc: float = 0.57941
    fjprot: float = 1.0
    fkind: float = 1.0
    fkzohm: float = 1.0245
    flhthresh: float = 1.4972
    foh_stress: float = 1.0
    fpeakb: float = 1.0
    fpinj: float = 1.0
    fpnetel: float = 1.0
    fpsepbqar: float = 1.0
    fstrcase: float = 1.0
    fstrcond: float = 0.92007
    ftaucq: float = 0.91874
    ftaulimit: float = 1.0
    ftburn: float = 1.0
    ftmargoh: float = 1.0
    ftmargtf: float = 1.0
    fvdump: float = 1.0
    fvsbrnni: float = 0.39566
    fwalld: float = 0.131
    gamma: float = 0.3
    gamma_ecrh: float = 0.3
    gapds: float = 0.02
    gapoh: float = 0.05
    gapomin: float = 0.2
    hfact: float = 1.1
    hldivlim: float = 10.0
    i_single_null: int = 1
    i_tf_sc_mat: int = 5
    i_tf_turns_integer: int = 1
    iavail: int = 0
    ibss: int = 4
    iculbl: int = 1
    icurr: int = 4
    idensl: int = 7
    iefrf: int = 10
    ieped: int = 1
    ifalphap: int = 1
    ifispact: int = 0
    ifueltyp: int = 1
    iinvqd: int = 1
    impvar: int = 13
    inuclear: int = 1
    iohcl: int = 1
    ioptimz: int = 1
    ipedestal: int = 1
    ipfloc: List[int] = field(default_factory=lambda: [2, 2, 3, 3])
    ipowerflow: int = 0
    iprimshld: int = 1
    iprofile: int = 1
    isc: int = 34
    ishape: int = 0
    isumatoh: int = 5
    isumatpf: int = 3
    kappa: float = 1.848
    ksic: float = 1.4
    lpulse: int = 1
    lsa: int = 2
    minmax: int = 1
    n_layer: int = 10
    n_pancake: int = 20
    n_tf: int = 16
    ncls: List[int] = field(default_factory=lambda: [1, 1, 2, 2])
    neped: float = 6.78e19
    nesep: float = 2e19
    ngrp: int = 4
    oacdcp: float = 8673900.0
    oh_steel_frac: float = 0.57875
    ohcth: float = 0.55242
    ohhghf: float = 0.9
    output_costs: int = 0
    pheat: float = 50.0
    pinjalw: float = 51.0
    plasma_res_factor: float = 0.66
    pnetelin: float = 500.0
    primary_pumping: int = 3
    prn1: float = 0.4
    psepbqarmax: float = 9.2
    pulsetimings: float = 0.0
    q0: float = 1.0
    q: float = 3.5
    qnuc: float = 12920.0
    ralpne: float = 0.06894
    rhopedn: float = 0.94
    rhopedt: float = 0.94
    ripmax: float = 0.6
    rjconpf: List[float] = field(
        default_factory=lambda: [1.1e7, 1.1e7, 6e6, 6e6, 8e6, 8e6, 8e6, 8e6]
    )
    rmajor: float = 8.8901
    rpf2: float = -1.825
    scrapli: float = 0.225
    scraplo: float = 0.225
    secondary_cycle: int = 2
    shldith: float = 1e-06
    shldlth: float = 1e-06
    shldoth: float = 1e-06
    shldtth: float = 1e-06
    sig_tf_case_max: float = 580000000.0
    sig_tf_wp_max: float = 580000000.0
    ssync: float = 0.6
    tbeta: float = 2.0
    tbrnmn: float = 7200.0
    tburn: float = 10000.0
    tdmptf: float = 25.829
    tdwell: float = 0.0
    te: float = 12.33
    teped: float = 5.5
    tesep: float = 0.1
    tfcth: float = 1.208
    tftmp: float = 4.75
    tftsgap: float = 0.05
    thicndut: float = 0.002
    thshield: float = 0
    thwcndut: float = 0.008
    tinstf: float = 0.008
    tlife: float = 40.0
    tmargmin: float = 1.5
    tramp: float = 500.0
    triang: float = 0.5
    ucblvd: float = 280.0
    ucdiv: float = 500000.0
    ucme: float = 300000000.0
    vdalw: float = 10.0
    vfshld: float = 0.6
    vftf: float = 0.3
    vgap2: float = 0.05
    vvblgap: float = 0.02
    walalw: float = 8.0
    zeffdiv: float = 3.5
    zref: List[float] = field(
        default_factory=lambda: [3.6, 1.2, 1.0, 2.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )

    def __iter__(self) -> Generator[Tuple[str, Union[float, List, Dict]], None, None]:
        """
        Iterate over this dataclass

        The order is based on the order in which the values were
        declared.
        """
        for _field in fields(self):
            yield _field.name, getattr(self, _field.name)

    def to_invariable(self) -> Dict[str, _INVariable]:
        """
        Wrap each value in an INVariable object

        Needed for compatibility with PROCESS InDat writer
        """
        out_dict = {}
        for name, value in self:
            if name not in ["icc", "ixc", "bounds"]:
                new_val = _INVariable(name, value, "Parameter", "", "")
                out_dict[name] = new_val
        out_dict["icc"] = _INVariable(
            "icc",
            self.icc,
            "Constraint Equation",
            "Constraint Equation",
            "Constraint Equations",
        )
        out_dict["ixc"] = _INVariable(
            "ixc",
            self.ixc,
            "Iteration Variable",
            "Iteration Variable",
            "Iteration Variables",
        )
        out_dict["bounds"] = _INVariable(
            "bounds", self.bounds, "Bound", "Bound", "Bounds"
        )
        return out_dict

    def to_dict(self) -> Dict[str, Union[float, List, Dict]]:
        """
        A dictionary representation of the dataclass

        """
        return dict(self)
