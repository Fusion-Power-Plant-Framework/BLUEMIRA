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
A helper script to generate ParameterFrames as a python file and json file
"""

import argparse
import inspect
import os
import sys
from abc import abstractproperty
from copy import deepcopy
from pathlib import Path
from pkgutil import iter_modules
from typing import Dict, Optional, Set

from setuptools import find_packages

from bluemira.base.logs import set_log_level
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print, bluemira_warn
from bluemira.base.parameter_frame import ParameterFrame
from bluemira.base.parameter_frame._parameter import ParamDictT
from bluemira.utilities.tools import get_module, json_writer


def def_param() -> Dict:
    """
    Get the defualt parameter json skeleton
    """
    dp = deepcopy(ParamDictT.__annotations__)
    del dp["name"]
    dp["description"] = dp["long_name"] = ""
    dp["unit"] = dp["source"] = "str"
    return dp


DEFAULT_PARAM = def_param()


def add_to_dict(pf: ParameterFrame, json_dict: Dict, params: Dict):
    """
    Add each parameter to the json dict and params dict
    """
    for param, param_type in pf.__annotations__.items():
        dv = deepcopy(DEFAULT_PARAM)
        dv["value"] = str(param_type).split("[")[-1][:-1]
        json_dict[param] = dv
        params[param] = param_type


def create_parameterframe(
    params: Dict, name: Optional[str] = None, *, header: bool = True
) -> str:
    """
    Create parameterframe python files as a string

    Parameters
    ----------
    params: Dict
        Dictionary of parameters to add to ParameterFrame
    name: Optional[str]
        name of ParameterFrame
    header: bool
        add import header

    """
    param_cls = (
        (
            "from dataclasses import dataclass\n\n"
            "from bluemira.base.parameterframe import Parameter, ParameterFrame\n\n\n"
        )
        if header
        else ""
    )

    param_cls += "@dataclass\nclass {}(ParameterFrame):\n"

    if name is None:
        param_cls = param_cls.format("ReactorParams")
    else:
        param_cls = param_cls.format(name)

    param_row = "    {name}: {_type}\n"

    for param, param_type in params.items():
        param_cls += param_row.format(name=param, _type=str(param_type).split(".")[-1])

    return param_cls


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate ParameterFrame files from module or package"
    )
    parser.add_argument("module", type=str, help="Module or Package to search through")
    parser.add_argument(
        "-c", "--collapse", action="store_true", help="Collapse to one ParameterFrame"
    )
    parser.add_argument(
        "-d", "--save-directory", dest="directory", type=str, default="./"
    )
    parser.add_argument(
        "-v",
        action="count",
        default=0,
        help="Increase logging severity level.",
    )
    parser.add_argument(
        "-q",
        action="count",
        default=0,
        help="Decrease logging severity level.",
    )

    args = parser.parse_args()
    args.module = str(Path(args.module).resolve())
    args.directory = str(Path(args.directory).resolve())

    set_log_level(min(max(0, 2 + args.q - args.v), 5))
    return args


def get_param_classes(module) -> Dict:
    """
    Get all ParameterFrame classes
    """
    return {
        f"{m[0]}: {m[1].param_cls.__name__}": m[1].param_cls
        for m in inspect.getmembers(module, inspect.isclass)
        if hasattr(m[1], "param_cls")
        and not isinstance(m[1].param_cls, (type(None), abstractproperty))
    }


def find_modules(path: str) -> Set:
    """Recursively get modules from package"""
    modules = set()
    for pkg in find_packages(path):
        if "test" in pkg:
            bluemira_debug(f"Ignoring {pkg}, possible test package")
            continue
        modules.add(pkg)
        pkgpath = path + "/" + pkg.replace(".", "/")
        for info in iter_modules([pkgpath]):
            if "test" in info.name:
                bluemira_debug(f"Ignoring {info.name}, possible test module")
            elif not info.ispkg:
                modules.add(f"{pkg}.{info.name}")
    bluemira_print("Found modules:\n" + "\n".join(sorted(m for m in modules)))
    if not os.path.commonprefix(list(modules)):
        bluemira_warn(
            "Not all modules come from the same package."
            " Is your module path one level too deep?"
        )

    return modules


def main():
    """
    Generate python and json paramterframe files
    """
    args = parse_args()

    param_classes = {}

    sys.path.insert(0, args.module)
    mods = find_modules(args.module)
    if len(mods) > 0:
        for mod in mods:
            path = Path(args.module, mod.replace(".", "/"))
            if not path.is_dir():
                module = get_module(f"{path}.py")
                param_classes.update(get_param_classes(module))
    else:
        module = get_module(args.module)
        param_classes.update(get_param_classes(module))

    bluemira_print(
        "Found ParameterFrames:\n" + "\n".join(sorted(k for k in param_classes.keys()))
    )

    output = {}
    params = {}

    bluemira_print(f"Writing output files to {args.directory}")
    if args.collapse:
        for vv in param_classes.values():
            add_to_dict(vv, output, params)

        with open(Path(args.directory, "params.py"), "w") as fh:
            fh.write(create_parameterframe(params))
        json_writer(output, file=Path(args.directory, "params.json"))

    else:
        for kk, vv in param_classes.items():
            output[kk] = {}
            params[kk] = {}
            add_to_dict(vv, output[kk], params[kk])

        with open(Path(args.directory, "params.py"), "w") as fh:
            header = True
            for out_name, out_val in output.items():
                pname = out_name.replace(" ", "")
                pname = pname.replace(":", "_")
                fh.write(create_parameterframe(params[out_name], pname, header=header))
                fh.write("\n")
                json_writer(out_val, file=Path(args.directory, f"{pname}.json"))
                header = False
    bluemira_print("Done")


if __name__ == "__main__":
    main()
