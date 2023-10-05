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
Update copyright on files
"""
import fileinput
from argparse import ArgumentParser
from datetime import date
from sys import exit
from typing import List


def arguments():
    """
    Parse arguments for copyright script
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        metavar="files",
        type=str,
        default=[],
        nargs="*",
        help="python files to be updated",
    )
    args = parser.parse_args()
    return args.files


def edit_files(files: List[str], copyright_line: str):
    """
    Edit files in place
    """
    for file in files:
        with fileinput.FileInput(file, inplace=True) as fh:
            for line in fh:
                if line.startswith(r"# Copyright") and copyright_line not in line:
                    print(line.replace(line.split(" ", 4)[-2], copyright_line), end="")
                    yield file
                elif (
                    "conf.py" in file
                    and line.startswith(r"copyright")
                    and copyright_line not in line
                ):
                    print(
                        line.replace(line.split(" ", 4)[-2], f'"{copyright_line},'),
                        end="",
                    )
                else:
                    print(line, end="")


def main():
    """Run the copyright updater"""
    edited = list(edit_files(arguments(), rf"2021-{date.today().year}"))  # noqa: DTZ011

    if edited != []:
        for file in edited:
            print(f"Updated {file}")
        exit(1)


main()
