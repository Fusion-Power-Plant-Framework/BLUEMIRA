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
Dummy external code
"""

import os
from argparse import ArgumentParser


def get_filename():
    """Get filename of script"""
    return os.path.realpath(__file__)


def parse_args():
    """
    Parse arguments for copyright script
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--add-header",
        action="store_true",
        default=False,
        help="add header to file",
    )
    parser.add_argument(
        "--number", action="store_true", default=False, help="add line numbers"
    )
    parser.add_argument("infile", type=str)
    parser.add_argument("outfile", type=str)
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    with open(args.infile) as _if:
        lines = _if.readlines()

    with open(args.outfile, "w") as of:
        if args.add_header:
            of.write("# this is a header\n")
        for no, line in enumerate(lines, start=1):
            of.write(f"{no if args.number else ''}    {line}")


if __name__ == "__main__":
    main()
