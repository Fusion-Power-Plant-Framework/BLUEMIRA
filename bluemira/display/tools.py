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
tools for display module
"""
import pprint
from dataclasses import asdict


class Options:
    """
    The options that are available for displaying objects.
    """

    __slots__ = ("_options",)

    def __init__(self, **kwargs):
        self.modify(**kwargs)

    def __setattr__(self, attr, val):
        """
        Set attributes in options dictionary
        """
        if getattr(self, "_options", None) is not None and (
            attr in self._options.__annotations__ or hasattr(self._options, attr)
        ):
            setattr(self._options, attr, val)
        else:
            super().__setattr__(attr, val)

    def __getattribute__(self, attr):
        """
        Get attributes or from "_options" dict
        """
        try:
            return super().__getattribute__(attr)
        except AttributeError as ae:
            if attr != "_options":
                try:
                    return getattr(self._options, attr)
                except AttributeError:
                    raise ae from None
            raise

    def modify(self, **kwargs):
        """Modify options"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def as_dict(self):
        """
        Returns the instance as a dictionary.
        """
        return asdict(self._options)

    def __repr__(self):
        """
        Representation string of the DisplayOptions.
        """
        return f"{type(self).__name__}({pprint.pformat(self.as_dict())}\n)"
