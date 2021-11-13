# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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
The bluemira external code wrapper
"""
from enum import Enum, auto


class RunMode(Enum):
    """Defines the possible runmode"""
    PROMINENCE = auto()
    BATCH = auto()
    MOCK = auto()

    def __call__(self, obj, *args, **kwargs):
        """
        Call function of object with lowercase name of
        enum
        Parameters
        ----------
        obj: instance
            instance of class the function will come from. If object is not specified,
            self will be used.
        *args
           args of function
        **kwargs
           kwargs of function
        Returns
        -------
        function result
        """
        func = getattr(obj, f"_{self.name.lower()}")
        return func(*args, **kwargs)


class Task():
    """
    A class for any task integration
    """
    # todo: ensure a correspondence between the specified runmode and the implemented
    #  functions (if possible).

    def _prominence(self, *args, **kwargs):
        raise NotImplementedError

    def _batch(self, *args, **kwargs):
        raise NotImplementedError

    def _mock(self, *args, **kwargs):
        raise NotImplementedError


class ExternalCode():
    """An external code wrapper"""
    def __init__(self, runmode, *args, **kwargs):
        self.runmode = None
        self.set_runmode(runmode)
        self.setup_obj = self.Setup(self, *args, **kwargs)
        self.run_obj = self.Run(self, *args, **kwargs)
        self.teardown_obj = self.Teardown(self, *args, **kwargs)

    def set_runmode(self, runmode):
        """Set the runmode"""
        self.runmode = RunMode[runmode]

    def run(self):
        self.runmode(self.setup_obj)
        self.runmode(self.run_obj)
        self.runmode(self.teardown_obj)

    class Setup(Task):
        """A class that specified the code setup"""
        def __init__(self, outer, *args, **kwargs):
            self.outer = outer

    class Run(Task):
        """A class that specified the code run process"""
        def __init__(self, outer, *args, **kwargs):
            self.outer = outer

    class Teardown(Task):
        """A class that for the teardown"""
        def __init__(self, outer, *args, **kwargs):
            self.outer = outer