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
import functools
import json
from copy import deepcopy
from pathlib import Path

DATA_DIR = Path(Path(__file__).parent, "test_data").as_posix()
READ_DIR = Path(DATA_DIR, "read").as_posix()
RUN_DIR = Path(DATA_DIR, "run").as_posix()
PARAM_FILE = Path(DATA_DIR, "params.json").as_posix()
FAKE_PROCESS_DICT = {  # Fake for the output of PROCESS's `get_dicts()`
    "DICT_DESCRIPTIONS": {"some_property": "its description"}
}

# Ugly workaround here: we want to be able to set '.data' on the
# FakeMFile class within tests, but then we don't want those changes
# propagating to subsequent tests. My workaround is to have a function
# that loads/caches the original MFile data, then copy that data onto
# the FakeMFile class. We then have a 'reset_data' class method that
# can be run in a test class's 'teardown_method' to set the data back
# to its original state. By caching the data within 'mfile_data', we
# only load the file once.


@functools.lru_cache
def mfile_data():
    """Load and cache MFile data stored in the JSON file."""
    with open(Path(DATA_DIR, "mfile_data.json")) as f:
        return json.load(f)


class FakeMFile:
    """
    A fake of PROCESS's MFile class.

    It replicates the :code:`.data` attribute with some PROCESS result's
    data. This allows us to test the logic in our API without having
    PROCESS installed.
    """

    data = mfile_data()

    def __init__(self, filename, name="PROCESS"):
        self.filename = filename
        self._name = name

    @classmethod
    def reset_data(cls):
        cls.data = deepcopy(mfile_data())
