# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
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

    data = deepcopy(mfile_data())

    def __init__(self, filename, name="PROCESS"):
        self.filename = filename
        self._name = name

    @classmethod
    def reset_data(cls):
        cls.data = deepcopy(mfile_data())


def mfw(*, radial_override: bool = True):
    from bluemira.codes.process._teardown import _MFileWrapper  # noqa: PLC0415

    class MFW(_MFileWrapper):
        # Overwrite some methods because data doesnt exist in 'mfile'

        def __init__(self, path, name):
            self.mfile = FakeMFile(path, name)
            self._name = name

        if radial_override:

            def _derive_radial_build_params(self, data):  # noqa: ARG002
                return {}

    return MFW
