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

import copy
import json
from pathlib import Path
from unittest import mock

import fortranformat as ff
import numpy as np
import pytest
from typeguard import TypeCheckError, check_type

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.file import EQDSKInterface
from bluemira.utilities.tools import compare_dicts
from tests._helpers import combine_text_mock_write_calls

OPEN = "builtins.open"


class TestEQDSKInterface:
    path = get_bluemira_path("equilibria/test_data", subfolder="tests")
    testfiles = (
        Path(get_bluemira_path("eqdsk", subfolder="data"), "jetto.eqdsk_out"),
        Path(path, "DN-DEMO_eqref.json"),
        Path(path, "eqref_OOB.json"),
    )

    @classmethod
    def setup_class(cls):
        data_dir = get_bluemira_path("equilibria", subfolder="data")
        data_file = Path(data_dir, "DN-DEMO_eqref.json")
        with open(data_file) as f:
            cls.eudemo_sof_data = json.load(f)

    def read_strict_geqdsk(self, file_path):
        """
        Reads an input EQDSK file in, assuming strict adherence to the
        GEQDSK format. Used to check bluemira outputs can be read by
        external readers.

        Note: The main bluemira GEQDSK reader is more forgiving to
        format variations than this!

        Parameters
        ----------
        file_path: str
            Full path string of the file

        """

        # Create FortranRecordReader objects with the Fortran format
        # edit descriptors to be used to parse the G-EQDSK input.
        f2000 = ff.FortranRecordReader("a48,3i4")
        f2020 = ff.FortranRecordReader("5e16.9")
        f2022 = ff.FortranRecordReader("2i5")
        fCSTM = ff.FortranRecordReader("i5")

        # Define helper function to read in flattened arrays.
        def read_flat_array(fortran_format, array_size):
            """
            Reads in a flat (1D) numpy array from a G-EQDSK file.

            Parameters
            ----------
            fortran_format: ff.FortranRecordReader
                FortranRecordReader object for Fortran format edit descriptor
                to be used to parse the format of each line of the output.
            array_size: int
                Number of elements in array to be read in.

            Returns
            -------
            array: np.array
                1D Numpy array of length array_size populated by elements from
                the GEQDSK file input.
            """

            # Initialise numpy array and read in first line.
            array = np.zeros((array_size,))
            line = fortran_format.read(file.readline())
            # Define a counter to track which column in a line
            # is currently being saved into the array.
            col = 0
            # Populate array. If the column index moves past the
            # end of a line, it is reset to zero and the next line is read.
            for i in range(array_size):
                if col == len(line):
                    line = fortran_format.read(file.readline())
                    col = 0
                array[i] = line[col]
                col += 1
            return array

        # Open file.
        with open(file_path) as file:
            # Read in data. Variable names are for readability;
            # strict format is as defined at
            # https://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf
            _id, _, nx, nz = f2000.read(file.readline())
            xdim, zdim, xcentre, xgrid1, zmid = f2020.read(file.readline())
            xmag, zmag, psimag, psibdry, bcentre = f2020.read(file.readline())
            cplasma, psimag, _, xmag, _ = f2020.read(file.readline())
            zmag, _, psibdry, _, _ = f2020.read(file.readline())
            fpol = read_flat_array(f2020, nx)
            pressure = read_flat_array(f2020, nx)
            ffprime = read_flat_array(f2020, nx)
            pprime = read_flat_array(f2020, nx)
            psi = read_flat_array(f2020, nx * nz)
            qpsi = read_flat_array(f2020, nx)
            nbdry, nlim = f2022.read(file.readline())
            xbdry_zbdry = read_flat_array(f2020, 2 * nbdry)
            xlim_zlim = read_flat_array(f2020, 2 * nlim)

            # Read in coil information, as found in the GEQDSK extension
            # used by bluemira.
            (ncoil,) = fCSTM.read(file.readline())
            coil = read_flat_array(f2020, 5 * ncoil)

    @pytest.mark.parametrize("file", testfiles)
    def test_read(self, file):
        # Create EQDSK file interface and read data to a dict
        eqdsk = EQDSKInterface.from_file(file)
        d1 = eqdsk.to_dict()

        # Write data read in from test file into a new EQDSK
        # file, with the suffix "_temp"
        name = Path(file).stem + "_temp"
        fname = Path(self.path, f"{name}.eqdsk")
        eqdsk.write(fname, file_format="eqdsk")
        d2 = eqdsk.to_dict()

        # Check eqdsk is readable by Fortran readers.
        # This demands stricter adherence to the G-EQDSK
        # format than bluemira's main reader.
        self.read_strict_geqdsk(fname)

        # Write data read in from test file into a new JSON
        # file, with the suffix "_temp"
        jname = fname.with_suffix("").with_suffix(".json")
        eqdsk.write(jname, file_format="json")
        d3 = EQDSKInterface.from_file(jname).to_dict()

        # Clean up temporary files
        fname.unlink()
        jname.unlink()

        # Compare dictionaries to check data hasn't
        # been changed.
        assert compare_dicts(d1, d2, verbose=True)
        assert compare_dicts(d1, d3, verbose=True)
        assert compare_dicts(d2, d3, verbose=True)

    def test_read_matches_values_in_file(self):
        eq = EQDSKInterface.from_file(self.testfiles[0])

        assert eq.nz == 151
        assert eq.nx == 151
        assert eq.xdim == pytest.approx(3.14981545)
        assert eq.ncoil == 0
        assert eq.xc.size == 0
        assert eq.nbdry == 72
        np.testing.assert_allclose(
            eq.xbdry[:3], [0.399993127e01, 0.399150254e01, 0.396906908e01]
        )
        np.testing.assert_allclose(
            eq.zbdry[-3:], [-0.507187454e00, -0.240712636e00, 0.263892047e-01]
        )

    def test_values_match_annotated_types(self):
        eq = EQDSKInterface.from_file(self.testfiles[0])

        mismatched = []
        for key, value_type in EQDSKInterface.__annotations__.items():
            value = getattr(eq, key)
            try:
                check_type(value, value_type)
            except TypeCheckError:
                mismatched.append((key, type(value), value_type))
        assert not mismatched

    def test_write_then_read_in_json_format(self):
        eq = EQDSKInterface.from_file(self.testfiles[0])

        with mock.patch(OPEN, new_callable=mock.mock_open) as open_mock:
            eq.write("some/path.json", file_format="json")
        written = combine_text_mock_write_calls(open_mock)

        with mock.patch(
            OPEN, new_callable=mock.mock_open, read_data=written
        ) as open_mock:
            eq2 = EQDSKInterface.from_file("/some/path.json")

        assert eq2.nz == 151
        assert eq2.nbdry == 72

    def test_derived_field_is_calculated_if_not_given(self):
        data = copy.deepcopy(self.eudemo_sof_data)
        for field in ["x", "z", "psinorm"]:
            del data[field]

        open_mock = mock.mock_open(read_data=json.dumps(data))
        with mock.patch(OPEN, new=open_mock):
            eqdsk = EQDSKInterface.from_file("/some/file.json")

        np.testing.assert_allclose(eqdsk.x, self.eudemo_sof_data["x"])
        np.testing.assert_allclose(eqdsk.z, self.eudemo_sof_data["z"])
        # The calculation used for psinorm has changed since the
        # eudemo_sof_data was created - so we can't compare to that in
        # this case.
        np.testing.assert_allclose(
            eqdsk.psinorm, np.linspace(0, 1, len(self.eudemo_sof_data["fpol"]))
        )
