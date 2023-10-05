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
Input and output file interface. EQDSK and json. NOTE: jsons are better :)
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import fortranformat as ff
import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.utilities.tools import is_num, json_writer

EQDSK_EXTENSIONS = [".eqdsk", ".eqdsk_out", ".geqdsk"]


@dataclass(repr=False)
class EQDSKInterface:
    """
    Container for data from an EQDSK file.

    Inspired by an EQDSK reader originally developed by B. Dudson:
        https://github.com/bendudson/pyTokamak/blob/master/tokamak/formats/geqdsk.py

    The G-EQDSK file format is described here:
        https://fusion.gat.com/conferences/snowmass/working/mfe/physics/p3/equilibria/g_eqdsk_s.pdf

    Notes
    -----
    G-EQDSK is from the 1980's and EQDSK files should generally only be
    read and not written. New equilibria should really just be saved as
    JSON files.

    Poloidal magnetic flux units not enforced here!

    Plasma current direction is not enforced here!
    """

    bcentre: float
    """Magnetic field at the reference radius [T]."""
    cplasma: float
    """Plasma current [A]."""
    dxc: np.ndarray
    """X half-thicknesses of the coils [m]."""
    dzc: np.ndarray
    """Z half-thicknesses of the coils [m]."""
    ffprime: np.ndarray
    """FF' function on 1-D flux grid [m.T^2/V.s/rad]."""
    fpol: np.ndarray
    """Poloidal current function f = R*B on 1-D flux [T.m]."""
    Ic: np.ndarray
    """Coil currents [A]."""
    name: str
    """Name of the equilibrium EQDSK [dimensionless]."""
    nbdry: int
    """Number of boundary points [dimensionless]."""
    ncoil: int
    """Number of coils [dimensionless]."""
    nlim: int
    """Number of limiters [dimensionless]."""
    nx: int
    """Number of grid points in the radial direction [dimensionless]."""
    nz: int
    """Number of grid points in the vertical direction [dimensionless]."""
    pprime: np.ndarray
    """P' function on 1-D flux grid [N/m^2/V.s/rad]."""
    pressure: np.ndarray
    """Plasma pressure function on 1-D flux grid [N/m^2]."""
    psi: np.ndarray
    """Poloidal magnetic flux on the 2-D grid [V.s/rad]."""
    psibdry: float
    """Poloidal flux at the magnetic axis [V.s/rad]."""
    psimag: float
    """Poloidal flux at the magnetic axis [V.s/rad]."""
    xbdry: np.ndarray
    """X coordinates of the plasma boundary [m]."""
    xc: np.ndarray
    """X coordinates of the coils [m]."""
    xcentre: float
    """Radius of the reference toroidal magnetic  [m]."""
    xdim: float
    """Horizontal dimension of the spatial grid [m]."""
    xgrid1: float
    """Minimum radius of the spatial grid [m]."""
    xlim: np.ndarray
    """X coordinates of the limiters [m]."""
    xmag: float
    """Radius of the magnetic axis [m]."""
    zbdry: np.ndarray
    """Z coordinates of the plasma boundary [m]."""
    zc: np.ndarray
    """Z coordinates of the coils [m]."""
    zdim: float
    """Vertical dimension of the spatial grid [m]."""
    zlim: np.ndarray
    """Z coordinates of the limiters [m]."""
    zmag: float
    """Z coordinate of the magnetic axis [m]."""
    zmid: float
    """Z coordinate of the middle of the spatial grid [m]."""
    x: Optional[np.ndarray] = None
    """X 1-D vector [m] (calculated if not given)."""
    z: Optional[np.ndarray] = None
    """Z 1-D vector [m] (calculated if not given)."""
    psinorm: Optional[np.ndarray] = None
    """Normalised psi vector [A] (calculated if not given)."""
    qpsi: Optional[np.ndarray] = None
    """Safety factor values on the 1-D flux grid [dimensionless]."""
    file_name: Optional[str] = None
    """The EQDSK file the data originates from."""

    def __post_init__(self):
        """Calculate derived parameters if they're not given."""
        if self.x is None:
            self.x = _derive_x(self.xgrid1, self.xdim, self.nx)
        if self.z is None:
            self.z = _derive_z(self.zmid, self.zdim, self.nz)
        if self.psinorm is None:
            self.psinorm = _derive_psinorm(self.fpol)

    @classmethod
    def from_file(cls, file_path: str):
        """
        Create an EQDSKInterface object from a file.

        Parameters
        ----------
        file_path:
            Path to a file of one of the following formats:

                * JSON
                * eqdsk
                * eqdsk_out
                * geqdsk

        Returns
        -------
        An instance of this class containing the EQDSK file's data.
        """
        file_extension = Path(file_path).suffix
        file_name = Path(file_path).name
        if file_extension.lower() in EQDSK_EXTENSIONS:
            return cls(file_name=file_name, **_read_eqdsk(file_path))
        if file_extension.lower() == ".json":
            return cls(file_name=file_name, **_read_json(file_path))
        raise ValueError(f"Unrecognised file format '{file_extension}'.")

    def to_dict(self) -> Dict:
        """Return a dictionary of the EQDSK data."""
        d = asdict(self)
        # Remove the file name as this is metadata, not EQDSK data
        del d["file_name"]
        return d

    def write(
        self,
        file_path: str,
        file_format: str = "json",
        json_kwargs: Optional[Dict] = None,
    ):
        """
        Write the EQDSK data to file in the given format.

        Parameters
        ----------
        file_path:
            Path to where the file should be written.
        file_format:
            The format to save the file in. One of 'json', 'eqdsk', or
            'geqdsk'.
        json_kwargs:
            Key word arguments to pass to the ``json.dump`` call. Only
            used if ``format`` is 'json'.
        """
        if file_format == "json":
            json_kwargs = {} if json_kwargs is None else json_kwargs
            json_writer(self.to_dict(), file_path, **json_kwargs)
        elif file_format in ["eqdsk", "geqdsk"]:
            bluemira_warn(
                "You are in the 21st century. Are you sure you want to be making an"
                " EDQSK in this day and age?"
            )
            _write_eqdsk(Path(file_path).as_posix(), self.to_dict())

    def update(self, eqdsk_data: Dict[str, Any]):
        """
        Update this object's data with values from a dictionary.

        Parameters
        ----------
        eqdsk_data:
            A dict containing the new eqdsk data.

        Raises
        ------
        ValueError
            If a key in ``eqdsk_data`` does not correspond to an
            attribute of this class.
        """
        for key, value in eqdsk_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Cannot update EQDSKInterface from dict. Unrecognised key '{key}'."
                )


def _read_json(file) -> Dict[str, Any]:
    if isinstance(file, (Path, str)):
        with open(file) as f_h:
            return _read_json(f_h)

    data = json.load(file)
    data_has_pnorm = False
    data_has_psinorm = False
    for k, value in data.items():
        if isinstance(value, list):
            data[k] = np.asarray(value)
        data_has_pnorm |= k == "pnorm"
        data_has_psinorm |= k == "psinorm"

    # For backward compatibility where 'psinorm' was sometimes 'pnorm'
    if data_has_pnorm:
        if data_has_psinorm:
            del data["pnorm"]
        else:
            data["psinorm"] = data.pop("pnorm")

    return data


def _read_array(tokens, n, name="Unknown"):
    data = np.zeros([n])
    try:
        for i in np.arange(n):
            data[i] = float(next(tokens))
    except StopIteration:
        raise ValueError(f"Failed reading array {name} of size {n}") from None
    return data


def _read_2d_array(tokens, n_x, n_y, name="Unknown"):
    data = np.zeros([n_y, n_x])
    for i in np.arange(n_y):
        data[i, :] = _read_array(tokens, n_x, name + "[" + str(i) + "]")
    return np.transpose(data)


def _eqdsk_generator(file):
    """
    Transforms a file object into a generator, following G-EQDSK number
    conventions

    Parameters
    ----------
    file:
        The file to read

    Returns
    -------
    The generator of the file handle being read
    """
    while True:
        line = file.readline()
        if not line:
            break

        # Distinguish negative/positive numbers from negative/positive exponent
        if "E" in line or "e" in line:
            line = line.replace("E-", "*")
            line = line.replace("e-", "*")
            line = line.replace("-", " -")
            line = line.replace("*", "e-")
            line = line.replace("E+", "*")
            line = line.replace("e+", "*")
            line = line.replace("+", " ")
            line = line.replace("*", "e+")
        generator_list = line.split()
        yield from generator_list


def _read_eqdsk(file) -> Dict:
    if isinstance(file, (Path, str)):
        with open(file) as f_handle:
            return _read_eqdsk(f_handle)

    description = file.readline()
    if not description:
        raise OSError(f"Could not read the file '{file}'.")
    description = description.split()

    ints = [value for value in description if is_num(value)]

    if len(ints) < 3:  # noqa: PLR2004
        raise OSError(
            f"Should be at least 3 numbers in the first line of the EQDSK {file}."
        )

    data = {}
    n_x = int(ints[-2])
    n_z = int(ints[-1])
    data["name"] = description[0]
    data["nx"] = n_x
    data["nz"] = n_z

    tokens = _eqdsk_generator(file)
    for name in [
        "xdim",
        "zdim",
        "xcentre",
        "xgrid1",
        "zmid",
        "xmag",
        "zmag",
        "psimag",
        "psibdry",
        "bcentre",
        "cplasma",
        "psimag",
        None,
        "xmag",
        None,
        "zmag",
        None,
        "psibdry",
        None,
        None,
    ]:
        if name is not None:  # Lots of dummies and duplication
            data[name] = float(next(tokens))
        else:
            next(tokens)  # Dummy

    for name in ["fpol", "pressure", "ffprime", "pprime"]:
        data[name] = _read_array(tokens, n_x, name)

    data["psi"] = _read_2d_array(tokens, n_x, n_z, "psi")
    data["qpsi"] = _read_array(tokens, n_x, "qpsi")
    nbdry = int(next(tokens))
    nlim = int(next(tokens))
    data["nbdry"] = nbdry
    data["nlim"] = nlim

    xbdry = np.zeros(nbdry)
    zbdry = np.zeros(nbdry)
    for i in range(nbdry):
        xbdry[i] = float(next(tokens))
        zbdry[i] = float(next(tokens))
    data["xbdry"] = xbdry
    data["zbdry"] = zbdry

    xlim = np.zeros(nlim)
    zlim = np.zeros(nlim)
    for i in range(nlim):
        xlim[i] = float(next(tokens))
        zlim[i] = float(next(tokens))
    data["xlim"] = xlim
    data["zlim"] = zlim

    try:
        ncoil = int(next(tokens))
    except StopIteration:  # No coils in file
        ncoil = 0

    x_c = np.zeros(ncoil)
    z_c = np.zeros(ncoil)
    dxc = np.zeros(ncoil)
    dzc = np.zeros(ncoil)
    i_c = np.zeros(ncoil)
    for i in range(ncoil):
        x_c[i] = float(next(tokens))
        z_c[i] = float(next(tokens))
        dxc[i] = float(next(tokens))
        dzc[i] = float(next(tokens))
        i_c[i] = float(next(tokens))
    data["ncoil"] = ncoil
    data["xc"] = x_c
    data["zc"] = z_c
    data["dxc"] = dxc
    data["dzc"] = dzc
    data["Ic"] = i_c

    # Additional utility data
    data["x"] = _derive_x(data["xgrid1"], data["xdim"], data["nx"])
    data["z"] = _derive_z(data["zmid"], data["zdim"], data["nz"])
    data["psinorm"] = _derive_psinorm(data["fpol"])
    return data


def _derive_x(xgrid1, xdim, nx):
    return np.linspace(xgrid1, xgrid1 + xdim, nx)


def _derive_z(zmid, zdim, nz):
    return np.linspace(zmid - zdim / 2, zmid + zdim / 2, nz)


def _derive_psinorm(fpol):
    return np.linspace(0, 1, len(fpol))


def _write_eqdsk(file: Union[Path, str], data: Dict):
    """
    Writes data out to a text file in G-EQDSK format.

    Parameters
    ----------
    file:
        The full path string of the file to be created
    data:
        Dictionary of EQDSK data.
    """
    if isinstance(file, (str, Path)):
        file = Path(file)
        if file.suffix not in EQDSK_EXTENSIONS:
            file = Path(file).with_suffix("").with_suffix(".eqdsk")
        with open(file, "w") as f_handle:
            return _write_eqdsk(f_handle, data)

    def write_header(
        fortran_format: ff.FortranRecordWriter, id_string: str, var_list: List[str]
    ):
        """
        Writes G-EQDSK header out to file.

        Parameters
        ----------
        fortran_format:
            FortranRecordWriter object for Fortran format edit descriptor
            to be used for header output.
        id_string:
            String containing name of file to be used as identification
            string. Will be trimmed if length exceeds 39 characters,
            so it will fit within the permitted header length of the
            GEQDSK specification when a timestamp is added.
        var_list:
            List of names of keys in EQDSKInterface.data identifying
            variables to add to the header following the id_string.
            Empty strings will be recorded as 0.
        """
        line = [id_string]
        line += [data[v] if v else 0 for v in var_list]
        file.write(fortran_format.write(line))
        file.write("\n")

    def write_line(fortran_format: ff.FortranRecordWriter, var_list: List[str]):
        """
        Writes a line of variable values out to a G-EQDSK file.

        Parameters
        ----------
        fortran_format:
            FortranRecordWriter object for Fortran format edit descriptor
            to be used for the format of the line output.
        var_list:
            List of names of keys in EQDSKInterface.data identifying
            variables to added to the current line.
            Empty strings will be recorded as 0.
        """
        line = [data[v] if v else 0 for v in var_list]
        file.write(fortran_format.write(line))
        file.write("\n")

    def write_array(fortran_format: ff.FortranRecordWriter, array: np.ndarray):
        """
        Writes a numpy array out to a G-EQDSK file.

        Parameters
        ----------
        fortran_format:
            FortranRecordWriter object for Fortran format edit descriptor
            to be used for the format of the line output.
        array:
            Numpy array of variables to be written to file.
            Array will be flattened in column-major (Fortran)
            order if is more than one-dimensional.
        """
        if array.ndim > 1:
            flat_array = array.flatten(order="F")
            file.write(fortran_format.write(flat_array))
        else:
            file.write(fortran_format.write(array))
        file.write("\n")

    # Create id string for file comprising of timestamp and trimmed filename
    # that fits the 48 character limit of strings in EQDSK headers.
    timestamp = time.strftime("%d%m%Y")
    trimmed_name = data["name"][0 : 48 - len(timestamp) - 1]
    file_id_string = f"{trimmed_name}_{timestamp}"

    # Define dummy data for qpsi if it has not been previously defined.
    qpsi = np.zeros(data["nx"]) if data["qpsi"] is None else data["qpsi"]

    # Create array containing coilset information.
    coil = np.zeros(5 * data["ncoil"])
    for i, value in enumerate(["xc", "zc", "dxc", "dzc", "Ic"]):
        coil[i::5] = data[value]

    # Create FortranRecordWriter objects with the Fortran format
    # edit descriptors to be used in the G-EQDSK output.
    f2000 = ff.FortranRecordWriter("a48,3i4")
    f2020 = ff.FortranRecordWriter("5e16.9")
    f2022 = ff.FortranRecordWriter("2i5")
    fCSTM = ff.FortranRecordWriter("i5")

    # Write header in f2000 (6a8,3i4) format.
    write_header(f2000, file_id_string, ["", "nx", "nz"])
    # Write out lines containing floats in f2020 (5e16.9) format.
    write_line(f2020, ["xdim", "zdim", "xcentre", "xgrid1", "zmid"])
    write_line(f2020, ["xmag", "zmag", "psimag", "psibdry", "bcentre"])
    write_line(f2020, ["cplasma", "psimag", "", "xmag", ""])
    write_line(f2020, ["zmag", "", "psibdry", "", ""])
    # Write out arrays in in f2020 (5e16.9) format.
    write_array(f2020, data["fpol"])
    write_array(f2020, data["pressure"])
    write_array(f2020, data["ffprime"])
    write_array(f2020, data["pprime"])
    write_array(f2020, data["psi"])
    write_array(f2020, qpsi)
    # Write out number of boundary points and limiters f2022 (2i5) format.
    write_line(f2022, ["nbdry", "nlim"])
    # Write out boundary point and limiter data as array of ordered pairs.
    write_array(f2020, np.array([data["xbdry"], data["zbdry"]]))
    write_array(f2020, np.array([data["xlim"], data["zlim"]]))

    # Output of coilset information. This is an extension to the
    # regular eqdsk format.
    write_line(fCSTM, ["ncoil"])
    write_array(f2020, coil)
    return None
