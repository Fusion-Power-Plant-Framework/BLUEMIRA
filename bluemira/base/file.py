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
File I/O functions and some path operations
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union

BM_ROOT = "!BM_ROOT!"
SUB_DIRS = ["equilibria", "neutronics", "systems_code", "CAD", "plots", "geometry"]


def _get_relpath(folder: str, subfolder: str) -> str:
    path = Path(folder, subfolder)
    if path.is_dir():
        return path.as_posix()
    raise ValueError(f"{path} Not a valid folder.")


def get_bluemira_root() -> str:
    """
    Get the bluemira root install folder.

    Returns
    -------
        The full path to the bluemira root folder, e.g.:
            '/home/user/code/bluemira'
    """
    import bluemira

    path = next(iter(bluemira.__path__))
    return os.path.split(path)[0]


def try_get_bluemira_private_data_root() -> Union[str, None]:
    """
    Get the bluemira-private-data root install folder.

    Returns
    -------
    The full path to the bluemira root folder, e.g.:
        '/home/user/code/bluemira-private-data'

    Notes
    -----
    Normal users will not have access to bluemira-private-data; it will be used
    exclusively for tests which require private data and files.
    """
    root = get_bluemira_root()
    code_root = os.path.split(root)[0]
    try:
        return _get_relpath(code_root, "bluemira-private-data")
    except ValueError:
        return None


def get_bluemira_path(path: str = "", subfolder: str = "bluemira") -> str:
    """
    Get a bluemira path of a module subfolder. Defaults to root folder.

    Parameters
    ----------
    path:
        The desired path from which to create a full path
    subfolder:
        The subfolder (from the bluemira root) in which to create a path
        Defaults to the source code folder, but can be e.g. 'tests', or 'data'

    Returns
    -------
    The full path to the desired `path` in the subfolder specified
    """
    root = get_bluemira_root()
    if "egg" in root:
        return f"/{subfolder}"

    path = path.replace("/", os.sep)
    bpath = _get_relpath(root, subfolder)
    return _get_relpath(bpath, path)


def try_get_bluemira_path(
    path: str = "", subfolder: str = "bluemira", allow_missing: bool = True
) -> Optional[str]:
    """
    Try to get the bluemira path of a module subfolder.

    If the path doesn't exist then optionally carry on regardless or raise an error.

    Parameters
    ----------
    path:
        The desired path from which to create a full path
    subfolder:
        The subfolder (from the bluemira root) in which to create a path
        Defaults to the source code folder, but can be e.g. 'tests', or 'data'
    allow_missing:
        Whether or not to raise an error if the path does not exist

    Returns
    -------
    The full path to the desired `path` in the subfolder specified, or None if the
    requested path doesn't exist.

    Raises
    ------
    ValueError
        If the requested path doesn't exist and the `allow_missing` flag is False.
    """
    try:
        return get_bluemira_path(path, subfolder)
    except ValueError:
        if allow_missing:
            return None
        raise


def make_bluemira_path(path: str = "", subfolder: str = "bluemira") -> str:
    """
    Create a new folder in the path, provided one does not already exist.
    """
    root = get_bluemira_root()
    if "egg" in root:
        root = "/"
    path = path.replace("/", os.sep)
    bpath = _get_relpath(root, subfolder)
    if bpath in path:
        path = path[len(bpath) :]  # Remove leading edge rootpath
    try:
        return _get_relpath(bpath, path)
    except ValueError:
        Path(bpath, path).mkdir(parents=True)
        return _get_relpath(bpath, path)


def force_file_extension(file_path: str, valid_extensions: Union[str, List[str]]) -> str:
    """
    If the file path does not have one of the valid extensions, append the first
    valid one

    Parameters
    ----------
    file_path:
        path to file
    valid_extensions:
        collection of valid extensions

    Returns
    -------
    File path
    """
    if isinstance(valid_extensions, str):
        valid_extensions = [valid_extensions]

    if Path(file_path).suffix.casefold() not in valid_extensions:
        file_path += valid_extensions[0]

    return file_path


def get_files_by_ext(folder: str, extension: str) -> List[str]:
    """
    Get filenames of files in folder with the specified extension.

    Parameters
    ----------
    folder:
        The full path directory in which to look for files
    extension:
        The extension of the desired file-type

    Returns
    -------
    The list of full path filenames found in the folder
    """
    files = [file for file in os.listdir(folder) if file.endswith(extension)]
    if len(files) == 0:
        from bluemira.base.look_and_feel import bluemira_warn

        bluemira_warn(f"No files with extension {extension} found in folder {folder}")
    return files


def file_name_maker(filename: Union[str, Path], lowercase: bool = False) -> str:
    """
    Ensure the file name is acceptable.

    Parameters
    ----------
    filename:
        Full filename or path
    lowercase:
        Whether or not to force lowercase filenames

    Returns
    -------
    Full filename or path, corrected
    """
    filename = Path(filename.replace(" ", "_"))
    if lowercase:
        split = filename.parts
        filename = Path(split[:-1], split[-1].lower())
    return filename.as_posix()


@contextmanager
def working_dir(directory: str):
    """Change working directory"""
    current_dir = Path.cwd().as_posix()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(current_dir)


class FileManager:
    """
    A class for managing file operations.
    """

    _reactor_name: str
    _reference_data_root: str
    _generated_data_root: str

    reference_data_dirs: Dict[str, str]
    generated_data_dirs: Dict[str, str]

    def __init__(
        self,
        reactor_name: str,
        reference_data_root: str = "data/bluemira",
        generated_data_root: str = "data/bluemira",
    ):
        self._reactor_name = reactor_name
        self._reference_data_root = reference_data_root
        self._generated_data_root = generated_data_root
        self.replace_bm_root()

    @property
    def reactor_name(self):
        """
        Gets the reactor name for this instance.
        """
        return self._reactor_name

    @property
    def generated_data_root(self) -> str:
        """
        Gets the generated data root directory for this instance.
        """
        return self._generated_data_root

    @property
    def reference_data_root(self) -> str:
        """
        Get the reference data root directory for this instance.
        """
        return self._reference_data_root

    def replace_bm_root(self, keyword: str = BM_ROOT):
        """
        Replace the keyword in input paths with path to local bluemira installation.
        """
        bm_root = get_bluemira_root()
        self._reference_data_root = self.reference_data_root.replace(keyword, bm_root)
        self._generated_data_root = self.generated_data_root.replace(keyword, bm_root)

    def _verify_reference_data_root(self):
        """
        Check that the reference data root defined in this instance is a valid
        directory.

        Raises
        ------
        ValueError
            If the reference data root for this instance is not a valid directory.
        """
        _get_relpath(self._reference_data_root, subfolder="")

    def make_reactor_folder(self, subfolder: str) -> Dict[str, str]:
        """
        Initialise a data storage folder tree.

        Parameters
        ----------
        subfolder:
            The subfolder of the bluemira directory in which to add the data structure

        Returns
        -------
        The dictionary of subfolder names to full paths (useful shorthand)
        """
        root = Path(subfolder, "reactors", self.reactor_name).as_posix()
        Path(root).mkdir(parents=True, exist_ok=True)

        mapping = {"root": root}
        for sub in SUB_DIRS:
            folder = Path(root, sub)
            Path(folder).mkdir(exist_ok=True)

            mapping[sub] = folder.as_posix()

        return mapping

    def set_reference_data_paths(self):
        """
        Generate the reference data paths for this instance, based on the reactor name.
        """
        self._verify_reference_data_root()
        self.reference_data_dirs = self.make_reactor_folder(self._reference_data_root)

    def create_reference_data_paths(self):
        """
        Generate the reference data paths for this instance, based on the reactor name.

        Also builds the relevant directory structure.
        """
        Path(self._reference_data_root).mkdir(parents=True, exist_ok=True)
        self.reference_data_dirs = self.make_reactor_folder(self._reference_data_root)

    def create_generated_data_paths(self):
        """
        Generate the generated data paths for this instance, based on the reactor name.

        Also builds the relevant directory structure.
        """
        Path(self._generated_data_root).mkdir(parents=True, exist_ok=True)
        self.generated_data_dirs = self.make_reactor_folder(self._generated_data_root)

    def build_dirs(self, create_reference_data_paths: bool = False):
        """
        Create the directory structures for this instance and sets the path references.
        """
        if create_reference_data_paths:
            self.create_reference_data_paths()
        else:
            self.set_reference_data_paths()
        self.create_generated_data_paths()

    def get_path(
        self, sub_dir_name: str, path: Union[Path, str], make_dir: bool = False
    ) -> str:
        """
        Get a path within the generated data sub-sdirectories.

        If the path does not exist then it will optionally be created as a directory.

        Parameters
        ----------
        sub_dir_name:
            The name of the sub-directory to create the path under. Must be one of the
            names in bluemira.base.file.SUB_DIRS.
        path:
            The path to create under the sub-directory.
        make_dir:
            Optionally create a directory at the path, by default False.

        Returns
        -------
        The path within the data sub-directories.
        """
        path = Path(self.generated_data_dirs[sub_dir_name], path)
        if make_dir and not path.is_dir():
            path.mkdir(parents=True)
        return path.as_posix()
