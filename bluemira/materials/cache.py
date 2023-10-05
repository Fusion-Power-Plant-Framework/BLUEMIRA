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
Classes and methods to load, store, and retrieve materials.
"""

import copy
import json
from typing import Any, ClassVar, Dict

from bluemira.materials.material import (
    BePebbleBed,
    Liquid,
    MassFractionMaterial,
    MaterialsError,
    NbSnSuperconductor,
    NbTiSuperconductor,
    Plasma,
    SerialisedMaterial,
    UnitCellCompound,
    Void,
)
from bluemira.materials.mixtures import HomogenisedMixture


class MaterialCache:
    """
    A helper class for loading and caching materials.

    Notes
    -----
    Extend the `available_classes` attribute to load custom classes.
    """

    _material_dict: ClassVar = {}

    default_classes = (
        Void,
        MassFractionMaterial,
        NbTiSuperconductor,
        NbSnSuperconductor,
        Liquid,
        UnitCellCompound,
        BePebbleBed,
        Plasma,
        HomogenisedMixture,
    )

    def __init__(self):
        self.available_classes = {
            mat_class.__name__: mat_class for mat_class in self.default_classes
        }

    def load_from_file(self, path: str) -> Dict[str, Any]:
        """
        Load materials from a file.

        Parameters
        ----------
        path:
            The path to the file from which to load the materials.

        Returns
        -------
        The dictionary containing the loaded materials.
        """
        with open(path) as fh:
            mats_dict = json.load(fh)
        return {name: self.load_from_dict(name, mats_dict) for name in mats_dict}

    def load_from_dict(
        self, mat_name: str, mats_dict: Dict[str, Any], overwrite: bool = True
    ):
        """
        Load a material or mixture from a dictionary.

        Parameters
        ----------
        mat_name:
            The name of the material or mixture.
        mat_dict:
            The dictionary containing the material or mixture attributes to be loaded.
        """
        material_class = mats_dict[mat_name]["material_class"]
        if material_class not in self.available_classes:
            raise MaterialsError(
                f"Request to load unknown material class {material_class}"
            )

        if issubclass(self.available_classes[material_class], HomogenisedMixture):
            self.mixture_from_dict(mat_name, mats_dict, overwrite=overwrite)
        else:
            self.material_from_dict(mat_name, mats_dict, overwrite=overwrite)

    def mixture_from_dict(
        self, mat_name: str, mats_dict: Dict[str, Any], overwrite: bool = True
    ):
        """
        Load a mixture from a dictionary.

        Parameters
        ----------
        mat_name:
            The name of the mixture.
        mat_dict:
            The dictionary containing the mixture attributes to be loaded.
        """
        class_name = mats_dict[mat_name].pop("material_class")
        mat_class = self.available_classes[class_name]
        mat = mat_class.from_dict(mat_name, mats_dict, self)
        self._update_cache(mat_name, mat, overwrite=overwrite)

    def material_from_dict(
        self, mat_name: str, mats_dict: Dict[str, Any], overwrite: bool = True
    ):
        """
        Load a material from a dictionary.

        Parameters
        ----------
        mat_name:
            The name of the material.
        mat_dict:
            The dictionary containing the material attributes to be loaded.
        """
        class_name = mats_dict[mat_name].pop("material_class")
        mat_class = self.available_classes[class_name]
        mat = mat_class.from_dict(mat_name, mats_dict)
        self._update_cache(mat_name, mat, overwrite=overwrite)

    def get_material(self, name: str, clone: bool = True) -> SerialisedMaterial:
        """
        Get the named material from the material dictionary

        Parameters
        ----------
        name:
            The name of the material to retrieve from the dictionary
        clone:
            If True, get a clone (deepcopy) of the material, else get the actual material
            as stored in the material dictionary. By default True.

        Returns
        -------
        The requested material.
        """
        if clone:
            return copy.deepcopy(self._material_dict[name])
        return self._material_dict[name]

    def _update_cache(
        self, mat_name: str, mat: SerialisedMaterial, overwrite: bool = True
    ):
        if not overwrite and mat_name in self._material_dict:
            raise MaterialsError(
                f"Attempt to load material {mat_name}, which already "
                "exists in the cache."
            )
        self._material_dict[mat_name] = mat
