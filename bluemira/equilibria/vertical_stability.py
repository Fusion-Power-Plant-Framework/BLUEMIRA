# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Vertical stability calculations
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bluemira.base.look_and_feel import bluemira_debug, bluemira_warn
from bluemira.equilibria.coils._tools import make_mutual_inductance_matrix

if TYPE_CHECKING:
    import numpy.typing as npt

    from bluemira.equilibria.coils._grouping import CoilSet
    from bluemira.equilibria.equilibrium import Equilibrium


def calculate_rzip_stability_criterion(eq: Equilibrium) -> float:
    """
    Calculate the rzip stability criterion for a given equilibrium and coilset

    Returns
    -------
    :
        Stability criterion

    Notes
    -----
    See ~:class:`~bluemira.equilibria.vertical_stability.RZIp` for details
    """
    return RZIp(eq.coilset)(eq)


class RZIp:
    """
    RZIp model

    Parameters
    ----------
    coilset:
        The full coilset

    Notes
    -----
    A value ~ 1.5 considered optimal and controllable.
    See https://doi.org/10.13182/FST89-A39747 for further explanation

        < 1 plasma mass becomes a factor
        = 1 massless plasma solution not valid (said to represent MHD effects)
        > 1 displacement growth dominated by L/R of passive system

    .. math::

        f = -\\frac{F_s}{F_d}
          = \\frac{I_p^T M^{\\prime}_p|_s [M_s|_s]^{-1} M^{\\prime}_s|_p I_p}
                  {I_p^T M^{\\prime\\prime}_p|_c I_c}
    """

    def __init__(self, coilset):
        self.coilset = coilset

    @property
    def coilset(self):
        return self._coilset

    @coilset.setter
    def coilset(self, cs: CoilSet):
        # TODO @je-cook Possibly implement quad indexing?
        self._coilset = cs
        self.ind_mat = make_mutual_inductance_matrix(cs, square_coil=True)

    def __call__(
        self,
        eq: Equilibrium,
        cc_current_vector: None | npt.NDArray = None,
    ) -> float:
        if not hasattr(eq, "profiles"):
            bluemira_warn("No profiles found on equilibrium")
            # this constraint is pretty irrelevant for breakdown
            return 0

        cc = self.coilset.get_control_coils()

        if cc_current_vector is None:
            cc_current_vector = cc.current
        else:
            cc.current = cc_current_vector

        eq._remap_greens()

        control_ind = self.coilset._control_ind
        uncontrolled_ind = list(
            set(self.coilset._get_type_index()) - set(self.coilset._control_ind)
        )

        return stab_destab(
            cc_current=cc_current_vector,
            # res_ps=self.coilset.get_uncontrolled_coils().resistance,
            l_ps_ps=self.ind_mat[uncontrolled_ind][:, uncontrolled_ind],
            l_ac_ps=self.ind_mat[control_ind][:, uncontrolled_ind],
            l_ac_ac=self.ind_mat[control_ind][:, control_ind],
            # ind=M,
            r_struct=self.coilset.x,
            x_ac=cc.x,
            i_plasma=eq._jtor * eq.grid.step,
            br_struct_grid=eq._bx_green,
            dbrdz_struct_grid=eq._db_green[..., control_ind],
            # max_eigens=self.max_eigens,
        )


def stab_destab(
    cc_current,
    # res_ps,
    l_ps_ps,
    l_ac_ps,
    l_ac_ac,
    r_struct,
    x_ac,
    i_plasma,
    br_struct_grid,
    dbrdz_struct_grid,
    # max_eigens,
) -> float:
    """
    Calculate the stabilising / destabilising effect of the equilibria and structures
    """
    # # single filament approx l_acps_eigen_vec
    # eigen_values, eigen_vectors = eigenv(res_ps, l_ps_ps, max_eigens=max_eigens)
    # l_acps_eigen_vec = l_ac_ps @ eigen_vectors

    # # l_struct in rzip
    # mss = np.hstack([
    #     np.vstack([l_ac_ac, l_acps_eigen_vec.T]),
    #     np.vstack([l_acps_eigen_vec, eigen_values]),
    # ])

    # l_struct in rzip
    mss = np.vstack([
        np.hstack([l_ac_ac, l_ac_ps]),
        np.hstack([l_ac_ps.T, l_ps_ps]),
    ])

    msp_prime = (2 * np.pi * r_struct * br_struct_grid).reshape(-1, r_struct.size)
    i_plasma = i_plasma.reshape(-1)

    # stabilising force/ destabilising force differentiated wrt to z coord
    # f = -d_fs / d_fd
    destabilising = np.einsum(
        "b, bd, d",
        i_plasma,
        (2 * np.pi * x_ac * dbrdz_struct_grid).reshape(-1, cc_current.size),
        cc_current,
        optimize=["einsum_path", (0, 1), (0, 1)],
    )

    stabilising = np.einsum(
        "d, db, bc, ac, a",
        i_plasma,
        msp_prime,
        np.linalg.inv(mss),
        msp_prime,
        i_plasma,
        optimize=["einsum_path", (0, 1), (1, 2), (0, 1), (0, 1)],
    )

    # from scipy.io import loadmat

    # ml = loadmat("leuer_test.mat")
    # import ipdb

    # ipdb.set_trace()
    # not infinite if destabilising is 0 because therefore it is stable
    criterion = (-stabilising / destabilising) if destabilising != 0 else 0
    bluemira_debug(f"{stabilising=}, {destabilising=}, {criterion=}")
    return criterion


# def mutual_ind(xc, zc, x, z):
#     r"""File to compute the mutual inductance between two coaxial circular filaments,
#     i.e. the planes of the two coils are parallel and they share the same central axis.
#     (William R. Smythe: Static and Dynaimc Electricity Third Edition, p.335 equation 1)

#     .. math::

#         M_{12} &= 2\mu_0 k^{-1} (ab)^{1/2} \left[(1-\frac{1}{2}k^2)K-E\right]

#                &= \frac{2 \mu_0 }{k}  \sqrt{ab}\left[\left(1-\frac{k^2}{2}\right)K-E\right]

#     where

#     .. math::

#         k^2 = \frac{4ab}{(a+b)^2 + z^2}

#     Note the symmetry between :math:`a` and :math:`b` in the equations:
#     :math:`a` and :math:`b` are commutable.

#     Parameters
#     ----------
#     a, b : radii of coil 1 and coil 2 respectively [m]
#         (denoted with :math:`a` and :math:`b` in the equation)

#     z: separation between the planes of coils [m]

#     Returns
#     -------
#     Mutual Inductance: unit: [N/m/A^2]
#     """
#     _, k2 = calc_a_k2(xc, zc, x, z)
#     e, k = calc_e_k(k2)
#     i_ch = np.nonzero(k2 == 1)
#     calc_a_k2(xc, zc, x, z)
#     # TODO: review this hack!
#     k2[i_ch] = 0.1
#     mut_ind = 2 * MU_0 / np.sqrt(k2) * np.sqrt(xc * zc) * ((1 - 0.5 * k2) * k - e)
#     mut_ind[i_ch] = 0
#     return mut_ind


def eigenv(res_ps, l_ps_ps, max_eigens=40):
    inv_r = np.linalg.inv(res_ps[..., None] * np.eye(res_ps.shape[-1]))
    inv_r_l = np.einsum("...ab, ...bc -> ...ac", inv_r, l_ps_ps)
    eig_val, eig_vec = np.linalg.eig(np.asarray(inv_r_l, dtype=np.complex128))

    # sort (eig_vec_sort) into ascending importance of values (vectors Es and values Vs)
    sort_ind = np.argsort(eig_val, axis=-1)[..., ::-1]
    eig_val = eig_val[..., sort_ind].real
    eig_vec = eig_vec[..., sort_ind].real
    norm_res = 1 / np.sum(  # TODO compare with rzip with multiple structures
        1 / np.atleast_2d(res_ps), axis=-1
    )
    norm_mat = np.einsum(  # diagonalise
        "...aa -> ...a",
        np.einsum(  # arr1 @ arr2 @ inv(arr3.T)
            "...ab, ...bc, ...cd -> ...ad",
            norm_res * np.linalg.inv(eig_vec),
            inv_r,
            np.linalg.inv(np.einsum("...ij -> ...ji", eig_vec)),
            optimize=["einsum_path", (0, 1), (0, 1)],
        ),
    )
    idx = np.nonzero(norm_mat >= 0)
    norm_mat[idx] = np.sqrt(norm_mat[idx])
    idx = np.nonzero(norm_mat < 0)
    norm_mat[idx] = -np.sqrt(-norm_mat[idx])

    eig_val *= norm_res
    eig_vec @= norm_mat[..., None] * np.eye(norm_mat.shape[-1])

    # eig_val = eig_val[..., :max_eigens]
    # eig_vec = eig_vec[..., :max_eigens]
    return eig_val * np.eye(eig_val.shape[0]), eig_vec

    # TODO inter structure interactions
    eigen_vectors_2 = np.zeros((n_passive_fils, n_eigen_modes_max))
    eigen_values_2 = np.zeros((1, n_eigen_modes_max))

    for k in range(n_passive_struct):
        passive_range = slice(passive_coil_ranges[k, 0], passive_coil_ranges[k, 1])
        eigen_range_start = n_eigen_modes_array_max[k] if k > 0 else 0
        eigen_range = slice(
            eigen_range_start, n_eigen_modes_array_max[k] + eigen_range_start
        )

        # structure to structure terms
        # TODO untested
        for k2 in range(k + 1, n_passive_struct):
            raise NotImplementedError(">1 passive structure not implemented")
            passive_range2 = slice(
                passive_coil_ranges[k2, 0], passive_coil_ranges[k2, 1]
            )

            eigen_range_start2 = sum(n_eigen_modes_array_max[:k2])
            eigen_range2 = slice(
                eigen_range_start2, n_eigen_modes_array_max[k2] + eigen_range_start2
            )

            et_me = (
                eigen_vectors[k2, :, : n_eigen_modes_array_max[k]]
                @ l_pf_pf[passive_range, passive_range2]
            ) @ eigen_vectors[k2, :, : n_eigen_modes_array_max[k2]]
            eigen_values_2[k, eigen_range, eigen_range2] = et_me
            eigen_values_2[k, eigen_range2, eigen_range] = et_me

        # TODO if there is more than 1 passive structure it will break
        eigen_values_2[eigen_range] = eigen_values[k]
        eigen_vectors_2[passive_range, eigen_range] = eigen_vectors[k]

    eigen_vectors = eigen_vectors_2
    eigen_values = eigen_values_2

    return eigen_values, eigen_vectors
