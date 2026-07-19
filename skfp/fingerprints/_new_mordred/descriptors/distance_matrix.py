import numpy as np
from rdkit.Chem import Mol

from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix
from skfp.fingerprints._new_mordred.utils.matrix_attributes import MatrixAttributes

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = [
    "SpAbs_D",
    "SpMax_D",
    "SpDiam_D",
    "SpAD_D",
    "SpMAD_D",
    "LogEE_D",
    "VE1_D",
    "VE2_D",
    "VE3_D",
    "VR1_D",
    "VR2_D",
    "VR3_D",
]


def calc(
    mol_regular: Mol, n_frags: int, distance_matrix_regular: DistanceMatrix
) -> tuple[np.ndarray, list[str]]:
    # avoids unnecessary eigendecomposition for disconnected molecules
    if n_frags != 1:
        return np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32), FEATURE_NAMES

    dist_matrix = distance_matrix_regular.matrix
    attrs = MatrixAttributes(
        dist_matrix,
        mol_regular,
        hermitian=distance_matrix_regular.hermitian,
        n_frags=n_frags,
    )
    values = np.asarray(
        [
            attrs.graph_energy,  # SpAbs_D
            attrs.leading_eigenvalue,  # SpMax_D
            attrs.spectral_diameter,  # SpDiam_D
            attrs.sp_ad,  # SpAD_D
            attrs.sp_mad,  # SpMAD_D
            attrs.log_ee,  # LogEE_D
            attrs.ve1,  # VE1_D
            attrs.ve2,  # VE2_D
            attrs.ve3,  # VE3_D
            attrs.vr1,  # VR1_D
            attrs.vr2,  # VR2_D
            attrs.vr3,  # VR3_D
        ],
        dtype=np.float32,
    )
    return values, FEATURE_NAMES
