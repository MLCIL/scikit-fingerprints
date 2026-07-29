import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.graph_matrix import AdjacencyMatrix
from skfp.fingerprints._new_mordred.utils.matrix_attributes import MatrixAttributes

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = [
    "SpAbs_A",
    "SpMax_A",
    "SpDiam_A",
    "SpAD_A",
    "SpMAD_A",
    "LogEE_A",
    "VE1_A",
    "VE2_A",
    "VE3_A",
    "VR1_A",
    "VR2_A",
    "VR3_A",
]


def calc(
    atomic_atomic_props_regular: AtomicProperties,
    n_frags: int,
    adjacency_matrix: AdjacencyMatrix,
) -> np.ndarray:
    # avoids unnecessary eigendecomposition for disconnected molecules
    if n_frags != 1:
        return np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)

    adj_matrix = adjacency_matrix.matrix
    attrs = MatrixAttributes(
        adj_matrix,
        atomic_atomic_props_regular,
        hermitian=adjacency_matrix.hermitian,
        n_frags=n_frags,
    )
    values = np.asarray(
        [
            attrs.graph_energy,  # SpAbs_A
            attrs.leading_eigenvalue,  # SpMax_A
            attrs.spectral_diameter,  # SpDiam_A
            attrs.sp_ad,  # SpAD_A
            attrs.sp_mad,  # SpMAD_A
            attrs.log_ee,  # LogEE_A
            attrs.ve1,  # VE1_A
            attrs.ve2,  # VE2_A
            attrs.ve3,  # VE3_A
            attrs.vr1,  # VR1_A
            attrs.vr2,  # VR2_A
            attrs.vr3,  # VR3_A
        ],
        dtype=np.float32,
    )
    return values
