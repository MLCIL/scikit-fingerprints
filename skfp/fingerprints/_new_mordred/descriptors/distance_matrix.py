import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix
from skfp.fingerprints._new_mordred.utils.matrix_attributes import (
    ATTRIBUTE_FEATURE_NAMES,
    MatrixAttributes,
)

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

# e.g. "distance matrix graph energy" (Mordred's SpAbs_D), "distance matrix
# LogEE" (LogEE_D)
FEATURE_NAMES = [
    f"distance matrix {name}"
    for attr, name in ATTRIBUTE_FEATURE_NAMES.items()
    if attr != "sm1"
]


def calc(
    atomic_props_regular: AtomicProperties,
    n_frags: int,
    distance_matrix_regular: DistanceMatrix,
) -> np.ndarray:
    # avoids unnecessary eigendecomposition for disconnected molecules
    if n_frags != 1:
        return np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)

    dist_matrix = distance_matrix_regular.matrix
    attrs = MatrixAttributes(
        dist_matrix,
        atomic_props_regular,
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
    return values
