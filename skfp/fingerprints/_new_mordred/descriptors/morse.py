import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    CARBON_PROPERTY_VALUES,
    AtomicProperties,
)
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix3D

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

# atomic properties used to weight distance matrices
_PROPS = [
    "unweighted",  # pure interatomic distance
    "mass",
    "van_der_Waals_volume",
    "Sanderson_electronegativity",
    "polarizability",
]
_DISTANCES = np.arange(1, 33)

FEATURE_NAMES = [
    f"MoRSE_{prop_name}_dist_{dist}" for prop_name in _PROPS for dist in _DISTANCES
]


@np.errstate(divide="ignore", invalid="ignore")
def calc(
    props_3d: AtomicProperties, distance_matrix_3d: DistanceMatrix3D
) -> np.ndarray:
    """
    MoRSE descriptors.

    Quantifies correlation of 3D interatomic distances, weighted by various
    properties. Property values are normalized by the value for carbon prior
    to weighting.
    """
    num_atoms = props_3d.num_atoms

    if num_atoms < 2:
        return np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)

    # scaled distances for all 32 kernels, shape (32, n, n)
    # the first kernel is the unscaled sin(x)/x limit of 1
    scaled_dists = np.multiply.outer(_DISTANCES[1:] - 1, distance_matrix_3d.matrix)
    diagonal = np.arange(num_atoms)
    scaled_dists[:, diagonal, diagonal] = 1.0

    dist_kernels = np.empty((len(_DISTANCES), num_atoms, num_atoms), dtype=np.float64)
    dist_kernels[0] = 1.0
    dist_kernels[1:] = np.sin(scaled_dists) / scaled_dists
    dist_kernels[:, diagonal, diagonal] = 0.0

    prop_vectors = np.stack(
        [
            np.ones(num_atoms)
            if name == "unweighted"
            # normalize by value for carbon
            else props_3d.get(name) / CARBON_PROPERTY_VALUES[name]
            for name in _PROPS
        ]
    )

    # one contraction for all property x kernel combinations, shape (n_props, 32)
    values = 0.5 * np.einsum(
        "pi,kij,pj->pk",
        prop_vectors,
        dist_kernels,
        prop_vectors,
        optimize=True,
    )

    return values.ravel().astype(np.float32)
