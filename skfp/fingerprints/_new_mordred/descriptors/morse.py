import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    CARBON_PROPERTY_VALUES,
    ELEMENT_PROPERTY_TABLES,
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


def calc(atomic_nums: np.ndarray, distance_matrix_3d: DistanceMatrix3D) -> np.ndarray:
    """
    MoRSE descriptors.

    Quantifies correlation of 3D interatomic distances, weighted by various
    properties. Property values are normalized by the value for carbon prior
    to weighting.

    Every descriptor sums ``w_i * w_j * sin(s * r_ij) / (s * r_ij)`` over the atom
    pairs, for one atom weighting ``w`` and one scale ``s``. Only distinct pairs
    contribute, so we cna sum over them, rather than using a symmetric matrix.
    """
    num_atoms = len(atomic_nums)

    if num_atoms < 2:
        return np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)

    first, second = np.triu_indices(num_atoms, k=1)
    pair_distances = distance_matrix_3d.matrix[first, second]

    # sin(s * r) / (s * r) for every scale, shape (32, n_pairs)
    # first scale is zero, where the kernel takes its limit value of 1
    kernels = np.empty((len(_DISTANCES), len(pair_distances)), dtype=np.float64)
    kernels[0] = 1.0
    kernels[1:] = _sines_of_multiples(pair_distances, len(_DISTANCES) - 1) / (
        np.multiply.outer(_DISTANCES[1:] - 1, pair_distances)
    )

    # property values of both atoms of every pair, multiplied, shape (n_props, n_pairs)
    prop_vectors = np.stack(
        [
            np.ones(num_atoms)
            if name == "unweighted"
            # normalize by value for carbon
            else ELEMENT_PROPERTY_TABLES[name].lookup(atomic_nums)
            / CARBON_PROPERTY_VALUES[name]
            for name in _PROPS
        ]
    )
    pair_weights = prop_vectors[:, first] * prop_vectors[:, second]

    # product for all property x scale combinations, shape (n_props, 32)
    values = pair_weights @ kernels.T

    return values.ravel().astype(np.float32)


def _sines_of_multiples(values: np.ndarray, count: int) -> np.ndarray:
    """
    ``sin(k * values)`` for ``k = 1..count``, of shape ``(count, len(values))``.

    Note the following recurrence:
    sin((k + 1) x) = 2 cos(x) sin(kx) - sin((k - 1) x)

    With this, instead of whole matrix of sines, like in canonical MoRSE formula,
    we have just one sine evaluation, plus application of this formula.
    """
    sines = np.empty((count + 1, len(values)), dtype=np.float64)
    sines[0] = 0.0  # sin(0)
    sines[1] = np.sin(values)
    twice_cosines = 2.0 * np.cos(values)
    for multiple in range(1, count):
        sines[multiple + 1] = twice_cosines * sines[multiple] - sines[multiple - 1]
    return sines[1:]
