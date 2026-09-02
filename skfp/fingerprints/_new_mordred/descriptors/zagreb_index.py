import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.graph_matrix import AdjacencyMatrix

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["Zagreb1", "Zagreb2", "mZagreb1", "mZagreb2"]


def calc(
    props: AtomicProperties, adjacency_matrix_regular: AdjacencyMatrix
) -> np.ndarray:
    """
    Zagreb indices.

    The first index sums the squared degree of every atom, the second the product of
    the degrees at both ends of every bond. Their modified variants sum the same
    quantities inverted, which is undefined when something has no degree to invert.

    Computes the same values as :func:`skfp.descriptors.zagreb_index_m1` and
    :func:`skfp.descriptors.zagreb_index_m2`, which sum over the bonds in Python.
    """
    degrees = adjacency_matrix_regular.degree
    bond_degrees = degrees[props.bond_begin_idxs] * degrees[props.bond_end_idxs]

    with np.errstate(divide="ignore", invalid="ignore"):
        zagreb_1 = (degrees**2).sum()
        zagreb_2 = bond_degrees.sum()
        modified_1 = np.nan if (degrees == 0).any() else (degrees**-2.0).sum()
        modified_2 = np.nan if (bond_degrees == 0).any() else (1.0 / bond_degrees).sum()

    return np.asarray([zagreb_1, zagreb_2, modified_1, modified_2], dtype=np.float32)
