import numpy as np

from skfp.fingerprints._new_mordred.utils.graph_matrix import (
    AdjacencyMatrix,
    DistanceMatrix,
)

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["ECIndex"]


def calc(
    adjacency_matrix_regular: AdjacencyMatrix, distance_matrix_regular: DistanceMatrix
) -> tuple[np.ndarray, list[str]]:
    r"""
    Compute the Mordred eccentric connectivity index descriptor.

    `ECIndex` is defined as :math:`\sum_{i}^{A} E_i D_i`, where :math:`E_i` is the
    eccentricity of atom :math:`i`, :math:`D_i` is its vertex degree, and :math:`A`
    is the number of heavy atoms.
    """
    E = distance_matrix_regular.eccentricities
    D = adjacency_matrix_regular.degree

    value = (E * D).sum()

    return np.array([value], dtype=np.float32), FEATURE_NAMES
