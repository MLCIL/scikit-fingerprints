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

# highest topological distance (order) for which charges are accumulated
_MAX_ORDER = 10

FEATURE_NAMES = [
    *[f"GGI{order}" for order in range(1, _MAX_ORDER + 1)],  # raw
    *[f"JGI{order}" for order in range(1, _MAX_ORDER + 1)],  # mean
    "JGT10",  # global
]


def calc(
    adjacency_matrix_regular: AdjacencyMatrix,
    distance_matrix_regular: DistanceMatrix,
) -> np.ndarray:
    """
    Topological charge descriptors (Galvez charge indices).

    Charge transfers between atom pairs are read off the charge term matrix,
    the antisymmetric product of the adjacency matrix, and the inverse-square
    topological distance matrix. For each topological distance (order) ``k``
    the absolute charge terms of atom pairs exactly ``k`` bonds apart are
    accumulated:

        * "raw" (GGI{k}): sum of the absolute charge terms
        * "mean" (JGI{k}): the raw sum divided by the number of such pairs
        * "global" (JGT10): sum of the mean charges over orders 1 to 10
    """
    charge_terms = _charge_term_matrix(
        adjacency_matrix_regular.matrix, distance_matrix_regular.matrix
    )

    # keep only lower-triangle atom pairs (strictly below the diagonal), so each
    # unordered pair is counted once; mark everything else as unreachable so it
    # is dropped when filtering by topological distance
    dist = distance_matrix_regular.matrix * np.tri(*charge_terms.shape)
    dist[dist == 0] = np.inf

    raw = np.zeros(_MAX_ORDER, dtype=np.float32)
    mean = np.zeros(_MAX_ORDER, dtype=np.float32)
    for order in range(1, _MAX_ORDER + 1):
        mask = dist == order
        count = np.count_nonzero(mask)
        raw_sum = np.abs(charge_terms[mask]).sum()

        raw[order - 1] = raw_sum
        mean[order - 1] = raw_sum / count if count else 0.0

    global_charge = mean.sum()

    values = np.concatenate([raw, mean, [global_charge]], dtype=np.float32)
    return values


def _charge_term_matrix(adj_matrix: np.ndarray, dist_matrix: np.ndarray) -> np.ndarray:
    """
    Charge term matrix ``M - M.T``, with ``M = A . D``, where ``A`` is the
    adjacency matrix and ``D`` the inverse-square topological distance matrix
    (zero on the diagonal).
    """
    inv_sq_dist = dist_matrix.astype(np.float64, copy=True)
    nonzero = inv_sq_dist != 0
    inv_sq_dist[nonzero] = inv_sq_dist[nonzero] ** -2
    np.fill_diagonal(inv_sq_dist, 0)

    galvez = adj_matrix.dot(inv_sq_dist)
    return galvez - galvez.T
