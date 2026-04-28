"""
Vertex adjacency information descriptor.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import (
    VERTEX_ADJACENCY_INFORMATION_FEATURE_NAMES,
    MordredMolCache,
)

FEATURE_NAMES = VERTEX_ADJACENCY_INFORMATION_FEATURE_NAMES


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute the Mordred vertex adjacency information descriptor.

    `VAdjMat` is ``1 + log2(m)``, where ``m`` is the number of heavy-heavy
    bonds. Molecules with no heavy-heavy bonds return NaN.
    """
    return cache.vertex_adjacency_information_values.copy(), FEATURE_NAMES
