"""
Distance matrix descriptors implemented with Mordred matrix attributes.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import (
    DISTANCE_MATRIX_FEATURE_NAMES,
    MordredMolCache,
)

FEATURE_NAMES = DISTANCE_MATRIX_FEATURE_NAMES


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred topological distance matrix spectral descriptors.
    """
    return cache.distance_matrix_values.copy(), FEATURE_NAMES
