"""
Molecular distance edge descriptors implemented with Mordred formulas.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import (
    MOLECULAR_DISTANCE_EDGE_FEATURE_NAMES,
    MordredMolCache,
)

FEATURE_NAMES = MOLECULAR_DISTANCE_EDGE_FEATURE_NAMES


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred molecular distance edge descriptors.
    """
    return cache.molecular_distance_edge_values.copy(), FEATURE_NAMES
