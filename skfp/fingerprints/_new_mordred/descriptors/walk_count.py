"""
Walk count descriptors from unweighted adjacency matrix powers.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import (
    WALK_COUNT_FEATURE_NAMES,
    MordredMolCache,
)

FEATURE_NAMES = WALK_COUNT_FEATURE_NAMES


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred molecular walk count descriptors.
    """
    return cache.walk_count_values.copy(), FEATURE_NAMES
