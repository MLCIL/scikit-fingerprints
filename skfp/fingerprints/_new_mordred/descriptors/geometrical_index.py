"""
Geometrical index descriptors.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import (
    GEOMETRICAL_INDEX_FEATURE_NAMES,
    MordredMolCache,
)

FEATURE_NAMES = GEOMETRICAL_INDEX_FEATURE_NAMES


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred geometrical index descriptors from 3D coordinates.
    """
    return cache.geometrical_index_values.copy(), FEATURE_NAMES
