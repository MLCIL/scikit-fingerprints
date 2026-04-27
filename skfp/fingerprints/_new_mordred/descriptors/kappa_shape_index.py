"""
Kappa shape index descriptors.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import (
    KAPPA_SHAPE_INDEX_FEATURE_NAMES,
    MordredMolCache,
)

FEATURE_NAMES = KAPPA_SHAPE_INDEX_FEATURE_NAMES


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred Kappa shape index descriptors without explicit hydrogens.
    """
    return cache.kappa_shape_index_values.copy(), FEATURE_NAMES
