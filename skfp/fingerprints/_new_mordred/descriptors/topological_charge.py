"""
Topological charge descriptors implemented with Mordred graph matrices.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import (
    TOPOLOGICAL_CHARGE_FEATURE_NAMES,
    MordredMolCache,
)

FEATURE_NAMES = TOPOLOGICAL_CHARGE_FEATURE_NAMES


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred raw, mean, and global topological charge descriptors.
    """
    return cache.topological_charge_values.copy(), FEATURE_NAMES
