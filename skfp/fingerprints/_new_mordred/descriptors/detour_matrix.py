"""
Detour matrix descriptors implemented with Mordred matrix attributes.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import (
    DETOUR_MATRIX_FEATURE_NAMES,
    MordredMolCache,
)

FEATURE_NAMES = DETOUR_MATRIX_FEATURE_NAMES


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred detour matrix spectral descriptors and detour index.
    """
    return cache.detour_matrix_values.copy(), FEATURE_NAMES
