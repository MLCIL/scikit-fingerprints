"""
Path count descriptors implemented with RDKit bond paths.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import (
    PATH_COUNT_FEATURE_NAMES,
    MordredMolCache,
)

FEATURE_NAMES = PATH_COUNT_FEATURE_NAMES


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred molecular path count descriptors.
    """
    return cache.path_count_values.copy(), FEATURE_NAMES
