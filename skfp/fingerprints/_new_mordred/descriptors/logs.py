"""
Filter-it LogS descriptor.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import LOGS_FEATURE_NAMES, MordredMolCache

FEATURE_NAMES = LOGS_FEATURE_NAMES


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute the Mordred Filter-it LogS descriptor.
    """
    return cache.logs_values.copy(), FEATURE_NAMES
