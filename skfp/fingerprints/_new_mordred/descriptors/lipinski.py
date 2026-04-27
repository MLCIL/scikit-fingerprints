"""
Lipinski-style rule-of-five and Ghose filter descriptors.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import LIPINSKI_FEATURE_NAMES, MordredMolCache

FEATURE_NAMES = LIPINSKI_FEATURE_NAMES


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred Lipinski and Ghose filter descriptors.
    """
    return cache.lipinski_values.copy(), FEATURE_NAMES
