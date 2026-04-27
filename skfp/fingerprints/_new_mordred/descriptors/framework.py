"""
Molecular framework ratio descriptor.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import (
    FRAMEWORK_FEATURE_NAMES,
    MordredMolCache,
)

FEATURE_NAMES = FRAMEWORK_FEATURE_NAMES


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute the Mordred molecular framework ratio descriptor.
    """
    return cache.framework_values.copy(), FEATURE_NAMES
