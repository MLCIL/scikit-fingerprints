"""
Constitutional descriptors.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import (
    CONSTITUTIONAL_FEATURE_NAMES,
    MordredMolCache,
)

FEATURE_NAMES = CONSTITUTIONAL_FEATURE_NAMES


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred Constitutional descriptors without adding explicit hydrogens.
    """
    return cache.constitutional_values.copy(), FEATURE_NAMES
