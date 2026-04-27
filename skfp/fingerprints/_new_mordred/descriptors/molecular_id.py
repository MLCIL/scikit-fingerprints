"""
Molecular ID descriptors implemented with Mordred formulas.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import (
    MOLECULAR_ID_FEATURE_NAMES,
    MordredMolCache,
)

FEATURE_NAMES = MOLECULAR_ID_FEATURE_NAMES


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred molecular ID descriptors.
    """
    return cache.molecular_id_values.copy(), FEATURE_NAMES
