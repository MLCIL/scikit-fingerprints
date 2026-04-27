"""
Barysz matrix descriptors implemented with Mordred matrix attributes.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import (
    BARYSZ_ATTRIBUTES,
    BARYSZ_PROPERTIES,
    MordredMolCache,
)

FEATURE_NAMES = [
    f"{attr}_Dz{prop}" for prop in BARYSZ_PROPERTIES for attr in BARYSZ_ATTRIBUTES
]


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred Barysz matrix spectral descriptors.
    """
    return cache.barysz_values.copy(), FEATURE_NAMES
