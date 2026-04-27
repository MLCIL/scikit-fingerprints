"""
EState atom type descriptors implemented with RDKit EState functions.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import (
    ESTATE_ATOM_TYPES as CACHE_ESTATE_ATOM_TYPES,
)
from skfp.fingerprints._new_mordred.cache import (
    ESTATE_FEATURE_NAMES,
    MordredMolCache,
)

ESTATE_ATOM_TYPES = CACHE_ESTATE_ATOM_TYPES
FEATURE_NAMES = ESTATE_FEATURE_NAMES


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred EState atom type count, sum, max, and min descriptors.
    """
    return cache.estate_values.copy(), FEATURE_NAMES
