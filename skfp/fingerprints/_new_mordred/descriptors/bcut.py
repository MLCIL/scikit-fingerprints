"""
BCUT descriptors.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import BCUT_PROPERTIES, MordredMolCache

FEATURE_NAMES = [
    name for prop in BCUT_PROPERTIES for name in (f"BCUT{prop}-1h", f"BCUT{prop}-1l")
]


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred BCUT descriptors.
    """
    return cache.bcut_values.copy(), FEATURE_NAMES
