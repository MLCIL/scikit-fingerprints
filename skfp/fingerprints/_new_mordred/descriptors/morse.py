"""
3D-MoRSE descriptors.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import MORSE_FEATURE_NAMES, MordredMolCache

FEATURE_NAMES = MORSE_FEATURE_NAMES


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred 3D-MoRSE descriptors from explicit-hydrogen 3D coordinates.
    """
    return cache.morse_values.copy(), FEATURE_NAMES
