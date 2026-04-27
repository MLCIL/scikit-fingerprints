"""
Extended Topochemical Atom (ETA) descriptors.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import (
    EXTENDED_TOPOCHEMICAL_ATOM_FEATURE_NAMES,
    MordredMolCache,
)

FEATURE_NAMES = EXTENDED_TOPOCHEMICAL_ATOM_FEATURE_NAMES


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred ETA descriptors on a no-explicit-H, kekulized molecule.
    """
    return cache.extended_topochemical_atom_values.copy(), FEATURE_NAMES
