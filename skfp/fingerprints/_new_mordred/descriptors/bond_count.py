"""
Bond count descriptors implemented with direct RDKit bond access.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import MordredMolCache

FEATURE_NAMES = [
    "nBonds",
    "nBondsO",
    "nBondsS",
    "nBondsD",
    "nBondsT",
    "nBondsA",
    "nBondsM",
    "nBondsKS",
    "nBondsKD",
]


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred bond count descriptors without adding explicit hydrogens.
    """
    return cache.bond_count_values.copy(), FEATURE_NAMES
