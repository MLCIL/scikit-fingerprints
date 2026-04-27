"""
Polarizability descriptors.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import (
    POLARIZABILITY_FEATURE_NAMES,
    MordredMolCache,
)

FEATURE_NAMES = POLARIZABILITY_FEATURE_NAMES


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred Polarizability descriptors without adding explicit hydrogens.

    `apol` sums atom polarizability contributions from the 1994 table. `bpol`
    sums the absolute differences between polarizabilities of bonded atom pairs.
    """
    return cache.polarizability_values.copy(), FEATURE_NAMES
