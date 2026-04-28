"""
Topological index descriptors.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import (
    TOPOLOGICAL_INDEX_FEATURE_NAMES,
    MordredMolCache,
)

FEATURE_NAMES = TOPOLOGICAL_INDEX_FEATURE_NAMES


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred topological diameter, radius, and shape indices.

    The descriptors use RDKit topological distances without bond orders, atom
    weights, or explicit hydrogens. Disconnected molecules keep RDKit's large
    disconnected-distance sentinel values, matching Mordred.
    """
    return cache.topological_index_values.copy(), FEATURE_NAMES
