"""
Charged partial surface area descriptors.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import (
    CPSA_FEATURE_NAMES,
    CPSA_FEATURE_NAMES_2D,
    CPSA_FEATURE_NAMES_3D,
    MordredMolCache,
)

FEATURE_NAMES = CPSA_FEATURE_NAMES
FEATURE_NAMES_2D = CPSA_FEATURE_NAMES_2D
FEATURE_NAMES_3D = CPSA_FEATURE_NAMES_3D


def calc_2d(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute 2D CPSA descriptors without adding explicit hydrogens.
    """
    return cache.cpsa_2d_values.copy(), FEATURE_NAMES_2D


def calc_3d(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute 3D CPSA descriptors from atomic charges and surface areas.
    """
    return cache.cpsa_3d_values.copy(), FEATURE_NAMES_3D
