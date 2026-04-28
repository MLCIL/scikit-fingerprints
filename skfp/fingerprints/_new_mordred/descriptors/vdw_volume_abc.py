"""
ABC van der Waals volume descriptor.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np

from skfp.fingerprints._new_mordred.cache import (
    VDW_VOLUME_ABC_FEATURE_NAMES,
    MordredMolCache,
)

FEATURE_NAMES = VDW_VOLUME_ABC_FEATURE_NAMES


def calc(cache: MordredMolCache) -> tuple[np.ndarray, list[str]]:
    """
    Compute the Mordred ABC van der Waals volume descriptor.

    `Vabc` uses Bondi atom volumes plus bond and SSSR ring corrections from
    the hydrogen-suppressed molecule.
    """
    return cache.vdw_volume_abc_values.copy(), FEATURE_NAMES
