import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["nAromAtom", "nAromBond"]


def calc(props: AtomicProperties) -> tuple[np.ndarray, list[str]]:
    """
    Compute the Mordred aromatic count descriptors.

    `nAromAtom` is the number of aromatic atoms and `nAromBond` is the number of
    aromatic bonds, both taken directly from RDKit's perceived aromaticity flags.
    """
    values = [props.is_aromatic.sum(), props.bond_is_aromatic.sum()]
    return np.asarray(values, dtype=np.float32), FEATURE_NAMES
