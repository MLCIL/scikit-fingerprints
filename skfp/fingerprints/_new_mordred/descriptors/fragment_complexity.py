import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["fragCpx"]


def calc(props: AtomicProperties) -> np.ndarray:
    """
    Compute the Mordred fragment complexity descriptor.
    """
    n_atoms = props.num_atoms
    n_bonds = props.num_bonds
    n_hetero = int(np.count_nonzero(props.is_hetero))
    value = abs(n_bonds**2 - n_atoms**2 + n_atoms) + n_hetero / 100
    return np.asarray([value], dtype=np.float32)
