import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.periodic_table import MC_GOWAN_VOLUME

"""
McGowan characteristic volume descriptor.

References
    * https://doi.org/10.1007/BF02311772

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["VMcGowan"]

# every bond shrinks the volume by a fixed amount, as the bonded atoms overlap
_VOLUME_PER_BOND = 6.56


def calc(atomic_props_hydrogens: AtomicProperties) -> np.ndarray:
    """
    Compute the McGowan characteristic volume, the sum of the atomic volume
    contributions less a fixed correction per bond.

    Hydrogens contribute their own volume and their bonds count towards the
    correction, so this must be the hydrogen-explicit molecule.
    """
    atom_volumes = MC_GOWAN_VOLUME.lookup(atomic_props_hydrogens.atomic_nums)
    volume = atom_volumes.sum() - atomic_props_hydrogens.num_bonds * _VOLUME_PER_BOND

    return np.asarray([volume], dtype=np.float32)
