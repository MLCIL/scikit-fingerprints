import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["apol", "bpol"]


def calc(atomic_props_hydrogens: AtomicProperties) -> tuple[np.ndarray, list[str]]:
    polarizabilities = atomic_props_hydrogens.get("polarizability")
    atom_polarizability = polarizabilities.sum()
    bond_polarizability = np.abs(
        polarizabilities[atomic_props_hydrogens.bond_begin_idxs]
        - polarizabilities[atomic_props_hydrogens.bond_end_idxs]
    ).sum()

    values = np.array(
        [atom_polarizability, bond_polarizability],
        dtype=np.float32,
    )
    return values, FEATURE_NAMES
