import numpy as np
from rdkit.Chem.rdchem import BondType

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = [
    "number of bonds",
    "number of heavy-atom bonds",
    "number of single bonds",
    "number of double bonds",
    "number of triple bonds",
    "number of aromatic bonds",
    "number of multiple bonds",
    "number of Kekule single bonds",
    "number of Kekule double bonds",
]


def calc(
    props_regular: AtomicProperties,
    props_hydrogens: AtomicProperties,
    kekulized_bond_types: np.ndarray,
    kekulized_bond_types_hydrogens: np.ndarray,
) -> np.ndarray:
    """
    Bond count descriptors.

    Counts bonds by type: all bonds, heavy-atom-only bonds, single, double, triple,
    aromatic, multiple, and kekulized single/double bonds.

    Each count uses the molecule the original Mordred implementation uses for it:
    all bonds, single and Kekule single bonds the hydrogen-explicit one, and every
    other count the hydrogen-suppressed one, where the non-H bond count takes all of
    its bonds. The Kekule single and double counts use the kekulized molecules,
    where aromatic bonds are expressed as alternating single and double bonds.
    """
    bond_types = props_regular.bond_types
    is_aromatic = props_regular.bond_is_aromatic | (bond_types == BondType.AROMATIC)

    n_bonds = props_hydrogens.num_bonds
    n_bonds_o = props_regular.num_bonds
    n_bonds_s = np.count_nonzero(props_hydrogens.bond_types == BondType.SINGLE)
    n_bonds_d = np.count_nonzero(bond_types == BondType.DOUBLE)
    n_bonds_t = np.count_nonzero(bond_types == BondType.TRIPLE)
    n_bonds_a = np.count_nonzero(is_aromatic)
    n_bonds_m = np.count_nonzero(is_aromatic | (bond_types != BondType.SINGLE))

    n_bonds_ks = np.count_nonzero(kekulized_bond_types_hydrogens == BondType.SINGLE)
    n_bonds_kd = np.count_nonzero(kekulized_bond_types == BondType.DOUBLE)

    return np.asarray(
        [
            n_bonds,
            n_bonds_o,
            n_bonds_s,
            n_bonds_d,
            n_bonds_t,
            n_bonds_a,
            n_bonds_m,
            n_bonds_ks,
            n_bonds_kd,
        ],
        dtype=np.float32,
    )
