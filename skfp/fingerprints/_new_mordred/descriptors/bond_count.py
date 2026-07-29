import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.rdchem import BondType

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

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


def calc(
    atomic_props_hydrogens: AtomicProperties, mol_kekulized_hydrogens: Mol
) -> np.ndarray:
    """
    Bond count descriptors.

    Counts bonds by type: all bonds, heavy-atom-only bonds, single, double, triple,
    aromatic, multiple, and kekulized single/double bonds.

    Following the original Mordred implementation, nBonds (any) and nBondsS (single)
    are computed on the hydrogen-explicit molecule, while nBondsO (heavy) counts only
    bonds between non-hydrogen atoms. nBondsKS and nBondsKD use the kekulized
    hydrogen-explicit molecule, where aromatic bonds are expressed as alternating
    single and double bonds.
    """
    bond_types = atomic_props_hydrogens.bond_types
    is_heavy = ~atomic_props_hydrogens.is_hydrogen
    is_single = bond_types == BondType.SINGLE
    is_aromatic = atomic_props_hydrogens.bond_is_aromatic | (
        bond_types == BondType.AROMATIC
    )

    n_bonds = atomic_props_hydrogens.num_bonds
    n_bonds_o = np.count_nonzero(
        is_heavy[atomic_props_hydrogens.bond_begin_idxs]
        & is_heavy[atomic_props_hydrogens.bond_end_idxs]
    )
    n_bonds_s = np.count_nonzero(is_single)
    n_bonds_d = np.count_nonzero(bond_types == BondType.DOUBLE)
    n_bonds_t = np.count_nonzero(bond_types == BondType.TRIPLE)
    n_bonds_a = np.count_nonzero(is_aromatic)
    n_bonds_m = np.count_nonzero(is_aromatic | ~is_single)

    kekulized_bond_types = np.fromiter(
        (bond.GetBondType() for bond in mol_kekulized_hydrogens.GetBonds()),
        dtype=np.intp,
        count=mol_kekulized_hydrogens.GetNumBonds(),
    )
    n_bonds_ks = np.count_nonzero(kekulized_bond_types == BondType.SINGLE)
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
