import numpy as np
from rdkit.Chem.rdchem import Bond, BondType
from rdkit.Chem import Mol

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


def calc(mol_regular: Mol, mol_kekulized: Mol) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred bond count descriptors without adding explicit hydrogens.
    """
    bonds_regular = mol_regular.GetBonds()
    bonds_kekulized = mol_kekulized.GetBonds()

    n_bonds = mol_regular.GetNumBonds()
    n_bonds_s = 0
    n_bonds_d = 0
    n_bonds_t = 0
    n_bonds_a = 0
    n_bonds_m = 0

    for bond in bonds_regular:
        bond_type = bond.GetBondType()
        is_aromatic = _is_aromatic_bond(bond)

        n_bonds_s += bond_type == BondType.SINGLE
        n_bonds_d += bond_type == BondType.DOUBLE
        n_bonds_t += bond_type == BondType.TRIPLE
        n_bonds_a += is_aromatic
        n_bonds_m += is_aromatic or bond_type != BondType.SINGLE

    n_bonds_ks = 0
    n_bonds_kd = 0
    for bond in bonds_kekulized:
        bond_type = bond.GetBondType()
        n_bonds_ks += bond_type == BondType.SINGLE
        n_bonds_kd += bond_type == BondType.DOUBLE

    return np.asarray(
        [
            n_bonds,
            n_bonds,
            n_bonds_s,
            n_bonds_d,
            n_bonds_t,
            n_bonds_a,
            n_bonds_m,
            n_bonds_ks,
            n_bonds_kd,
        ],
        dtype=np.float32,
    ), FEATURE_NAMES


def _is_aromatic_bond(bond: Bond) -> bool:
    return bond.GetIsAromatic() or bond.GetBondType() == BondType.AROMATIC
