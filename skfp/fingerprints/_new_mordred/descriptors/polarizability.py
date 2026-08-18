import numpy as np
from rdkit.Chem import Mol

from skfp.fingerprints._new_mordred.utils.atomic_properties import get_polarizability

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["apol", "bpol"]


def calc(mol_hydrogens: Mol) -> np.ndarray:
    atom_polarizability = sum(
        get_polarizability(atom) for atom in mol_hydrogens.GetAtoms()
    )
    bond_polarizability = sum(
        abs(
            get_polarizability(bond.GetBeginAtom())
            - get_polarizability(bond.GetEndAtom())
        )
        for bond in mol_hydrogens.GetBonds()
    )

    values = np.array(
        [atom_polarizability, bond_polarizability],
        dtype=np.float32,
    )
    return values
