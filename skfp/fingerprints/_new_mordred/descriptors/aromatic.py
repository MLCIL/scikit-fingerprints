import numpy as np
from rdkit.Chem import Mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["nAromAtom", "nAromBond"]


def calc(mol: Mol) -> tuple[np.ndarray, list[str]]:
    """
    Compute the Mordred aromatic count descriptors.

    `nAromAtom` is the number of aromatic atoms and `nAromBond` is the number of
    aromatic bonds, both taken directly from RDKit's perceived aromaticity flags.
    """
    values = [
        sum(atom.GetIsAromatic() for atom in mol.GetAtoms()),
        sum(bond.GetIsAromatic() for bond in mol.GetBonds()),
    ]

    return np.asarray(values, dtype=np.float32), FEATURE_NAMES
