import numpy as np
from rdkit.Chem import Mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["fragCpx"]

def calc(mol: Mol) -> tuple[np.ndarray, list[str]]:
    """
    Compute the Mordred fragment complexity descriptor.
    """
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()
    n_hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 6)
    value = abs(n_bonds**2 - n_atoms**2 + n_atoms) + n_hetero / 100
    return np.asarray([value], dtype=np.float32), FEATURE_NAMES